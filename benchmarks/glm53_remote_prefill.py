"""TP1 NoPE MLA remote-only prefill diagnostic, with a real LoD cache.

Build a 16K history before timing a separate query slab. There is no exact
prefix shortcut or concurrent local-attention kernel in the measured region.
Inputs are random, as in the reduced GLM integration fixture: this tests
kernel geometry/correctness, not trained-model quality or end-to-end speed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import lod_attention._core as core

from benchmarks._vllm import write_json
from lod_attention._config import LODConfig, LODMode, ModelFamily
from lod_attention._engines import KernelTwoLevelLODAttention
from lod_attention._profile import configure_engine
from lod_attention.kernels.aiter_mla_prefill_attention import aiter_mla_prefill_route_coarse_attention


def elapsed(call):
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    value = call()
    end.record()
    end.synchronize()
    return value, begin.elapsed_time(end)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history", type=int, default=16384)
    p.add_argument("--queries", type=int, default=16384)
    p.add_argument("--profile-refinement", action="store_true",
                   help="Separately record dispatch and leaf-kernel GPU intervals.")
    p.add_argument("--compare-page-lookups", action="store_true",
                   help="Compare repeated per-leaf versus scalar per-page directory lookups.")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.manual_seed(47)
    engine = KernelTwoLevelLODAttention(LODConfig(), query_heads=64,
                                       key_value_heads=1, scale=0.0625).cuda().eval()
    engine.head_dim = 512
    configure_engine(engine, family=ModelFamily.GLM53_FLASH, mode=LODMode.TWO_TIER,
        request_capacity=args.history + args.queries, has_query_norm=True, has_key_norm=False)
    engine._lod_prefill_attention_buffers = {}
    history = (torch.randn(1, 1, args.history, 512, device="cuda") * 0.4).bfloat16()
    cache = engine._build_cache_from_bf16(history, history, finalize_cache_for_decode=False)
    pages, slots = cache["page_cache"], int(cache["state_len"])
    # The standalone builder allocates K/V separately; the GLM vLLM pool
    # aliases them because both are the same latent. Verify that identity
    # before reproducing the real dispatch/storage geometry in this probe.
    torch.testing.assert_close(cache["state_k"], cache["state_v"], atol=0, rtol=0)
    leaf_count = int(pages["leaf_count"])
    torch.testing.assert_close(pages["leaf_k"][..., :leaf_count, :],
                               pages["leaf_v"][..., :leaf_count, :], atol=0, rtol=0)
    cache["state_v"] = cache["state_k"]
    pages["leaf_v"] = pages["leaf_k"]
    q = (torch.randn(1, 64, args.queries, 512, device="cuda") * 0.3).bfloat16()
    result = dict(scope="TP1 remote-only kernel diagnostic; random inputs; not model timings",
                  history=args.history, queries=args.queries, slots=slots,
                  policy="top-eight; oversized selected regions remain closed above 1024 leaves",
                  shared_latent=True,
                  candidates=[])

    def route():
        selected, coarse, _, _ = aiter_mla_prefill_route_coarse_attention(
            q, cache["state_k"].contiguous(), cache["state_v"].contiguous(),
            cache["counts"].contiguous(), state_len=slots, kv_group_size=64,
            scale=0.0625, normalize_route_query=False,
            buffers=engine._lod_prefill_attention_buffers)
        lengths = pages["slot_lengths"][..., :slots].expand(1, 64, slots)
        selected_lengths = lengths.unsqueeze(2).expand(1, 64, args.queries, slots).gather(-1, selected)
        selected.masked_fill_(selected_lengths > 1024, -1)
        return selected, coarse

    route()  # Untimed allocation/JIT warmup.
    (selected, coarse), result["route_coarse_ms"] = elapsed(route)
    positions = sorted({0, min(37, args.queries - 1), args.queries - 1})
    counts = cache["counts"][..., :slots, :].clamp_min(1)
    means = (cache["state_k"][..., :slots, :].float() / counts).bfloat16().float()
    scores = q[:, :, positions].float() @ means.transpose(-1, -2) * 0.0625
    scores += counts.squeeze(-1).log().unsqueeze(2)
    torch.testing.assert_close(coarse.output_0.permute(0, 2, 1, 3)[:, :, positions].float(),
                               scores.softmax(-1) @ means, atol=0.004, rtol=0.025)
    torch.testing.assert_close(coarse.lse_0[:, :, positions], scores.logsumexp(-1),
                               atol=0.003, rtol=0.003)
    expected = scores.argsort(dim=-1, descending=True, stable=True)[..., :8]
    lengths = pages["slot_lengths"][..., :slots].expand(1, 64, slots)
    closed = lengths.unsqueeze(2).expand(1, 64, len(positions), slots).gather(-1, expected) > 1024
    assert torch.equal(selected[:, :, positions], expected.masked_fill(closed, -1))
    result["coarse_reference"] = "FP32 output/LSE and top-eight checked at three query positions"
    reference_output = reference_lse = None
    original_leaf_attention = core.paged_leaf_attention
    candidates = ((32, 16, 2, True), (16, 16, 2, True), (64, 16, 2, True),
                  (32, 16, 4, True), (64, 16, 4, True))
    if args.compare_page_lookups:
        candidates = ((32, 16, 2, False), (32, 16, 2, True))
    for bm, bn, waves, scalar_lookup in candidates:
        entry = dict(block_m=bm, block_n=bn, waves=waves, scalar_page_lookup=scalar_lookup)
        def leaf_attention(*args, **kwargs):
            kwargs["scalar_page_lookup"] = scalar_lookup
            return original_leaf_attention(*args, **kwargs)
        core.paged_leaf_attention = leaf_attention
        engine.leaf_block_m, engine.leaf_block_n, engine.leaf_num_warps = bm, bn, waves
        try:
            def fine():
                return engine._paged_leaf_attention(q, selected, pages,
                    active_slots=slots, reduce_routes=False)
            output, lse = fine()  # Untimed allocation/JIT warmup for this tile.
            torch.cuda.synchronize()
            small_output, small_lse = output[:, :, positions], lse[:, :, positions]
            if reference_output is None:
                reference_output, reference_lse = small_output.clone(), small_lse.clone()
            else:
                valid = reference_lse.isfinite()
                torch.testing.assert_close(small_lse, reference_lse, atol=0.003, rtol=0.003)
                torch.testing.assert_close(small_output[valid], reference_output[valid],
                                           atol=0.004, rtol=0.025)
            _, entry["refinement_ms"] = elapsed(fine)
            if args.profile_refinement:
                engine._lod_leaf_timing_events = {}
                fine()
                torch.cuda.synchronize()
                entry["phases_ms"] = {
                    name: sum(begin.elapsed_time(end) for begin, end in events)
                    for name, events in engine._lod_leaf_timing_events.items()
                }
                del engine._lod_leaf_timing_events
            entry["status"] = "complete"
        except Exception as exc:
            entry.update(status="failed", error=str(exc))
        finally:
            core.paged_leaf_attention = original_leaf_attention
            if hasattr(engine, "_lod_leaf_timing_events"):
                del engine._lod_leaf_timing_events
        result["candidates"].append(entry)
        print(json.dumps(entry), flush=True)
        write_json(args.output, result)


if __name__ == "__main__":
    main()
