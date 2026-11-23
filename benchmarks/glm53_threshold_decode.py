"""Benchmark warmed serving decode graphs on captured trained GLM MLA latents."""

import argparse
import json
import random
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch

from lod_attention.kernels import glm_threshold_decode
from vllm_lod_plugin.config import VLLMLODSettings
from vllm_lod_plugin.pool import VLLMLayerLODPool


def reference(pool, q, new, initial, c, audit_masks, *, packed=None, page_size=None):
    p = pool.state["page_cache"]
    arena = p["unified_page1_k"]
    counts = pool.state["counts"]
    lengths = p["slot_lengths"]
    fixed = p["unified_page1_fixed_indices"]
    offsets = p["unified_page1_fixed_slot_offsets"]
    sink = pool.state["sink_k"].size(2)
    start = pool.state_capacity + sink + pool.decode_local_limit
    errors = []
    ref_lses = []
    stats = []
    for b, slot in enumerate(pool.active_indices.tolist()):
        n = int(pool.state_lens[slot])
        z = counts[slot, 0, :n, 0].float()
        m = p["unified_page1_coarse_offset"] + slot * pool.state_capacity
        means = arena[m : m + n].float()
        Q = q[b].float()
        score = (Q @ means.T) * pool.engine.scaling + z.log()
        mask = (score.softmax(-1) > c / (z > 0).sum()) & (lengths[slot, 0, :n] <= 1024)
        if getattr(pool, "_glm_threshold_tuning", {}).get("union"):
            mask = mask.any(0, keepdim=True).expand_as(mask)
        accepted = mask.sum(-1)
        sizes = lengths[slot, 0, :n].long()
        stats.append(
            (
                accepted.float().mean().item(),
                (mask.float() @ sizes.float()).mean().item(),
                (mask.any(0).float() @ sizes.float()).item(),
            )
        )
        ends = offsets[slot, 0, 1 : n + 1].long()
        owners = torch.arange(n, device=q.device).repeat_interleave(sizes)
        remote = arena[fixed[slot, 0, start : start + int(ends[-1])].long()].float()
        local = pool.state["recent_k"][
            slot, 0, : min(int(initial[slot]), pool.decode_local_limit)
        ].float()
        tail = torch.cat([local, new[b].float(), pool.state["sink_k"][slot, 0].float()])
        s1 = ((Q @ remote.T) * pool.engine.scaling).masked_fill(~mask[:, owners], -torch.inf)
        s2 = score.masked_fill(mask, -torch.inf)
        s3 = Q @ tail.T * pool.engine.scaling
        log_z = torch.logsumexp(torch.cat([s1, s2, s3], 1), 1)
        value = (
            torch.exp(s1 - log_z[:, None]) @ remote
            + torch.exp(s2 - log_z[:, None]) @ means
            + torch.exp(s3 - log_z[:, None]) @ tail
        )
        errors.append(value)
        ref_lses.append(log_z)
        if hasattr(pool, "_glm_threshold_buffers"):
            buf = pool._glm_threshold_buffers
            bits = audit_masks[b, 0].flatten()[:n]
            actual = (bits[None, :] & (1 << torch.arange(q.size(1), device=q.device)[:, None])) != 0
            assert torch.equal(actual, mask), (actual != mask).sum().item()
            size = int(buf["total"][b, 0])
            expected_ids = fixed[slot, 0, start : start + owners.numel()][mask.any(0)[owners]]
            expected_bits = (
                mask.int() * (1 << torch.arange(q.size(1), device=q.device)[:, None])
            ).sum(0)[owners][mask.any(0)[owners]]
            expected_ids, order = expected_ids.sort()
            width = (32 if q.size(0) == 1 else 8) if page_size is None else page_size
            indices, membership = packed if packed is not None else (buf["flat"], buf["bits"])
            if packed is None and getattr(pool, "_glm_threshold_tuning", {}).get("half_masks"):
                membership = membership.view(torch.int16).flatten()[:membership.numel()].view_as(membership)
            desc = indices[b, 0, :size].long()
            flags = membership[b, 0, :size].int() & 65535
            config = getattr(pool, "_glm_threshold_tuning", {})
            if config.get("ordered"):
                ng = (pool.state_capacity + config.get("selector", 64) - 1) // config.get("selector", 64)
                counts_by_tile = buf["ordered_count"].flatten()[b * ng:(b + 1) * ng]
                bases = buf["ordered_starts"].flatten()[b * ng:(b + 1) * ng]
                positions = torch.cat([torch.arange(int(base), int(base) + int(count), device=q.device)
                                       for base, count in zip(bases, counts_by_tile)])
                desc = buf["ordered_ids"][b, 0, positions].long()
                flags = buf["ordered_bits"][b, 0, positions]
            if width:
                j = torch.arange(width, device=q.device)
                live = j[None, :] <= desc[:, None] % width
                offsets_in_directory = (desc[:, None] // width + j[None, :])[live]
                leaf_ids = fixed[slot, 0, start + offsets_in_directory]
                per_leaf_bits = flags[:, None].expand_as(live)[live]
            else:
                leaf_ids, per_leaf_bits = desc, flags
            ids, order2 = leaf_ids.sort()
            assert torch.equal(ids, expected_ids) and torch.equal(
                per_leaf_bits[order2], expected_bits[order]
            )
    return torch.stack(errors), torch.stack(ref_lses), torch.tensor(stats).mean(0).tolist()


@torch.inference_mode()
def run(a):
    data = torch.load(a.capture, map_location="cpu", weights_only=True)
    batch = a.batch
    q = data["mla_absorbed_queries"][-batch:].cuda().contiguous().bfloat16()
    heads = q.size(1)
    assert heads == 16
    rows = torch.arange(batch - 1, -1, -1, device="cuda", dtype=torch.int32)
    layer = SimpleNamespace(
        num_heads=heads,
        num_kv_heads=1,
        head_size=512,
        head_size_v=512,
        kv_lora_rank=512,
        scale=data["mla_scale"],
        _vllm_lod_absorbed_mla=True,
        _vllm_lod_glm53=True,
    )
    pool = VLLMLayerLODPool(
        layer,
        settings=VLLMLODSettings.production(mode="two-tier", pool_size=batch),
        max_requests=batch,
        request_capacity=67584,
        active_indices=rows,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    ctx = data["query_positions"][-batch:].long()
    latents = data["mla_latents"].cuda().bfloat16()
    for i, row in enumerate(rows.tolist()):
        prefix = latents[: int(ctx[i])][None, None].contiguous()
        pool.install(row, pool.engine.build_cache_from_bf16(prefix, prefix))
    pool.active_decode_rows = rows.tolist()
    pool._refresh_unified_page1_coarse(tuple(range(batch)))
    pool.ensure_unified_page1_fixed(tuple(range(batch)))
    current = latents[ctx.cuda()].unsqueeze(1)
    out = torch.empty(heads, batch, 512, dtype=q.dtype, device=q.device).transpose(0, 1)
    init = pool.local_lens.clone()
    results = {}
    graphs = {}
    torch.set_float32_matmul_precision("highest")
    if a.compare_packing:
        from benchmarks.glm53_flat_pack import compare
        return compare(a, data, ctx, pool, q, current, out, init)
    cases = [("top8", None), ("c16", 16.0), ("c8", 8.0), ("c32", 32.0)]
    schedules = {"original": (False, False, False), "raw": (False, True, False),
                 "combined": (True, False, False), "combined_raw": (True, True, False),
                 "contiguous": (False, False, True), "raw_contiguous": (False, True, True),
                 "combined_raw_contiguous": (True, True, True)}
    if a.transfer_cases:
        cases = [(name, 16.) for name in a.transfer_cases]
    if a.compare_schedules:
        cases = [(name,16.) for name in (a.schedules or schedules)]
    for name, c in cases:
        if a.transfer_cases:
            pool._glm_threshold_execution = (batch == 1, True, batch == 1)
            pool._glm_threshold_tuning = a.transfer_cases[name]
        if a.compare_schedules:
            pool._glm_threshold_execution = schedules[name]
        pool.glm_decode_threshold = c

        def step():
            pool.local_lens.copy_(init)
            return pool.decode(q, current, current, SimpleNamespace(num_actual_tokens=batch), out)

        if a.transfer_cases:
            from triton.runtime.errors import OutOfResources
            try:
                step()
            except OutOfResources as error:
                results[name] = {"unsupported": str(error)}
                continue
        else:
            step()
        got = out.clone().float()
        assert torch.equal(pool.local_lens, init + 1)
        if c:
            audit_masks = torch.empty(
                batch, 1, (pool.state_capacity + 63) // 64, 64, device=q.device, dtype=torch.int32
            )
            pool.local_lens.copy_(init)
            _, got_lse = glm_threshold_decode.attention(
                pool, q, current, out, c, mask_audit=audit_masks
            )
            ref, lse, work = reference(pool, q, current, init, c, audit_masks)
            torch.testing.assert_close(got, ref, atol=0.035, rtol=0.025)
            torch.testing.assert_close(got_lse, lse, atol=0.0005, rtol=0.0002)
            results[name] = dict(
                open_regions=work[0],
                exact_leaves=work[1],
                union_loaded_leaves=work[2],
                max_abs=(got - ref).abs().max().item(),
                relative_l2=((got - ref).norm() / ref.norm()).item(),
            )
        else:
            scratch = pool._dcp_buffers(q.unsqueeze(2), batch)
            stamps = scratch["gqa_union_seen_stamps"][:batch]
            epochs = scratch["gqa_union_epochs"][:batch].reshape(batch, 1)
            opened = stamps == epochs
            lens = pool.state["page_cache"]["slot_lengths"][rows.long(), 0, :]
            leaves = (opened * lens).sum(-1).float().mean().item()
            results[name] = dict(
                union_loaded_leaves=leaves,
                exact_leaves=leaves,
                open_regions=opened.sum(-1).float().mean().item(),
            )
        for i, row in enumerate(rows.tolist()):
            assert torch.equal(pool.state["recent_k"][row, 0, int(init[row])], current[i, 0])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        if a.transfer_cases:
            graph.replay()
            torch.testing.assert_close(out.float(), ref, atol=0.035, rtol=0.025)
        graphs[name] = graph
        results.setdefault(name, {})["timings_us"] = []
    rand = random.Random(a.pack_seed)
    for round_index in range(11):
        order = list(graphs)
        rand.shuffle(order)
        for name in order:
            graph = graphs[name]
            graph.replay()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(20):
                graph.replay()
            end.record()
            end.synchronize()
            results[name]["timings_us"].append(start.elapsed_time(end) * 1000 / 20)
    if a.profile_kernels:
        for name, graph in graphs.items():
            graph.replay()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(50):
                    graph.replay()
                # Leave a trailing device event for ROCTracer to flush before stopping.
                out.zero_()
                torch.cuda.synchronize()
            kernels = {}
            for event in prof.events():
                if event.device_type == torch.autograd.DeviceType.CUDA:
                    for kernel in ("_score_regions", "_select_and_pack", "_attend_exact", "_merge_fields", "_write_cutoff"):
                        if kernel in event.name:
                            kernels.setdefault(kernel, []).append(event.device_time_total)
                            break
            if not all(40 <= len(kernels.get(k, [])) <= 50 for k in
                       ("_score_regions", "_select_and_pack", "_attend_exact", "_merge_fields")):
                raise RuntimeError(f"Incomplete GPU profile: {[(k, len(v)) for k,v in kernels.items()]}")
            results[name]["profile_samples"] = {k: len(v) for k, v in kernels.items()}
            results[name]["kernel_medians_us"] = {k: statistics.median(v) for k, v in kernels.items()}
    for row in results.values():
        if "timings_us" not in row:
            continue
        row["median_us"] = statistics.median(row["timings_us"])
    result = dict(
        scope="one TP4-rank MLA serving pool, warmed graph; saved final-prefill queries used as decode inputs",
        source=str(a.capture),
        layer=data["module"],
        batch=batch,
        lengths=ctx.tolist(),
        initial_local=init.tolist(),
        regions=pool.state_lens.tolist(),
        heads=heads,
        gpu=torch.cuda.get_device_name(),
        transfer_cases=a.transfer_cases,
        results=results,
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print({name: round(x["median_us"], 2) if "median_us" in x else x.get("unsupported") for name, x in results.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-kernels", action="store_true")
    parser.add_argument("--transfer-cases", type=json.loads)
    parser.add_argument("--compare-packing", action="store_true")
    parser.add_argument("--pack-deferred", action="store_true", help="Use the optimized deferred-merge schedule")
    parser.add_argument("--pack-seed", type=int, default=8347)
    parser.add_argument("--pack-variants", nargs="+", help="Restrict packing comparisons by variant name")
    parser.add_argument("--compare-schedules", action="store_true")
    parser.add_argument("--schedules", nargs="+")
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--batch", type=int, choices=(1, 8), required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
