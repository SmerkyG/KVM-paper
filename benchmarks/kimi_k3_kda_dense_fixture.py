"""Four real K3 attention layers: three KDA, one dense MLA, TP8, no FFNs.

Same seeded weights, inputs, conv, gates, norms and TP communication for both
variants. Only chunk KDA changes. This is fixture performance/numerical
evidence, never a full trained-K3 speed or perplexity result.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zlib
from pathlib import Path
from types import MethodType


def run_fixture(worker, length, groups, direct_state=False):
    import torch

    with torch.inference_mode():
        return _run_fixture(worker, length, groups, direct_state)


def _run_fixture(worker, length, groups, direct_state):
    import torch
    from aiter import flash_attn_varlen_func
    from vllm.distributed import get_tp_group
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.models.kimi_k3.amd import kda as kda_module
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
    from benchmarks.kimi_k3_kda_upstream_probe import candidate, check

    group = get_tp_group()
    rank = group.rank_in_group
    core = next(m for m in worker.model_runner.model.modules()
                if type(m).__name__ == "KimiLinearModel")
    layers = list(core.layers)
    assert len(layers) == 4 and group.world_size == 8
    assert [type(x.self_attn).__name__ for x in layers[:3]] == ["KimiK3DeltaAttention"] * 3
    assert all(isinstance(x.mlp, torch.nn.Identity) for x in layers)

    with torch.inference_mode():
        for name, module in core.named_modules():
            for pname, param in module.named_parameters(recurse=False):
                partitioned = any(getattr(module, a + "_size_per_partition", None)
                                  not in (None, getattr(module, a + "_size", None))
                                  for a in ("input", "output"))
                generator = torch.Generator(device="cuda").manual_seed(
                    zlib.adler32((name + "/" + pname).encode()) + (rank if partitioned else 0))
                if "norm" in name.lower() and param.ndim == 1:
                    param.fill_(1)
                elif pname in ("A_log", "dt_bias"):
                    param.zero_()
                elif param.ndim >= 2:
                    param.normal_(0, float(getattr(module, "input_size", param.shape[-1]))**-0.5,
                                  generator=generator)
                else:
                    param.zero_()
        for layer in layers[:3]:
            obj = layer.self_attn
            offset = 4 * obj.local_projection_size
            gen = torch.Generator(device="cuda").manual_seed(3000 + layer.layer_idx)
            obj.in_proj_qkvgfab.weight[offset:offset + 128].normal_(0, 7168**-0.5, generator=gen)
            if obj.in_proj_padding:
                obj.in_proj_qkvgfab.weight[-obj.in_proj_padding:].zero_()
            conv = torch.zeros(2, 3 * obj.local_projection_size, 3, dtype=torch.bfloat16, device="cuda")
            if not is_conv_state_dim_first():
                conv = conv.transpose(-1, -2).contiguous()
            obj.kv_cache = (conv, torch.zeros(2, 12, 128, 128, dtype=torch.float32, device="cuda"))

    # The MLA control stays full attention, with native 192-d QK/128-d V and
    # all Q/K/V/O projections. No LoD state/routing exists in either variant.
    wrapper = layers[3].self_attn.mla_attn
    original_attention = layers[3]._run_self_attn
    cu = torch.tensor([0, length], dtype=torch.int32, device="cuda")
    def dense_mla(self, positions, hidden):
        qkv = wrapper.fused_qkv_a_proj(hidden)[0]
        qc, kv = qkv.split((1536, 576), dim=-1)
        latent, direct = kv.split((512, 64), dim=-1)
        qc, latent = wrapper._normalize_q_kv(qc, latent)
        q = wrapper.q_b_proj(qc)[0].view(length, 12, 192)
        projected = wrapper.kv_b_proj(latent)[0].view(length, 12, 256)
        k = torch.cat((projected[..., :128], direct[:, None, :].expand(-1, 12, -1)), -1)
        out = flash_attn_varlen_func(q, k, projected[..., 128:], cu, cu,
            length, length, softmax_scale=192**-0.5, causal=True, return_lse=False)
        out = out.reshape(length, -1) * wrapper.g_proj(hidden)[0].sigmoid()
        return wrapper.o_proj(out)[0]
    layers[3]._run_self_attn = MethodType(dense_mla, layers[3])

    cu_cpu = torch.tensor([0, length], dtype=torch.int32)
    nums, bp, tp = compute_causal_conv1d_metadata(cu_cpu,
        device=torch.device("cuda", torch.cuda.current_device()))
    ci = torch.tensor([[0, c] for c in range((length + 63) // 64)], dtype=torch.int32, device="cuda")
    co = torch.tensor([0, (length + 63) // 64], dtype=torch.int64, device="cuda")
    meta = GDNAttentionMetadata(num_prefills=1, num_prefill_tokens=length,
        num_decodes=0, num_decode_tokens=0, num_spec_decodes=0, num_spec_decode_tokens=0,
        num_actual_tokens=length, has_initial_state=torch.tensor([False], device="cuda"),
        non_spec_query_start_loc=cu,
        non_spec_state_indices_tensor=torch.tensor([1], dtype=torch.int32, device="cuda"),
        chunk_indices=ci, chunk_offsets=co, nums_dict=nums, batch_ptr=bp, token_chunk_offset_ptr=tp)
    metadata = {layer.self_attn.prefix: meta for layer in layers[:3]}
    gen = torch.Generator(device="cuda").manual_seed(1234)
    hidden = torch.randn(length, 7168, dtype=torch.bfloat16, device="cuda", generator=gen)
    positions = torch.arange(length, device="cuda")
    original_kda = kda_module.chunk_kda_prefill
    context = {"direct_state": False}

    def patched(*, q, k, v, raw_g, raw_beta, A_log, g_bias, **kwargs):
        if kwargs.get("checkpoint_offsets") is not None:
            raise NotImplementedError("experiment does not export checkpoint snapshots")
        from vllm.model_executor.layers.mamba.ops.gather_initial_states import gather_initial_states

        cache, indices = kwargs["state_cache"], kwargs["state_indices"]
        inp = dict(q=q, k=k, v=v, g=raw_g, beta=raw_beta, A_log=A_log,
            dt_bias=g_bias, cu_seqlens=kwargs["cu_seqlens"],
            chunk_indices=kwargs["chunk_indices"], chunk_offsets=kwargs["chunk_offsets"])
        if context["direct_state"]:
            return candidate(inp, config={"G": groups}, state_cache=cache,
                state_indices=indices, has_initial_state=kwargs["has_initial_state"], out=kwargs["out"])
        initial = gather_initial_states(cache, indices, kwargs["has_initial_state"])
        out, state = candidate(inp, initial_state=initial, config={"G": groups})
        # First isolate only the chunk kernel replacement. Direct paged-state
        # reads/writes and direct output placement are a separate experiment.
        destination = kwargs["out"]
        destination.copy_(out)
        cache[indices.long()] = state
        return destination, None

    def reset():
        for layer in layers[:3]:
            for state in layer.self_attn.kv_cache:
                state.zero_()

    def execute():
        h = hidden
        with set_forward_context(metadata, worker.vllm_config, num_tokens=length):
            # Normalization and residuals identical for both variants; omit
            # AttnRes mixing to isolate the four complete attention modules.
            for layer in layers:
                h = h + layer._run_self_attn(positions, layer.input_layernorm(h))
        return h

    def snapshot():
        return [[x.clone() for x in layer.self_attn.kv_cache] for layer in layers[:3]]

    points = {}
    try:
        reference = reference_states = copied_output = None
        variants = ("current", "gluon", "gluon_paged") if direct_state else ("current", "gluon")
        for variant in variants:
            kda_module.chunk_kda_prefill = original_kda if variant == "current" else patched
            context["direct_state"] = variant == "gluon_paged"
            reset()
            execute()  # Exact-shape compilation is outside timing.
            torch.cuda.synchronize()
            group.barrier()
            reset()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            out = execute()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            states = snapshot()
            point = dict(seconds=seconds)
            if reference is None:
                reference, reference_states = out, states
            else:
                point["output"] = check(reference, out, tolerance=0.02)
                point["state_checks"] = [dict(layer=i, conv=check(a[0], b[0], tolerance=0.02),
                    recurrent=check(a[1], b[1], tolerance=0.02))
                    for i, (a, b) in enumerate(zip(reference_states, states, strict=True))]
                if variant == "gluon":
                    copied_output = out
                elif variant == "gluon_paged":
                    point["matches_gluon_copies_bitwise"] = torch.equal(copied_output, out)
                    assert point["matches_gluon_copies_bitwise"], "direct I/O changed fixture output"
            points[variant] = point
            print("KDA_DENSE_FIXTURE " + json.dumps(dict(rank=rank, variant=variant, **point)), flush=True)
        return dict(rank=rank, layers=["KDA", "KDA", "KDA", "dense MLA"], points=points)
    finally:
        kda_module.chunk_kda_prefill = original_kda
        layers[3]._run_self_attn = original_attention


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--length", type=int, default=16384)
    p.add_argument("--groups", type=int, default=1)
    p.add_argument("--direct-state", action="store_true", help="Separately test direct paged state/output I/O")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    from benchmarks._vllm import llm_kwargs, close_llm, write_json
    from vllm import LLM
    kwargs = llm_kwargs(checkpoint="tests/fixtures/kimi-k3-mixed4", mode="full",
        max_model_len=args.length + 1, batch_size=1, tensor_parallel_size=8,
        decode_context_parallel_size=1, gpu_memory_utilization=0.1,
        full_attention_backend="ROCM_AITER_UNIFIED_ATTN")
    kwargs.update(load_format="dummy", skip_tokenizer_init=True, enforce_eager=True,
                  kv_cache_memory_bytes=1 << 29, max_num_batched_tokens=args.length)
    llm = LLM(**kwargs)
    try:
        workers = llm.collective_rpc(run_fixture, args=(args.length, args.groups, args.direct_state))
        assert {x["rank"] for x in workers} == set(range(8))
        points = {name: max(x["points"][name]["seconds"] for x in workers)
                  for name in workers[0]["points"]}
        result = dict(scope=__doc__, length=args.length, batch=1, tp=8,
            groups=args.groups, seconds=points, speedup=points["current"] / points["gluon"],
            workers=workers, random_weights=True, quality_evidence=False,
            timing="one warmed synchronized pass, max over TP ranks; no profiler in interval",
            changes=("only KDA chunk prefill; dense MLA, convolution, projections, gates and TP unchanged; "
                "gluon retains native state/output copies, gluon_paged removes them"
                if args.direct_state else "only KDA chunk prefill; dense MLA, convolution, projections, gates, state gather/scatter, output copy and TP unchanged"))
        write_json(args.output, result)
        print("KDA_DENSE_RESULT " + json.dumps(points), flush=True)
    finally:
        close_llm(llm)


if __name__ == "__main__":
    main()
