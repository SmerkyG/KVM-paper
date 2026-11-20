"""Check the summary ablation on identical BF16 query and cache states."""

import argparse
from pathlib import Path

from benchmarks.amd_threshold_prefill_quality_panel import write


def prefill_check():
    import torch

    from lod_attention.kernels.aiter_prefill_attention import (
        aiter_prefill_route_coarse_attention, merge_aiter_prefill_refinement,
        record_prefill_exact_work,
    )

    torch.manual_seed(42)
    shape = (1, 6, 17, 256)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    mean_k = torch.randn((1, 1, 128, 256), device=q.device, dtype=q.dtype) / 4
    mean_v = torch.randn_like(mean_k) / 4
    counts = 2 ** (torch.arange(128, device=q.device) % 5 + 1)
    populations = counts.reshape(1, 1, 128, 1).float()
    state_k, state_v = (mean_k * populations).to(q.dtype), (mean_v * populations).to(q.dtype)
    local_out = torch.randn_like(q)
    local_lse = torch.rand(q.shape[:-1], device=q.device)
    route_shape = (*q.shape[:-1], 8)
    route_out = torch.randn((*route_shape, q.size(-1)), device=q.device, dtype=q.dtype)
    route_lse = torch.rand(route_shape, device=q.device)
    sink = mean_k[:, :, :0]
    scale = 256 ** -0.5
    expected_routes = None
    errors = {}
    for mode in ("weighted", "sparse", "unweighted"):
        slots, coarse, _, _ = aiter_prefill_route_coarse_attention(
            q, state_k, state_v, populations, state_len=128, kv_group_size=6,
            scale=scale, normalize_route_query=False, summary_mode=mode,
        )
        if expected_routes is None:
            expected_routes = slots
        torch.testing.assert_close(slots, expected_routes, rtol=0, atol=0)
        actual = merge_aiter_prefill_refinement(
            q, sink, sink, coarse, slots, route_out, route_lse, local_out, local_lse,
            kv_group_size=6, scale=scale,
        )
        summary_logits = torch.matmul(q.float(), mean_k.float().transpose(-1, -2)) * scale
        if mode == "weighted":
            summary_logits += counts.float().log()
        if mode == "sparse":
            summary_logits.fill_(float("-inf"))
        summary_logits.scatter_(-1, slots.long(), float("-inf"))
        weights = torch.cat((local_lse.unsqueeze(-1), route_lse, summary_logits), -1).softmax(-1)
        expected = (weights[..., :1] * local_out.float()
                    + (weights[..., 1:9, None] * route_out.float()).sum(-2)
                    + weights[..., 9:] @ mean_v.float())
        torch.testing.assert_close(actual.float(), expected, rtol=0.03, atol=0.015)
        work = torch.zeros(6, device=q.device, dtype=torch.int64)
        record_prefill_exact_work(
            work, slots, counts.to(torch.int32).reshape(1, 1, 128),
            kv_group_size=6, exact_prefix=3,
        )
        remote = int(counts[slots.long()].sum())
        expected_total = remote + 6 * sum(range(4, 21))
        assert work.tolist() == [remote, expected_total, 6 * 17, 0, 0, 0]
        errors[mode] = (actual.float() - expected).abs().max().item()
    return errors


def decode_check():
    import torch

    from lod_attention.kernels.paged_decode_buffers import _materialize_decode_gqa_union_page_descriptors_kernel

    def ints(values):
        return torch.tensor(values, device="cuda", dtype=torch.int32)

    lengths = ints([10] * 16)
    counts = lengths.to(torch.float32)
    offsets = ints(list(range(0, 161, 10)))
    selected = ints([2, 6])
    seen = ints([1 if i in (2, 6) else 0 for i in range(16)])
    bias = torch.full((24,), 99.0, device="cuda")
    k = torch.zeros((24, 256), device="cuda", dtype=torch.bfloat16)
    current = k[:1].unsqueeze(0)
    descriptor = ints([0] * 8)
    context = ints([0])
    padded = ints([0])
    work = torch.zeros(6, device="cuda", dtype=torch.int64)
    actual = {}
    for mode in ("weighted", "sparse", "unweighted"):
        _materialize_decode_gqa_union_page_descriptors_kernel[(1, 2)](
            ints([0]), ints([3]), ints([16]), lengths, ints([2]), padded,
            selected, offsets, counts, seen, ints([1]), current, current,
            k, k, bias, descriptor, context, work, 16, 16, 1,
            256, 256, 256, 256, 17, KV_HEADS=1, STATE_LEN=16,
            STATE_CAPACITY=16, INDEX_CAPACITY=8, UNION_CAPACITY=2, UNION_BLOCK=2,
            LOCAL_OFFSET=0, COARSE_OFFSET=8, LOCAL_CAPACITY=8, LOCAL_LIMIT=4,
            SINK_LEN=1, LEAF_BEGIN=21, HEAD_DIM=256, PAGE_SIZE=16, BLOCK_K=64,
            INCLUDE_NEW=True, USE_STATE_LENS=True, SUMMARY_MODE=mode,
            TRACK_EXACT_WORK=True, QUERY_HEADS_PER_KV=6, num_warps=1,
        )
        assert context.item() == 5 + 32 + (0 if mode == "sparse" else 16)
        assert padded.item() == 32
        assert descriptor.tolist()[:2] == [(10 << 24) | 41, (10 << 24) | 81]
        assert work.tolist() == [0, 0, 0, 120, 150, 6]
        if mode != "sparse":
            assert torch.isneginf(bias[8 + selected.long()]).all()
            expected_bias = counts[0].log() if mode == "weighted" else counts[0] * 0
            torch.testing.assert_close(bias[8], expected_bias)
        actual[mode] = context.item()
        work.zero_()
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write(args.output, {"stage": "complete", "prefill_max_abs_error": prefill_check(),
                        "decode_context_lengths": decode_check()})


if __name__ == "__main__":
    main()
