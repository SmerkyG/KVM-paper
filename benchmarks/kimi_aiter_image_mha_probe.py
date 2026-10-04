#!/usr/bin/env python3
"""Time K3 attention kernels shipped in the reference AITER image.

``mla-prefill`` folds Kimi's 96 query heads into six independent 16-head
pseudo-sequences.  That lets the gfx942 absorbed-MLA assembly kernel consume
the model's native 576-wide query/key and 512-wide value geometry without
changing its supported GQA=16 launch shape.
"""

from __future__ import annotations

import argparse
import json

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=256)
    parser.add_argument("--states", type=int, default=512)
    parser.add_argument("--heads", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--splits",
        type=int,
        default=128,
        help="split-K workgroups for the native gfx942 Gluon MLA probe",
    )
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--distinct-sequences",
        action="store_true",
        help="Give every decode row a disjoint KV range instead of sharing one.",
    )
    parser.add_argument(
        "--kernel",
        choices=(
            "mha",
            "mla-prefill",
            "mla-decode",
            "mla-triton",
            "vllm-triton",
            "mla-gluon-gfx942",
        ),
        default="mha",
    )
    args = parser.parse_args()

    torch.manual_seed(19)
    q = torch.randn(
        1, args.queries, args.heads, 576, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(1, args.states, 1, 576, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, args.states, 1, 512, device="cuda", dtype=torch.bfloat16)
    bias = (
        torch.randn(
            1, args.heads, 1, args.states, device="cuda", dtype=torch.bfloat16
        )
        if args.bias
        else None
    )
    out = torch.empty(
        1, args.queries, args.heads, 512, device="cuda", dtype=torch.bfloat16
    )

    if args.kernel == "mha":
        from aiter.ops.mha import _flash_attn_forward

        def run():
            return _flash_attn_forward(
                q,
                k,
                v,
                0.0,
                576**-0.5,
                False,
                -1,
                -1,
                0,
                bias,
                None,
                None,
                None,
                None,
                True,
                False,
                out=out,
            )

    elif args.kernel == "mla-prefill":
        if args.bias:
            raise ValueError("the absorbed-MLA assembly has no bias input")
        if args.heads % 16:
            raise ValueError("absorbed-MLA folding requires a multiple of 16 heads")
        from aiter.mla import mla_prefill_fwd

        groups = args.heads // 16
        folded_q = (
            q[0]
            .view(args.queries, groups, 16, 576)
            .permute(1, 0, 2, 3)
            .contiguous()
            .view(groups * args.queries, 16, 576)
        )
        # The absorbed kernel aliases K and V: the first 512 channels are V.
        kv = k.view(args.states, 1, 1, 576).contiguous()
        folded_out = out[0].view(args.queries, groups, 16, 512).permute(1, 0, 2, 3)
        folded_out = folded_out.contiguous().view(groups * args.queries, 16, 512)
        qo_indptr = torch.arange(
            groups + 1, device="cuda", dtype=torch.int32
        ) * args.queries
        kv_indptr = torch.arange(
            groups + 1, device="cuda", dtype=torch.int32
        ) * args.states
        kv_indices = torch.arange(
            args.states, device="cuda", dtype=torch.int32
        ).repeat(groups)
        kv_last_page_lens = torch.ones(groups, device="cuda", dtype=torch.int32)

        def run():
            return mla_prefill_fwd(
                folded_q,
                kv,
                folded_out,
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_lens,
                args.queries,
                576**-0.5,
            )

    elif args.kernel == "mla-decode":
        if args.bias:
            raise ValueError("the absorbed-MLA assembly has no bias input")
        from aiter.mla import mla_decode_fwd

        # K3 has 96 heads, hence 12 local heads at TP8.  The MI325X image
        # pads that shard to the assembly kernel's 16-head tile; padding all
        # the way to 128 measures a different and needlessly expensive shape.
        padded_heads = max(16, args.heads)
        if padded_heads % 16:
            padded_heads = ((padded_heads + 15) // 16) * 16
        decode_q = torch.zeros(
            args.queries,
            padded_heads,
            576,
            device="cuda",
            dtype=torch.bfloat16,
        )
        decode_q[:, : args.heads].copy_(q[0])
        kv = k.view(args.states, 1, 1, 576).contiguous()
        decode_out = torch.empty(
            args.queries,
            padded_heads,
            512,
            device="cuda",
            dtype=torch.bfloat16,
        )
        qo_indptr = torch.tensor(
            [0, args.queries], device="cuda", dtype=torch.int32
        )
        kv_indptr = torch.tensor(
            [0, args.states], device="cuda", dtype=torch.int32
        )
        kv_indices = torch.arange(args.states, device="cuda", dtype=torch.int32)
        kv_last_page_lens = torch.ones(1, device="cuda", dtype=torch.int32)

        def run():
            return mla_decode_fwd(
                decode_q,
                kv,
                decode_out,
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_lens,
                args.queries,
                page_size=1,
                nhead_kv=1,
                sm_scale=576**-0.5,
            )

    elif args.kernel == "mla-triton":
        if args.bias:
            raise ValueError("the public Triton MLA path has no bias input")
        from aiter.ops.triton.attention.mla import mla_decode_fwd

        decode_q = q[0].contiguous()
        kv = k.view(args.states, 1, 1, 576).contiguous()
        decode_out = torch.empty(
            args.queries,
            args.heads,
            512,
            device="cuda",
            dtype=torch.bfloat16,
        )
        cu_seqlens_q = torch.tensor(
            [0, args.queries], device="cuda", dtype=torch.int32
        )
        seqused_k = torch.tensor(
            [args.states], device="cuda", dtype=torch.int32
        )
        block_tables = torch.arange(
            args.states, device="cuda", dtype=torch.int32
        ).view(1, -1)

        def run():
            return mla_decode_fwd(
                decode_q,
                kv,
                decode_out,
                cu_seqlens_q,
                seqused_k,
                args.states,
                block_tables,
                576**-0.5,
                512,
                64,
                True,
                None,
                None,
            )

    elif args.kernel == "vllm-triton":
        if args.bias:
            raise ValueError("the vLLM Triton MLA path has no bias input")
        from vllm.v1.attention.ops.triton_decode_attention import (
            decode_attention_fwd,
        )

        decode_q = q[0].contiguous()
        kv = k.view(args.states, 1, 1, 576).contiguous()
        decode_out = torch.empty(
            args.queries,
            args.heads,
            512,
            device="cuda",
            dtype=torch.bfloat16,
        )
        decode_lse = torch.empty(
            args.queries, args.heads, device="cuda", dtype=torch.float32
        )
        block_tables = torch.arange(
            args.states, device="cuda", dtype=torch.int32
        ).view(1, -1).expand(args.queries, -1)
        seqused_k = torch.tensor(
            [args.states] * args.queries, device="cuda", dtype=torch.int32
        )
        num_splits = min(
            608,
            1 << max(0, (max(1, args.states // 512) - 1).bit_length()),
        )
        attn_logits = torch.empty(
            args.queries,
            args.heads,
            num_splits,
            513,
            device="cuda",
            dtype=torch.float32,
        )

        def run():
            return decode_attention_fwd(
                decode_q,
                kv,
                kv[..., :512],
                decode_out,
                decode_lse,
                block_tables,
                seqused_k,
                attn_logits,
                num_splits,
                576**-0.5,
                1,
                is_mla=True,
            )

    elif args.kernel == "mla-gluon-gfx942":
        if args.bias:
            raise ValueError("the gfx942 Gluon MLA path has no bias input")
        from kimi_gluon_mla_decode import absorbed_mla_decode_gfx942

        decode_q = q[0].contiguous()
        if args.distinct_sequences:
            kv = torch.randn(
                args.queries * args.states,
                576,
                device="cuda",
                dtype=torch.bfloat16,
            )
            block_tables = (
                torch.arange(
                    args.queries * args.states,
                    device="cuda",
                    dtype=torch.int32,
                )
                .view(args.queries, args.states)
                .contiguous()
            )
        else:
            kv = k.view(args.states, 576).contiguous()
            block_tables = torch.arange(
                args.states, device="cuda", dtype=torch.int32
            ).view(1, -1).expand(args.queries, -1)
        decode_out = torch.empty(
            args.queries,
            args.heads,
            512,
            device="cuda",
            dtype=torch.bfloat16,
        )
        seqused_k = torch.tensor(
            [args.states] * args.queries, device="cuda", dtype=torch.int32
        )

        def run():
            return absorbed_mla_decode_gfx942(
                decode_q,
                kv,
                decode_out,
                block_tables,
                seqused_k,
                576**-0.5,
                num_splits=args.splits,
            )

    run()
    torch.cuda.synchronize()
    if args.check and args.kernel != "mha":
        if args.kernel == "mla-prefill":
            result = folded_out.view(groups, args.queries, 16, 512)
            result = result.permute(1, 0, 2, 3).reshape(
                args.queries, args.heads, 512
            )
        else:
            result = decode_out[:, : args.heads]
        scores = torch.einsum(
            "qhd,sd->qhs", q[0].float(), k[0, :, 0].float()
        ) * (576**-0.5)
        reference = torch.einsum(
            "qhs,sd->qhd", scores.softmax(-1), k[0, :, 0, :512].float()
        )
        query_pos = torch.arange(args.queries, device="cuda")
        key_pos = torch.arange(args.states, device="cuda")
        causal_limit = args.states - args.queries + query_pos
        causal_scores = scores.masked_fill(
            key_pos[None, None, :] > causal_limit[:, None, None],
            -float("inf"),
        )
        causal_reference = torch.einsum(
            "qhs,sd->qhd",
            causal_scores.softmax(-1),
            k[0, :, 0, :512].float(),
        )
        print(json.dumps({
            "full_max_error": float((result.float() - reference).abs().max()),
            "causal_max_error": float(
                (result.float() - causal_reference).abs().max()
            ),
        }))
        torch.testing.assert_close(
            result.float(),
            causal_reference if args.kernel == "mla-prefill" else reference,
            atol=4e-2,
            rtol=4e-2,
        )
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(args.iterations):
        run()
    end.record()
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "queries": args.queries,
                "states": args.states,
                "heads": args.heads,
                "bias": args.bias,
                "kernel": args.kernel,
                "distinct_sequences": args.distinct_sequences,
                "milliseconds": begin.elapsed_time(end) / args.iterations,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
