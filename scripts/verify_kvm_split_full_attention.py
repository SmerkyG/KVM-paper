#!/usr/bin/env python3
"""Check split full attention against an explicit causal dense reference."""

from __future__ import annotations

import math

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("verification requires a CUDA/ROCm GPU")
    torch.manual_seed(17)
    batch, heads, seq_len, dim = 1, 2, 13, 16
    chunk_len, bswa_len = 4, 8
    q, local_k, remote_k, v = (
        torch.randn(
            batch,
            heads,
            seq_len,
            dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        for _ in range(4)
    )
    scale = 1.0 / math.sqrt(dim)

    from model.kvm_split_full_attention_mixer import SequenceMixer
    from model.rwkv7_backbone import MixerConfigDataclass

    config = MixerConfigDataclass(
        num_hidden_layers=1,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        hidden_size=heads * dim,
        d_qk_head=dim,
        d_v_head=dim,
        ffn_expansion=2,
        chunk_len=chunk_len,
        n_bswa_chunks=bswa_len // chunk_len,
        rope_partial_dim=8,
        kvm_use_head_temps=1,
        use_value_residual=0,
        use_tokenshift_att=0,
        param_dtype="bfloat16",
    )
    mixer = SequenceMixer(config, 0).cuda().bfloat16()
    expected_bswa_begins = {8: 0, 9: 4, 12: 4, 13: 8, 16: 8, 17: 12}
    actual_bswa_begins = {
        length: mixer._bswa_begin_for_total_len(length)
        for length in expected_bswa_begins
    }
    if actual_bswa_begins != expected_bswa_begins:
        raise AssertionError(
            f"chunk-aligned decode boundaries differ: {actual_bswa_begins}"
        )
    local_block_mask, remote_block_mask = mixer._split_masks(seq_len, q.device)
    split = mixer._split_prefill_attention(
        q,
        local_k,
        remote_k,
        v,
        local_block_mask,
        remote_block_mask,
    )

    q_idx = torch.arange(seq_len, device="cuda").unsqueeze(1)
    kv_idx = torch.arange(seq_len, device="cuda").unsqueeze(0)
    chunk_end = (torch.div(q_idx, chunk_len, rounding_mode="floor") + 1) * chunk_len
    local_begin = torch.where(q_idx < bswa_len, 0, chunk_end - bswa_len)
    local_mask = (kv_idx >= local_begin) & (kv_idx <= q_idx)
    remote_mask = (q_idx >= bswa_len) & (kv_idx < (chunk_end - bswa_len))

    reference_inputs = tuple(
        item.detach().float().requires_grad_(True)
        for item in (q, local_k, remote_k, v)
    )
    q_ref, local_k_ref, remote_k_ref, v_ref = reference_inputs
    local_score = torch.matmul(
        q_ref, local_k_ref.transpose(-1, -2)
    ) * scale
    remote_score = torch.matmul(
        q_ref, remote_k_ref.transpose(-1, -2)
    ) * scale
    local_score = local_score.masked_fill(~local_mask, float("-inf"))
    remote_score = remote_score.masked_fill(~remote_mask, float("-inf"))
    combined_score = torch.where(local_mask, local_score, remote_score)
    dense = torch.softmax(combined_score, dim=-1) @ v_ref
    output_error = float((split - dense).abs().max().item())
    split_grad = torch.autograd.grad(
        split.square().sum(), (q, local_k, remote_k, v)
    )
    dense_grad = torch.autograd.grad(dense.square().sum(), reference_inputs)
    gradient_stats = []
    for actual, expected in zip(split_grad, dense_grad, strict=True):
        delta = actual.float() - expected.float()
        expected_float = expected.float()
        gradient_stats.append(
            {
                "max_abs": float(delta.abs().max().item()),
                "relative_l2": float(
                    delta.norm().div(expected_float.norm().clamp_min(1e-12)).item()
                ),
                "cosine": float(
                    torch.nn.functional.cosine_similarity(
                        actual.float().flatten(), expected_float.flatten(), dim=0
                    ).item()
                ),
            }
        )
    gradient_error = max(stat["max_abs"] for stat in gradient_stats)
    if output_error > 0.02 or gradient_error > 0.08:
        raise AssertionError(
            f"split equivalence failed: output={output_error}, grad={gradient_error}"
        )
    print(
        {
            "output_max_abs": output_error,
            "gradient_max_abs": gradient_error,
            "gradient_stats_q_local_k_remote_k_v": gradient_stats,
        }
    )


if __name__ == "__main__":
    main()
