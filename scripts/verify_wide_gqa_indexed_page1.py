#!/usr/bin/env python3
"""Check the D=512 split-QK/PV indexed decode specialization on ROCm."""

from __future__ import annotations

import torch

from model.kernels.paged_leaf_attention import wide_gqa_indexed_page1_attention


def reference(
    q: torch.Tensor,
    table: torch.Tensor,
    cache_indices: torch.Tensor,
    lengths: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    scale: float,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_count, heads, dimension = q.shape
    out = torch.empty_like(q)
    lse = torch.empty(sequence_count, heads, dtype=torch.float32, device=q.device)
    for sequence in range(sequence_count):
        physical_sequence = int(cache_indices[sequence].item())
        length = int(lengths[sequence].item())
        indices = table[physical_sequence, :length].long()
        visible = torch.ones(length, dtype=torch.bool, device=q.device)
        if mask is not None:
            visible &= mask[sequence, :length].bool()
        indices = indices[visible]
        logits = (
            q[sequence].float() @ key[indices].float().T * scale
            + bias[indices].float()[None, :]
        )
        probability = torch.softmax(logits, dim=-1)
        out[sequence] = (probability @ value[indices].float()).to(q.dtype)
        lse[sequence] = torch.logsumexp(logits, dim=-1)
    return out, lse


def run(masked: bool) -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    sequence_count, heads, dimension, capacity, arena_size = 2, 8, 512, 320, 1024
    segments = 8
    scale = dimension**-0.5
    q = torch.randn(
        sequence_count, heads, dimension, dtype=torch.bfloat16, device=device
    )
    key = torch.randn(arena_size, dimension, dtype=torch.bfloat16, device=device)
    value = torch.randn_like(key)
    bias = torch.randn(arena_size, dtype=torch.float16, device=device) * 0.2
    table = torch.stack(
        [torch.randperm(arena_size, device=device)[:capacity] for _ in range(2)]
    ).to(torch.int32)
    cache_indices = torch.tensor([1, 0], dtype=torch.int64, device=device)
    lengths = torch.tensor([257, 193], dtype=torch.int32, device=device)
    active_mask = None
    active_blocks = None
    if masked:
        active_mask = torch.zeros(
            sequence_count, capacity, dtype=torch.uint8, device=device
        )
        active_mask[:, :96] = 1
        active_mask[0, 192:257] = 1
        active_mask[1, 128:160] = 1
        active_blocks = active_mask.view(sequence_count, -1, 64).any(-1).to(torch.uint8)

    scores = torch.empty(
        sequence_count, heads, capacity, dtype=torch.float16, device=device
    )
    segment_out = torch.empty(
        sequence_count,
        heads,
        segments,
        dimension,
        dtype=torch.float32,
        device=device,
    )
    segment_max = torch.empty(
        sequence_count, heads, segments, dtype=torch.float32, device=device
    )
    segment_sum = torch.empty_like(segment_max)
    out = torch.empty_like(q)
    lse = torch.empty(sequence_count, heads, dtype=torch.float32, device=device)
    wide_gqa_indexed_page1_attention(
        q,
        cache_indices=cache_indices,
        sequence_lengths=lengths,
        block_table=table,
        key_cache=key,
        value_cache=value,
        key_bias=bias,
        scores=scores,
        segment_out=segment_out,
        segment_max=segment_max,
        segment_exp_sum=segment_sum,
        output=out,
        output_lse=lse,
        kv_heads=1,
        scale=scale,
        active_mask=active_mask,
        active_blocks=active_blocks,
    )
    expected, expected_lse = reference(
        q, table, cache_indices, lengths, key, value, bias, scale, active_mask
    )
    torch.cuda.synchronize()
    first_length = int(lengths[0].item())
    first_indices = table[int(cache_indices[0].item()), :first_length].long()
    first_visible = torch.ones(first_length, dtype=torch.bool, device=device)
    if active_mask is not None:
        first_visible &= active_mask[0, :first_length].bool()
    first_indices = first_indices[first_visible]
    score_reference = (
        q[0].float() @ key[first_indices].float().T * scale
        + bias[first_indices].float()[None, :]
    )
    stored_scores = scores[0, :, :first_length][:, first_visible].float()
    stored_scores_natural = stored_scores * 0.6931471805599453
    score_error = float(
        (stored_scores_natural - score_reference).abs().max().item()
    )
    score_lse_error = float(
        (
            torch.logsumexp(stored_scores_natural, dim=-1)
            - torch.logsumexp(score_reference, dim=-1)
        )
        .abs()
        .max()
        .item()
    )
    stored_output = (
        torch.softmax(stored_scores_natural, dim=-1)
        @ value[first_indices].float()
    ).to(q.dtype)
    stored_output_error = float(
        (stored_output.float() - expected[0].float()).abs().max().item()
    )
    output_error = float((out.float() - expected.float()).abs().max().item())
    lse_error = float((lse - expected_lse).abs().max().item())
    if output_error > 0.02 or lse_error > 0.02:
        raise AssertionError(
            f"masked={masked}: output error={output_error}, LSE error={lse_error}, "
            f"score error={score_error}, score LSE error={score_lse_error}, "
            f"stored output error={stored_output_error}"
        )
    print(
        f"masked={masked}: output_max_abs={output_error:.6f}, "
        f"lse_max_abs={lse_error:.6f}"
    )


if __name__ == "__main__":
    run(masked=False)
    run(masked=True)
