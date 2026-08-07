#!/usr/bin/env python3
"""Focused correctness and autograd checks for two-level KVM attention."""

from __future__ import annotations

import argparse
import math

import torch

from model.kvm_two_level_mixer import (
    _PackedAttentionWithLSE,
    _merge_lse_branches,
)


def _max_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def verify_identical_cluster_equivalence() -> dict[str, float]:
    """A count-corrected centroid is exact when a cluster's keys coincide."""
    torch.manual_seed(7)
    batch, heads, queries, clusters, dim = 1, 2, 5, 4, 8
    owner_row = torch.tensor([0, 0, 1, 1, 1, 2, 3, 3, 3, 3])
    owners = owner_row.view(1, 1, -1).expand(batch, heads, -1)
    counts = torch.tensor([2, 3, 1, 4], dtype=torch.float32)
    state_counts = counts.view(1, 1, clusters, 1).expand(batch, heads, -1, -1)

    q = torch.randn(batch, heads, queries, dim, requires_grad=True)
    cluster_k = torch.randn(batch, heads, clusters, dim, requires_grad=True)
    leaf_v = torch.randn(
        batch, heads, int(owner_row.numel()), dim, requires_grad=True
    )
    scale = 1.0 / math.sqrt(dim)

    with torch.no_grad():
        route_scores = torch.matmul(q, cluster_k.transpose(-1, -2)) * scale
        route_scores = route_scores + counts.log().view(1, 1, 1, clusters)
        top_slots = route_scores.topk(2, dim=-1).indices
    leaf_k_archive = cluster_k.gather(
        2, owners.unsqueeze(-1).expand(-1, -1, -1, dim)
    )
    selected_leaf = (
        owners.unsqueeze(2).unsqueeze(-1) == top_slots.unsqueeze(-2)
    ).any(dim=-1)
    exact_scores = torch.matmul(q, leaf_k_archive.transpose(-1, -2)) * scale
    exact_scores = exact_scores.masked_fill(~selected_leaf, float("-inf"))
    exact_lse = torch.logsumexp(exact_scores, dim=-1)
    exact_out = torch.matmul(torch.softmax(exact_scores, dim=-1), leaf_v)

    state_v_sum = torch.zeros(batch, heads, clusters, dim).scatter_add(
        2, owners.unsqueeze(-1).expand(-1, -1, -1, dim), leaf_v
    )
    state_v_mean = state_v_sum / state_counts
    coarse_scores = torch.matmul(q, cluster_k.transpose(-1, -2)) * scale
    coarse_scores = coarse_scores + counts.log().view(1, 1, 1, clusters)
    coarse_scores = coarse_scores.scatter(
        -1,
        top_slots,
        torch.full_like(top_slots, float("-inf"), dtype=coarse_scores.dtype),
    )
    coarse_lse = torch.logsumexp(coarse_scores, dim=-1)
    coarse_probability = torch.softmax(coarse_scores, dim=-1)
    coarse_out = torch.matmul(coarse_probability, state_v_mean)
    two_level = _merge_lse_branches(
        coarse_out, coarse_lse, exact_out, exact_lse
    )

    dense_scores = torch.matmul(q, leaf_k_archive.transpose(-1, -2)) * scale
    dense = torch.matmul(torch.softmax(dense_scores, dim=-1), leaf_v)
    output_error = _max_error(two_level, dense)

    two_level_grad = torch.autograd.grad(
        two_level.square().sum(), (q, cluster_k, leaf_v), retain_graph=True
    )
    dense_grad = torch.autograd.grad(
        dense.square().sum(), (q, cluster_k, leaf_v)
    )
    gradient_error = max(
        _max_error(actual, expected)
        for actual, expected in zip(two_level_grad, dense_grad, strict=True)
    )
    if output_error > 2.0e-6 or gradient_error > 2.0e-5:
        raise AssertionError(
            f"two-level equivalence failed: output={output_error}, "
            f"gradient={gradient_error}"
        )
    return {"output_max_abs": output_error, "gradient_max_abs": gradient_error}


def verify_gpu_mixer_smoke() -> dict[str, float | tuple[int, ...]]:
    if not torch.cuda.is_available():
        raise RuntimeError("--gpu requires a CUDA/ROCm device")
    from model.kvm_two_level_mixer import SequenceMixer
    from model.rwkv7_backbone import MixerConfigDataclass

    config = MixerConfigDataclass(
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        hidden_size=32,
        d_qk_head=16,
        d_v_head=16,
        ffn_expansion=2,
        chunk_len=4,
        n_bswa_chunks=2,
        n_max_d_chunks=100,
        state_budget_mode="power_law",
        state_growth_factor=2,
        state_growth_exponent=0.5,
        state_round_down=1,
        state_min_len=4,
        state_saturation_n=None,
        sink_len=1,
        rope_partial_dim=8,
        kvm_use_merge_gate_keys=0,
        kvm_use_merge_gate_values=0,
        kvm_use_vlens=0,
        use_value_residual=0,
        use_tokenshift_att=0,
        param_dtype="bfloat16",
    )
    mixer = SequenceMixer(config, 0).cuda().bfloat16().train()
    mixer.c_proj.weight.data.normal_(std=0.02)
    q = torch.randn(
        1, 2, 12, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    k = torch.randn(
        1, 1, 12, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    v = torch.randn(
        1, 1, 12, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    gate = torch.ones(1, 1, 12, 1, device="cuda")
    output = mixer.forward_prefill(q, k, v, gate, None, None, None)
    loss = output.float().square().mean()
    loss.backward()
    gradients = (q.grad, k.grad, v.grad, mixer.state_head_temp.grad)
    if any(item is None or not torch.isfinite(item).all() for item in gradients):
        raise AssertionError("two-level GPU smoke produced missing/nonfinite gradients")
    from model.statesdictcache import StatesDictCache

    mixer.eval()
    cache = StatesDictCache()
    with torch.no_grad():
        prefill = mixer.forward_prefill(q.detach(), k.detach(), v.detach(), gate, None, None, None, cache)
        decode = mixer.forward_single(
            q[:, :, :1].detach(),
            k[:, :, :1].detach(),
            v[:, :, :1].detach(),
            gate[:, :, :1],
            None,
            None,
            None,
            cache,
        )
    if tuple(prefill.shape) != (1, 12, 32) or tuple(decode.shape) != (1, 1, 32):
        raise AssertionError("two-level cached inference returned a wrong shape")
    if not torch.isfinite(decode).all():
        raise AssertionError("two-level cached inference returned nonfinite values")
    return {
        "output_shape": tuple(output.shape),
        "decode_shape": tuple(decode.shape),
        "loss": float(loss.detach().item()),
        "q_grad_norm": float(q.grad.float().norm().item()),
    }


def verify_packed_lse_gradient() -> dict[str, float]:
    """Compare the custom packed LSE VJP with explicit dense attention."""
    torch.manual_seed(11)
    q_lengths = torch.tensor([2, 1], device="cuda", dtype=torch.long)
    k_lengths = torch.tensor([4, 3], device="cuda", dtype=torch.long)
    cu_q = torch.nn.functional.pad(q_lengths.cumsum(0), (1, 0)).to(torch.int32)
    cu_k = torch.nn.functional.pad(k_lengths.cumsum(0), (1, 0)).to(torch.int32)
    scale = 1.0 / math.sqrt(16)
    packed_inputs = [
        torch.randn(
            length,
            1,
            16,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        for length in (3, 7, 7)
    ]
    q, k, v = packed_inputs
    out, lse = _PackedAttentionWithLSE.apply(
        q, k, v, cu_q, cu_k, q_lengths, k_lengths, 2, 4, scale
    )
    loss = out.float().square().sum() + 0.17 * lse.float().square().sum()
    packed_grad = torch.autograd.grad(loss, packed_inputs)

    reference_inputs = [
        item.detach().float().requires_grad_(True) for item in packed_inputs
    ]
    q_ref, k_ref, v_ref = reference_inputs
    reference_out = []
    reference_lse = []
    for expert in range(2):
        q_begin, q_end = map(int, cu_q[expert : expert + 2].tolist())
        k_begin, k_end = map(int, cu_k[expert : expert + 2].tolist())
        score = torch.matmul(
            q_ref[q_begin:q_end, 0], k_ref[k_begin:k_end, 0].T
        ) * scale
        probability = torch.softmax(score, dim=-1)
        reference_out.append(probability @ v_ref[k_begin:k_end, 0])
        reference_lse.append(torch.logsumexp(score, dim=-1))
    dense_out = torch.cat(reference_out).unsqueeze(1)
    dense_lse = torch.cat(reference_lse)
    dense_loss = dense_out.square().sum() + 0.17 * dense_lse.square().sum()
    dense_grad = torch.autograd.grad(dense_loss, reference_inputs)
    output_error = _max_error(out, dense_out)
    lse_error = _max_error(lse, dense_lse)
    gradient_error = max(
        _max_error(actual, expected)
        for actual, expected in zip(packed_grad, dense_grad, strict=True)
    )
    if output_error > 0.02 or lse_error > 0.02 or gradient_error > 0.08:
        raise AssertionError(
            "packed LSE parity failed: "
            f"out={output_error}, lse={lse_error}, grad={gradient_error}"
        )
    return {
        "output_max_abs": output_error,
        "lse_max_abs": lse_error,
        "gradient_max_abs": gradient_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    print({"equivalence": verify_identical_cluster_equivalence()})
    if args.gpu:
        print({"packed_lse": verify_packed_lse_gradient()})
        print({"gpu_smoke": verify_gpu_mixer_smoke()})


if __name__ == "__main__":
    main()
