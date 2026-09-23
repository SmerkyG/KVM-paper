from __future__ import annotations

import math

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_inverse_coherence_mass_agrees_in_prefill_and_decode() -> None:
    from lod_attention.kernels.aiter_prefill_attention import (
        _prepare_aiter_state_kernel,
    )
    from lod_attention.kernels.paged_decode_kernels import (
        materialize_page1_coarse_means,
    )

    device = torch.device("cuda")
    state_k = torch.zeros(1, 1, 8, 256, dtype=torch.bfloat16, device=device)
    state_k[0, 0, 0, :2] = 1  # Sum of two orthogonal unit keys.
    state_k[0, 0, 1, 0] = 1
    state_v = state_k.clone()
    counts = torch.zeros(1, 1, 8, 1, dtype=torch.float32, device=device)
    counts[0, 0, 0, 0] = 2
    counts[0, 0, 1, 0] = 1
    key_norm_sums = counts / math.sqrt(256)

    mean_k = torch.empty(1, 1, 2, 256, dtype=state_k.dtype, device=device)
    mean_v = torch.empty_like(mean_k)
    masses = torch.empty(1, 1, 2, 1, dtype=torch.float32, device=device)
    prefill_bias = torch.empty(1, 2, 1, 2, dtype=torch.float16, device=device)
    _prepare_aiter_state_kernel[(2,)](
        state_k,
        state_v,
        counts,
        key_norm_sums,
        mean_k,
        mean_v,
        masses,
        prefill_bias,
        2,
        2,
        STATE_CAPACITY=8,
        KV_HEADS=1,
        KV_GROUP_SIZE=2,
        BLOCK_G=2,
        HEAD_DIM=256,
        BLOCK_D=256,
        HAS_KEY_NORM_SUMS=True,
    )

    coarse_k = torch.empty_like(state_k)
    coarse_v = torch.empty_like(state_v)
    decode_bias = torch.empty(1, 1, 8, dtype=torch.float16, device=device)
    materialize_page1_coarse_means(
        state_k,
        state_v,
        counts,
        coarse_k,
        coarse_v,
        decode_bias,
        active_state_len=2,
        key_norm_sums=key_norm_sums,
    )

    expected_mass = 2 * math.sqrt(2)
    assert masses[0, 0, 0, 0].item() == pytest.approx(expected_mass, rel=1e-5)
    assert masses[0, 0, 1, 0].item() == pytest.approx(1.0, rel=1e-5)
    expected_bias = math.log(expected_mass)
    assert prefill_bias[0, 0, 0, 0].item() == pytest.approx(expected_bias, abs=1e-3)
    assert decode_bias[0, 0, 0].item() == pytest.approx(expected_bias, abs=1e-3)
    assert decode_bias[0, 0, 1].item() == pytest.approx(0.0, abs=1e-3)
    torch.testing.assert_close(coarse_k[0, 0, :2], mean_k[0, 0])
