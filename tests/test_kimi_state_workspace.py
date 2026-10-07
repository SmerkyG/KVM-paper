"""MLA catch-up score storage follows overflow, not prefill vector capacity."""

import pytest
import torch

from lod_attention.kernels.lod_kernels import _materialized_score_output


@pytest.mark.parametrize("tokens", [32, 256, 257, 1024, 16384])
def test_score_workspace_reserves_only_actual_overflow_bucket(tokens):
    torch.manual_seed(89)
    left = torch.randn(2, 1, tokens, 3)
    right = torch.randn(2, 1, 3, 17)
    buffers = {}
    result = _materialized_score_output(buffers, "scores", left, right,
                                        token_capacity=16384, state_capacity=32)
    torch.testing.assert_close(result, left @ right, atol=0, rtol=0)
    assert buffers["scores"].numel() == 2 * ((tokens + 255) // 256) * 256 * 32
    assert result.is_contiguous()
    address = buffers["scores"].data_ptr()
    _materialized_score_output(buffers, "scores", left, right,
                               token_capacity=16384, state_capacity=32)
    assert buffers["scores"].data_ptr() == address


def test_score_workspace_grows_for_prefill_and_reuses_for_decode():
    buffers = {}
    right = torch.randn(1, 1, 3, 17)
    for tokens in (256, 1024, 256):
        left = torch.randn(1, 1, tokens, 3)
        result = _materialized_score_output(buffers, "scores", left, right,
                                            token_capacity=16384, state_capacity=32)
        torch.testing.assert_close(result, left @ right, atol=0, rtol=0)
        if tokens == 1024:
            address = buffers["scores"].data_ptr()
    assert buffers["scores"].numel() == 1024 * 32
    assert buffers["scores"].data_ptr() == address


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HIP MLA catch-up")
@torch.inference_mode()
def test_layer_batched_spherical_scan_keeps_scores_indices_and_bounded_workspace():
    from lod_attention.kernels.lod_kernels import new_state_maxsim_buffers, streaming_state_maxsim

    torch.manual_seed(89)
    leaves = torch.randn(16, 1, 256, 576, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(16, 1, 512, 576, device="cuda", dtype=torch.bfloat16)
    counts = torch.ones(16, 1, 512, 1, device="cuda", dtype=torch.float32)
    buffers = new_state_maxsim_buffers(leaves, 16384)
    scores, indices, select = streaming_state_maxsim(leaves, state, counts, buffers,
        state_len=512, sink_len=0, geometry="spherical",
        materialize_prepared_scores=True, mask_invalid_state=False)
    reference = leaves @ buffers["prepared_route_state"].transpose(-1, -2)
    expected_scores, expected_indices = reference.max(-1)
    torch.testing.assert_close(scores, expected_scores, atol=0, rtol=0)
    torch.testing.assert_close(indices, expected_indices, atol=0, rtol=0)
    torch.testing.assert_close(select, expected_scores, atol=0, rtol=0)
    assert buffers["materialized_route_scores"].numel() == 16 * 256 * 512
    address = buffers["materialized_route_scores"].data_ptr()
    streaming_state_maxsim(leaves, state, counts, buffers, state_len=512,
        sink_len=0, geometry="spherical", materialize_prepared_scores=True,
        mask_invalid_state=False)
    assert buffers["materialized_route_scores"].data_ptr() == address
