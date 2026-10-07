"""Dense decode partitions must follow live rows, including graph replay."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gfx942 kernel check")
@pytest.mark.parametrize("page_size", [1, 768])
def test_live_dense_splits_ragged_graph_replay(page_size):
    from lod_attention.kernels.kimi_gluon_decode import absorbed_mla_decode_gfx942

    torch.manual_seed(41)
    heads, capacity = 17, 73728
    # The occupancy floor is 16 for this geometry. Cross larger boundaries
    # too, so replay changes the actual partition count (16 -> 32 -> 64 -> 128).
    lengths = [0, 127, 16383, 16384, 32767, 32768, 65535, 65536]
    batch = len(lengths)
    blocks = capacity // page_size
    kv = torch.randn(batch * blocks, page_size, 576, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(batch, heads, 576, device="cuda", dtype=torch.bfloat16)
    table = torch.randperm(batch * blocks, device="cuda").to(torch.int32).view(batch, blocks)
    seq_lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    out = torch.empty(batch, heads, 512, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(batch, heads, device="cuda")
    partial = torch.empty(batch, heads, 128, 512, device="cuda", dtype=q.dtype)
    partial_lse = torch.empty(batch, heads, 128, device="cuda")
    scale = 576**-0.5

    def run():
        absorbed_mla_decode_gfx942(q, kv, out, table, seq_lens, scale,
            num_splits=128, partial=partial, partial_lse=partial_lse,
            final_lse=lse, adaptive_splits=True)

    run()  # Compile outside capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for current in [lengths, list(reversed(lengths)), [16385] * batch]:
        seq_lens.copy_(torch.tensor(current, device="cuda", dtype=torch.int32))
        # Inactive slots retain arbitrary data; never consume them on replay.
        partial.fill_(float("nan"))
        partial_lse.fill_(float("nan"))
        graph.replay()
        for row, length in enumerate(current):
            if not length:
                assert torch.equal(out[row], torch.zeros_like(out[row]))
                assert torch.isneginf(lse[row]).all()
                continue
            keys = kv[table[row].long()].reshape(capacity, 576)[:length].float()
            scores = q[row].float() @ keys.T * scale
            reference = scores.softmax(-1) @ keys[:, :512]
            torch.testing.assert_close(out[row].float(), reference, atol=2e-3, rtol=1e-2)
            torch.testing.assert_close(lse[row], scores.logsumexp(-1), atol=3e-4, rtol=3e-4)


def test_dense_adapter_does_not_select_splits_from_capture_capacity():
    import inspect
    from vllm_lod_plugin.models.kimi_k3 import _install_dense_gluon_decode

    source = inspect.getsource(_install_dense_gluon_decode)
    assert "attn_metadata.max_seq_len" not in source
    assert "adaptive_splits=True" in source


def test_dense_kernel_defaults_to_live_splits():
    import inspect
    from lod_attention.kernels.kimi_gluon_decode import absorbed_mla_decode_gfx942

    assert inspect.signature(absorbed_mla_decode_gfx942).parameters["adaptive_splits"].default is True
