"""The tiled reducer must ignore unwritten splits and advance each row once."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU split reducer")


@pytest.mark.parametrize("batch,splits", [(1, 1), (1, 32), (1, 33), (8, 16), (8, 64), (8, 128)])
@pytest.mark.parametrize("has_lse", [False, True])
def test_split_reduce_masks_empty_and_all_masked_splits(batch, splits, has_lse):
    from benchmarks.kimi_k3_decode_reduce_tune import inputs, launch_parallel, reference
    p, lse, lengths, live = inputs(batch, splits, poison_masked=True)
    if splits in (16, 32):
        # Production retains 64-split scratch capacity for graph-safe batch
        # changes; live B1/B8 consume strided 32/16-split views.
        backing = torch.full((batch, 96, 64, 512), float("nan"),
                             dtype=p.dtype, device=p.device)
        backing_lse = torch.full((batch, 96, 64), float("nan"), device=p.device)
        backing[:, :, :splits].copy_(p)
        backing_lse[:, :, :splits].copy_(lse)
        p, lse = backing[:, :, :splits], backing_lse[:, :, :splits]
    out = torch.empty(batch, 96, 512, dtype=torch.bfloat16, device="cuda")
    final = torch.full((batch, 96), 123., device="cuda")
    launch_parallel(p, lse, lengths, out, final, has_lse=has_lse)
    expected, expected_lse = reference(p, lse, live)
    torch.testing.assert_close(out, expected, rtol=0.008, atol=0.002)
    assert bool(out.isfinite().all())
    if has_lse:
        torch.testing.assert_close(final, expected_lse, rtol=1e-6, atol=2e-5)
    else:
        assert bool(final.eq(123.).all())


@pytest.mark.parametrize("rank,interleave", [(0, 1), (7, 1), (3, 64)])
def test_captured_split_reduce_updates_indirected_dcp_rows_once(rank, interleave):
    from benchmarks.kimi_k3_decode_reduce_tune import inputs, launch_parallel, reference
    p, lse, lengths, live = inputs(8, 64, poison_masked=True)
    out = torch.empty(8, 96, 512, dtype=torch.bfloat16, device="cuda")
    final = torch.empty(8, 96, device="cuda")
    indices = torch.tensor([9, 7, 5, 3, 8, 6, 4, 2], dtype=torch.int32, device="cuda")
    local = torch.arange(12, dtype=torch.int64, device="cuda") * 5
    global_lens = torch.arange(12, dtype=torch.int64, device="cuda") * interleave
    before_local, before_global = local.clone(), global_lens.clone()
    def run():
        launch_parallel(p, lse, lengths, out, final, indices=indices, local=local,
                        global_lens=global_lens, rank=rank, interleave=interleave, advance=True)
    run()  # Compile before graph capture.
    torch.cuda.synchronize()
    local.copy_(before_local)
    global_lens.copy_(before_global)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    local.copy_(before_local)
    global_lens.copy_(before_global)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    expected_local, expected_global = before_local.clone(), before_global.clone()
    ids = indices.long()
    for _ in range(3):
        owned = ((expected_global[ids] // interleave) % 8) == rank
        expected_local[ids] += owned.long()
        expected_global[ids] += 1
    torch.testing.assert_close(local, expected_local)
    torch.testing.assert_close(global_lens, expected_global)
    expected, expected_lse = reference(p, lse, live)
    torch.testing.assert_close(out, expected, rtol=0.008, atol=0.002)
    torch.testing.assert_close(final, expected_lse, rtol=1e-6, atol=2e-5)


@pytest.mark.parametrize("uniform", [False, True])
def test_split_reduce_preserves_broad_attention_mass(uniform):
    from benchmarks.kimi_k3_decode_reduce_tune import launch_parallel, reference
    torch.manual_seed(23)
    p = torch.randn(8, 96, 64, 512, dtype=torch.bfloat16, device="cuda")
    lse = torch.randn(8, 96, 64, device="cuda")
    if uniform:
        lse.zero_()
    lengths = torch.full((8*6,), 4096, dtype=torch.int32, device="cuda")
    out = torch.empty(8, 96, 512, dtype=torch.bfloat16, device="cuda")
    final = torch.empty(8, 96, device="cuda")
    launch_parallel(p, lse, lengths, out, final)
    expected, expected_lse = reference(p, lse, torch.ones_like(lse, dtype=torch.bool))
    torch.testing.assert_close(out, expected, rtol=0.008, atol=0.002)
    torch.testing.assert_close(final, expected_lse, rtol=1e-6, atol=2e-5)
