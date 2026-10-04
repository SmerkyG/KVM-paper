from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("LOD_RUN_GPU_TESTS") != "1" or not torch.cuda.is_available(),
    reason="set LOD_RUN_GPU_TESTS=1 on a CUDA/ROCm worker",
)


@pytest.mark.parametrize("dimensions", [(64, 32), (576, 512)])
@pytest.mark.parametrize("indirect", [False, True])
def test_prefix_alias_merge_matches_scatter_and_replays(dimensions, indirect):
    from lod_attention.kernels.lod_kernels import merge_state_in_place, new_state_delta_buffers

    width, value_width = dimensions
    batch, heads, slots, tokens = 2, 2, 31, 519
    sources = tokens + 43 if indirect else tokens
    generator = torch.Generator(device="cuda").manual_seed(9)
    # Exact binary fractions make FP32 atomic order irrelevant for this test.
    keys = torch.randint(-4, 5, (batch, heads, slots, width), generator=generator,
                         device="cuda").to(torch.bfloat16) / 8
    source = torch.randint(-4, 5, (batch, heads, sources, width), generator=generator,
                           device="cuda").to(torch.bfloat16) / 8
    counts = torch.zeros(batch, heads, slots, 1, device="cuda")
    merge_counts = torch.ones(batch, heads, sources, 1, device="cuda")
    indices = (torch.arange(tokens, device="cuda") + (17 if indirect else 0))
    indices = indices[None, None].repeat(batch, heads, 1).contiguous()
    destination = (torch.arange(tokens, device="cuda") % 23)[None, None]
    destination = destination.repeat(batch, heads, 1).contiguous()
    destination[..., :400] = 1  # Heavy contention, plus untouched slots.
    owners = torch.full((batch, heads, sources), -1, dtype=torch.long, device="cuda")
    buffers = new_state_delta_buffers(keys, keys[..., :value_width], slots)
    initial = keys.clone()

    def restore():
        keys.copy_(initial)
        counts.zero_()
        owners.fill_(-1)

    def update(shared):
        merge_state_in_place(
            keys, keys[..., :value_width], counts, source,
            source[..., :value_width] if shared else source[..., :value_width].contiguous(),
            merge_counts, indices, destination, owners, buffers,
            active_slots=slots, indirect_source=indirect, shared_value_prefix=shared,
        )

    def check():
        selected = source.gather(2, indices[..., None].expand(-1, -1, -1, width))
        expected = initial.float().scatter_add(
            2, destination[..., None].expand_as(selected), selected.float()).bfloat16()
        expected_counts = torch.zeros_like(counts).scatter_add(
            2, destination[..., None], torch.ones_like(destination[..., None]).float())
        expected_owners = torch.full_like(owners, -1).scatter(2, indices, destination)
        torch.testing.assert_close(keys, expected, atol=0, rtol=0)
        torch.testing.assert_close(counts, expected_counts, atol=0, rtol=0)
        torch.testing.assert_close(owners, expected_owners, atol=0, rtol=0)
        assert not buffers["delta_k"].any()
        assert not buffers["delta_v"].any()
        assert not buffers["delta_counts"].any()
        assert not buffers["touched"].any()

    with torch.inference_mode():
        for shared in (False, True, True, False):
            restore()
            update(shared)
            check()
        restore()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            update(True)
        source.mul_(-0.5)
        initial.mul_(0.5)
        for _ in range(2):
            restore()
            graph.replay()
            check()


@pytest.mark.parametrize("separate", ["state", "source"])
def test_shared_merge_rejects_nonaliased_values(separate):
    from lod_attention.kernels.lod_kernels import merge_state_in_place, new_state_delta_buffers

    key = torch.zeros(1, 1, 8, 64, dtype=torch.bfloat16, device="cuda")
    source = torch.ones(1, 1, 3, 64, dtype=torch.bfloat16, device="cuda")
    value = key[..., :32].clone() if separate == "state" else key[..., :32]
    source_v = source[..., :32].clone() if separate == "source" else source[..., :32]
    index = torch.arange(3, device="cuda")[None, None]
    with pytest.raises(ValueError, match="actual K-prefix aliases"):
        merge_state_in_place(
            key, value, torch.zeros(1, 1, 8, 1, device="cuda"), source, source_v,
            None, index, index, torch.empty_like(index),
            new_state_delta_buffers(key, value, 8), shared_value_prefix=True,
        )
