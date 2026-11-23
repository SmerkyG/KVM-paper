"""Per-head threshold selection, cache-slot reassignment and latent output checks."""

import os
from types import SimpleNamespace
import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("LOD_RUN_GPU_TESTS") != "1" or not torch.cuda.is_available(),
    reason="requires the ROCm GPU tests",
)


@pytest.mark.parametrize("c", [1.0e-6, 16.0, 1.0e6])
@torch.inference_mode()
@pytest.mark.parametrize("schedule", [(False, False, False), (False, True, False), (True, True, True), "ports-b1", "ports-b8"])
def test_glm_threshold_decode(c, schedule, monkeypatch):
    from vllm_lod_plugin.config import VLLMLODSettings
    from vllm_lod_plugin.pool import VLLMLayerLODPool

    monkeypatch.setenv("LOD_GLM_DECODE_THRESHOLD", str(c))
    torch.manual_seed(749)
    heads = 32
    if schedule in ("ports-b1", "ports-b8"):
        monkeypatch.setenv("LOD_GLM_THRESHOLD_GQA_PORTS", "1")
        monkeypatch.setenv("LOD_GLM_THRESHOLD_AUDIT_CAPTURE", "1")
        batch = 1 if schedule == "ports-b1" else 8
    else:
        batch = 2
    capacity = max(3, batch + 1)
    first = [(2 + i) % capacity for i in range(batch)]
    second = [(1 + i) % capacity for i in range(batch)]
    mapping = torch.tensor(first, dtype=torch.int32, device="cuda")
    layer = SimpleNamespace(
        num_heads=heads,
        num_kv_heads=1,
        head_size=512,
        head_size_v=512,
        kv_lora_rank=512,
        scale=0.0625,
        _vllm_lod_absorbed_mla=True,
        _vllm_lod_glm53=True,
    )
    pool = VLLMLayerLODPool(
        layer,
        settings=VLLMLODSettings.production(mode="two-tier", pool_size=capacity),
        max_requests=capacity,
        request_capacity=8192,
        active_indices=mapping,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    if isinstance(schedule, tuple):
        pool._glm_threshold_execution = schedule
    pool.reserve_decode_buffers(batch)
    # Include an empty remote field and a centroid above the per-region leaf cap.
    latent = torch.randn(4096, 512, device="cuda").bfloat16()
    latent[50:1800] = latent[0]
    for slot in range(capacity):
        n = (32, 4096, 3072)[slot % 3]
        keys = latent[:n][None, None]
        pool.install(slot, pool.engine.build_cache_from_bf16(keys, keys))
    pool._refresh_unified_page1_coarse(tuple(range(capacity)))
    pool.ensure_unified_page1_fixed(tuple(range(capacity)))
    page = pool.state["page_cache"]
    assert int(page["slot_lengths"].max()) > 1024
    lengths = pool.local_lens.clone()
    query = torch.randn(heads, batch, 512, device="cuda").bfloat16().transpose(0, 1)
    current = torch.randn(batch, 1, 512, device="cuda").bfloat16()
    out = torch.empty_like(query)

    def run():
        return pool.decode_dcp(query, current, current, out)

    run()
    pool.local_lens.copy_(lengths)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, lse = run()
    if schedule in ("ports-b1", "ports-b8"):
        from lod_attention.kernels.glm_threshold_decode import AMD_GQA_TUNING
        assert pool.glm_threshold_capture_audit == {batch: AMD_GQA_TUNING[batch]}
    for slots in (first, second):
        mapping.copy_(torch.tensor(slots, device="cuda", dtype=torch.int32))
        pool.local_lens.copy_(lengths)
        query.normal_()
        current.normal_()
        graph.replay()
        expected_lengths = lengths.clone()
        expected_lengths[mapping.long()] += 1
        torch.testing.assert_close(pool.local_lens, expected_lengths, rtol=0, atol=0)
        for b, slot in enumerate(slots):
            count = pool.state["counts"][slot, 0, :, 0].float()
            size = page["slot_lengths"][slot, 0]
            d = pool.state_capacity
            arena = page["unified_page1_k"]
            coarse = page["unified_page1_coarse_offset"] + slot * d
            mean = arena[coarse : coarse + d].float()
            n = int(lengths[slot])
            local = pool.state["recent_k"][slot, 0, : min(n, pool.decode_local_limit)].float()
            assert torch.equal(pool.state["recent_k"][slot, 0, n], current[b, 0])
            sink = pool.state["sink_k"][slot, 0].float()
            exact_local = torch.cat((local, current[b].float(), sink))
            start = pool.decode_local_limit + sink.size(0) + d
            owners = torch.arange(d, device="cuda").repeat_interleave(size.long())
            ids = page["unified_page1_fixed_indices"][
                slot, 0, start : start + owners.numel()
            ].long()
            for h in (0, 15, 16, 31):
                q = query[b, h].float()
                scores = (mean @ q * 0.0625 + count.log()).masked_fill(count <= 0, -torch.inf)
                # Empty remote fields leave only the protected and local branches.
                accept = (
                    (scores.softmax(0) > c / (count > 0).sum().clamp_min(1))
                    & (size <= 1024)
                    & (count > 0)
                )
                exact = torch.cat((arena[ids[accept[owners]]].float(), exact_local))
                closed = (count > 0) & ~accept
                logits = torch.cat((exact @ q * 0.0625, scores[closed]))
                ref = logits.softmax(0) @ torch.cat((exact, mean[closed]))
                torch.testing.assert_close(actual[b, h].float(), ref, rtol=0.025, atol=0.025)
                torch.testing.assert_close(lse[b, h], logits.logsumexp(0), rtol=0.0002, atol=0.0005)
