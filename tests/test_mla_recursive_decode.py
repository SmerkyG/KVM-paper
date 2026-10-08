"""Small integration checks for graph-captured MLA top-two-page decode."""

import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("LOD_RUN_GPU_TESTS") != "1" or not torch.cuda.is_available(),
    reason="set LOD_RUN_GPU_TESTS=1 on a CUDA/ROCm worker",
)


@pytest.mark.parametrize("dim,heads,batch", [(512, 16, 2), (576, 96, 2), (512, 48, 8)])
@pytest.mark.parametrize("mode", ["three-tier-bf16", "three-tier-int4"])
@pytest.mark.parametrize("query_scale", [.3, 16.0])
@torch.inference_mode()
def test_recursive_mla_pool_baseline_lse_and_graph(dim, heads, batch, mode, query_scale):
    from lod_attention._engines import KernelLODCache
    from vllm_lod_plugin.config import VLLMLODSettings
    from vllm_lod_plugin.pool import VLLMLayerLODPool

    torch.manual_seed(143)
    layer = SimpleNamespace(num_heads=heads, num_kv_heads=1, head_size=dim,
                            head_size_v=512, kv_lora_rank=512, scale=.0625,
                            _vllm_lod_absorbed_mla=True,
                            _vllm_lod_glm53=dim == 512)
    rows = torch.arange(batch - 1, -1, -1, device="cuda", dtype=torch.int32)
    pool = VLLMLayerLODPool(layer, settings=VLLMLODSettings.production(mode=mode, pool_size=batch),
        max_requests=batch, request_capacity=8192, active_indices=rows,
        dtype=torch.bfloat16, device=torch.device("cuda"), has_query_norm=True)
    # Repeated directions give multi-page centroids; perturbations distinguish
    # the pages/leaves and prevent a uniform-value test from hiding bad strides.
    directions = torch.randn(batch, 1, 16, dim, device="cuda") * .3
    keys = (directions.repeat_interleave(128, 2) +
            torch.randn(batch, 1, 2048, dim, device="cuda") * .05).bfloat16()
    converted = pool.engine.build_cache_from_bf16(keys, keys[..., :512])
    assert isinstance(converted, KernelLODCache)
    pool.install_range(0, batch, converted)
    pool._refresh_unified_page1_coarse(tuple(range(batch)))
    # Sharp queries expose cancellation when selected coarse mass approaches
    # one; low-entropy attention is essential for exact string retrieval.
    query = (torch.randn(batch, heads, dim, device="cuda") * query_scale).bfloat16()
    current = (torch.randn(batch, 1, dim, device="cuda") * .3).bfloat16()
    # GLM's NoPE absorption returns [H,B,D].transpose(0,1), and empty_like
    # preserves that layout. The reducer must honor output strides too.
    output = torch.empty(heads, batch, 512, dtype=torch.bfloat16, device="cuda").transpose(0, 1)
    initial_lens = pool.local_lens.clone()

    def run():
        return pool.decode_dcp(query, current, current[..., :512], output)

    actual, lse = run()
    assert actual.isfinite().all() and lse.isfinite().all()
    expected, expected_lse = actual.clone(), lse.clone()
    # Check the combined field, not just each page kernel: selected parents
    # disappear and are replaced by two exact pages plus one disjoint residual.
    page = pool.state["page_cache"]
    routes = pool._dcp_buffers(query.unsqueeze(2), batch)["route_top_slots"].cpu()
    data = {name: value.cpu() for name, value in page.items() if isinstance(value, torch.Tensor)}
    for b, row in enumerate(rows.tolist()):
        n, m = int(pool.state_lens[row]), int(initial_lens[row])
        count = pool.state["counts"][row, 0, :n, 0].float().cpu()
        sums = pool.state["state_k"][row, 0, :n].float().cpu()
        for head in (0, heads - 1):
            q = query[b, head].float().cpu()
            opened = routes[b, head].flatten().tolist()
            means = (sums / count[:, None]).bfloat16().float()
            route_scores = means @ q * .0625 + count.log()
            expected_slots = route_scores.topk(8).indices
            sizes = data["slot_lengths"][row, 0, expected_slots]
            expected_slots = expected_slots.masked_fill(sizes > 1024, -1)
            assert sorted(opened) == sorted(expected_slots.tolist())
            exact = torch.cat((pool.state["recent_k"][row, 0, :m],
                               current[b], pool.state["sink_k"][row, 0]), 0).float().cpu()
            scores, values = [exact @ q * .0625], [exact[:, :512]]
            for slot in range(n):
                if slot not in opened:
                    mean = (sums[slot] / count[slot]).bfloat16().float()
                    scores.append((mean @ q * .0625 + count[slot].log().half().float())[None])
                    values.append(mean[None, :512])
                    continue
                length = int(data["slot_lengths"][row, 0, slot])
                inline = data["slot_pages"][row, 0, slot]
                page_ids = []
                for ordinal in range((length + 15) // 16):
                    if ordinal < len(inline):
                        page_ids.append(int(inline[ordinal]))
                    else:
                        match = data["overflow_page_keys"][row, 0] == slot * 65536 + ordinal
                        page_ids.append(int(data["overflow_page_values"][row, 0, match].item()))
                ids = torch.tensor(page_ids, dtype=torch.long)
                counts = data["page_counts"][row, 0, ids].float()
                if page["summary_quantization_finalized"]:
                    summary = data["quantized_page_sum_k"][row, 0, ids].float()
                    summary_scale = data["page_sum_k_scales"][row, 0, ids].float()
                    summary *= summary_scale.repeat_interleave(dim // summary_scale.size(-1), -1)
                else:
                    summary = data["page_sum_k"][row, 0, ids].float()
                ranking = (summary / counts[:, None]) @ q * .0625 + counts.log()
                chosen = ranking.topk(min(2, len(ids))).indices
                for index in chosen:
                    pid = int(ids[index])
                    leaves = data["page_indices"][row, 0, pid]
                    leaves = leaves[leaves >= 0].long()
                    if page["quantization_finalized"]:
                        packed = data["quantized_leaf_k"][row, 0, leaves].int()
                        codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).float() - 8
                        key = codes * data["page_k_scales"][row, 0, pid].float().repeat_interleave(4)
                        key += summary[index] / counts[index]
                    else:
                        key = data["leaf_k"][row, 0, leaves].float()
                    scores.append(key @ q * .0625)
                    values.append(key[:, :512])
                remaining = count[slot] - counts[chosen].sum()
                if remaining > 0:
                    mean = (sums[slot] - summary[chosen].sum(0)) / remaining
                    scores.append((mean @ q * .0625 + remaining.log())[None])
                    values.append(mean[None, :512])
            logits, field = torch.cat(scores), torch.cat(values)
            torch.testing.assert_close(actual[b, head].float().cpu(), logits.softmax(0) @ field,
                                       atol=.008, rtol=.04)
            torch.testing.assert_close(lse[b, head].cpu(), logits.logsumexp(0), atol=.002, rtol=.001)
    # Independently validate the entire baseline, including the separated sink
    # and current token. Closing all routes must become coarse-only attention.
    lengths = pool.state["page_cache"]["slot_lengths"]
    original_lengths = lengths.clone()
    lengths.fill_(1025)
    pool.local_lens.copy_(initial_lens)
    closed, closed_lse = run()
    assert pool._dcp_buffers(query.unsqueeze(2), batch)["route_top_slots"].eq(-1).all()
    state, scale = pool.state, .0625
    for b, row in enumerate(rows.tolist()):
        n, m = int(pool.state_lens[row]), int(initial_lens[row])
        count = state["counts"][row, 0, :n, 0].float()
        means = (state["state_k"][row, 0, :n].float() / count[:, None]).bfloat16().float()
        exact = torch.cat((state["recent_k"][row, 0, :m],
                           current[b], state["sink_k"][row, 0]), 0).float()
        field = torch.cat((exact, means), 0)
        bias = torch.cat((torch.zeros(len(exact), device="cuda"), count.log().half().float()))
        scores = query[b].float() @ field.T * scale + bias
        torch.testing.assert_close(closed[b].float(), scores.softmax(-1) @ field[:, :512], atol=.004, rtol=.03)
        torch.testing.assert_close(closed_lse[b], scores.logsumexp(-1), atol=.001, rtol=.001)
    lengths.copy_(original_lengths)
    # Stable output and LSE pointers, including a repeated request-row remap,
    # must survive replay without Python allocation or stale page selections.
    pool.local_lens.copy_(initial_lens)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, captured_lse = run()
    pool.local_lens.copy_(initial_lens)
    graph.replay()
    torch.testing.assert_close(captured, expected, atol=0, rtol=0)
    torch.testing.assert_close(captured_lse, expected_lse, atol=0, rtol=0)
    assert torch.equal(pool.local_lens, initial_lens + 1)
