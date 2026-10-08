"""Offline correctness/selection checks on real GLM final-prefill tensors.

This is not a serving option or a timing benchmark. Capture sixteen queries
from one request at every MLA layer; test the existing leaf consumer and exact
coarse replacement, then change ONLY which eight regions are opened.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch


def directory_owners(cache, state_len):
    """Independently enumerate the two-level directory (CPU, shared KV head)."""
    if not cache["paged_page_directory"] or cache["slot_lengths"].shape[:2] != (1, 1):
        raise ValueError("diagnostic requires one shared-head row and direct directories")
    n = int(cache["leaf_count"])
    owners = torch.full((n,), -1, dtype=torch.int64)
    roots = cache["slot_pages"][0, 0].cpu().long()
    directories = cache["overflow_page_values"][0, 0].cpu().long()
    indices = cache["page_indices"][0, 0].cpu().long()
    lengths = cache["slot_lengths"][0, 0, :state_len].cpu().long()
    for slot, count in enumerate(lengths.tolist()):
        ordinals = torch.arange((count + 15) // 16)
        if not count:
            continue
        directory = roots[slot, ordinals // 64]
        if (directory < 0).any() or (directory >= directories.size(0)).any():
            raise AssertionError("missing/out-of-range directory row")
        pages = directories[directory, ordinals % 64]
        if (pages < 0).any() or (pages >= indices.size(0)).any():
            raise AssertionError("missing/out-of-range physical page")
        leaves = indices[pages].flatten()[:count]
        if (leaves < 0).any() or (leaves >= n).any():
            raise AssertionError("missing/out-of-range chronological leaf")
        if leaves.unique().numel() != count or (owners[leaves] >= 0).any():
            raise AssertionError("a leaf appears more than once")
        owners[leaves] = slot
    if (owners < 0).any() or lengths.sum().item() != n:
        raise AssertionError("directory does not cover all remote leaves exactly once")
    return owners


def relative_error(actual, reference):
    delta = actual.float() - reference.float()
    per_query = delta.norm(dim=-1) / reference.float().norm(dim=-1).clamp_min(1e-10)
    return dict(relative_l2=(delta.norm() / reference.float().norm().clamp_min(1e-10)).item(),
                mean_head_query_relative_l2=per_query.mean().item(),
                max_head_query_relative_l2=per_query.max().item(),
                max_absolute=delta.abs().max().item())


def region_oracle(scores, owners, regions, *, mode):
    """Actual leaf max or log-sum-exp, never centroid scores, per head/query."""
    shape = (*scores.shape[:-1], regions)
    index = owners.view(*([1] * (scores.ndim - 1)), -1).expand_as(scores)
    maximum = scores.new_full(shape, -float("inf"))
    maximum.scatter_reduce_(-1, index, scores, reduce="amax", include_self=True)
    if mode == "max":
        return maximum
    if mode != "mass":
        raise ValueError("choose max or mass")
    summed = torch.zeros_like(maximum)
    summed.scatter_add_(-1, index, (scores - maximum.gather(-1, index)).exp())
    return maximum + summed.log()


def selection_mask(selection, regions):
    """Sentinels must not overwrite a genuine route to region zero."""
    opened = torch.zeros((*selection.shape[:-1], regions), dtype=torch.int32, device=selection.device)
    opened.scatter_add_(-1, selection.clamp_min(0).long(), (selection >= 0).int())
    return opened.bool()


def install_capture(worker, *, slot=1, queries=16):
    """RPC after model initialization; never change attention outputs."""
    from vllm_lod_plugin.models import glm53_flash as glm
    from lod_attention.kernels import glm_projected_prefill as kernels
    layers = [(name, layer) for name, layer in worker.model_runner.get_model().named_modules()
              if getattr(layer, "_vllm_lod_glm53", False)]
    if not layers:
        raise RuntimeError("no trained GLM LoD layers to capture")
    for name, layer in layers:
        layer._vllm_lod_pool.engine._glm53_capture_name = name
    original_attention = glm.latent_attention
    original_projected = kernels.projected_two_level_attention

    def attention(layer, query, latent, direct_key, *args, **kwargs):
        pool = layer._vllm_lod_pool
        plan = pool.direct_prefill_plan
        final = plan is not None and any(
            s == slot and previous > 0 and previous + end - begin == pool.direct_prefill_prompt_lengths[s]
            for s, begin, end, previous in plan)
        pool.engine._glm53_capture_final = final
        try:
            return original_attention(layer, query, latent, direct_key, *args, **kwargs)
        finally:
            pool.engine._glm53_capture_final = False

    def projected(engine, q, local_k, local_v, state_k, counts, **kwargs):
        output = original_projected(engine, q, local_k, local_v, state_k, counts, **kwargs)
        if not getattr(engine, "_glm53_capture_final", False):
            return output
        if q.size(0) != 1:
            raise RuntimeError("capture expects the usual single-row scheduler prefill")
        rows = min(queries, q.size(2))
        cache = kwargs["page_cache"]
        n = int(cache["leaf_count"])
        # The original call has already waited for its local attention stream.
        # Blocking CPU copies also finish outstanding foreground kernels before
        # an asynchronous update can reuse their workspace. No GPU snapshot
        # storage is kept across layers, and nothing changes serving outputs.
        names = ("slot_pages", "overflow_page_keys", "overflow_page_values", "overflow_used",
                 "slot_lengths", "page_indices", "leaf_lens", "next_page", "overflow_flag")
        snapshot_cache = {name: cache[name].detach().cpu().clone() for name in names}
        for name in ("leaf_count", "leaf_capacity", "paged_page_directory", "page_size",
                     "overflow_active", "overflow_safe_until", "overflow_hash_capacity", "region_owned_pages"):
            if name in cache:
                snapshot_cache[name] = cache[name]
        # No leaf sealing is enabled for GLM; enumerate the actual directory
        # before doing any rebuilt all-exposed control.
        snapshot_cache["leaf_k"] = cache["leaf_k"][..., :n, :].detach().cpu().clone()
        snapshot_cache["leaf_v"] = snapshot_cache["leaf_k"]
        snapshot_cache["leaf_capacity"] = n
        engine._glm53_diagnostic_snapshot = dict(
            layer=engine._glm53_capture_name, slot=slot,
            q=q[..., -rows:, :].detach().cpu().clone(),
            expanded_q=engine._lod_kimi_expanded_prefill_chunk[..., -rows:, :].detach().cpu().clone(),
            normal_output=output[..., -rows:, :].detach().cpu().clone(),
            local=local_k.detach().cpu().clone(), sink=kwargs["sink_k"].detach().cpu().clone(),
            state=state_k.detach().cpu().clone(), counts=counts.detach().cpu().clone(),
            state_len=kwargs["state_len"], scale=engine.scaling,
            uk=engine._lod_kimi_w_uk_t.detach().cpu().clone(),
            uv=engine._lod_kimi_w_uv.detach().cpu().clone(), page_cache=snapshot_cache)
        print("GLM53_CAPTURE " + engine._glm53_capture_name + f" queries={rows} leaves={n}", flush=True)
        return output

    glm.latent_attention = attention
    kernels.projected_two_level_attention = projected
    return dict(layers=len(layers), slot=slot, queries=queries)


def analyze_snapshot(engine, snapshot):
    """GPU replay, dense FP32 control, and actual-leaf region-selection oracles."""
    from lod_attention.kernels.glm_projected_prefill import projected_route_coarse, projected_local_attention
    from lod_attention.kernels.aiter_mla_prefill_attention import (
        project_kimi_head_values, project_kimi_shared_values, merge_aiter_mla_prefill_refinement)
    cpu_owners = directory_owners(snapshot["page_cache"], snapshot["state_len"])
    device = torch.device("cuda", torch.cuda.current_device())
    data = {name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in snapshot.items()}
    cache = {name: value.to(device) if isinstance(value, torch.Tensor) else value
             for name, value in snapshot["page_cache"].items()}
    owners = cpu_owners.to(device)
    q, xq, local, sink, state, counts, uk, uv = (
        data[name] for name in ("q", "expanded_q", "local", "sink", "state", "counts", "uk", "uv"))
    n, regions, rows = owners.numel(), data["state_len"], q.size(2)
    leaves = cache["leaf_k"]
    scale = data["scale"]
    local_out, local_lse = projected_local_attention(xq, local, uk, uv,
        query_offset=local.size(2) - rows, scale=scale, buffers={})
    projected_sink = project_kimi_shared_values(sink, uv)
    slots, coarse = projected_route_coarse(xq, state, counts, uk, uv, state_len=regions,
        scale=scale, slot_lengths=cache["slot_lengths"], max_open_leaf_tokens=1024, buffers={})
    # Reference uses the same absorbed BF16 q as refined leaves/sink. Local
    # remains projected BF16, as in serving. All arithmetic below is FP32.
    remote_scores = torch.matmul(q.float(), leaves[0, 0].float().T) * scale
    local_keys = torch.einsum("bnt,hdt->bhnd", local[:, 0].float(), uk.float()).to(local.dtype)
    local_values = torch.einsum("bnt,htd->bhnd", local[:, 0].float(), uv.float()).to(local.dtype)
    local_scores = torch.matmul(xq.float(), local_keys.float().transpose(-1, -2)) * scale
    mask = torch.arange(local.size(2), device=device)[None, :] > (
        torch.arange(rows, device=device)[:, None] + local.size(2) - rows)
    local_scores.masked_fill_(mask, -float("inf"))
    sink_scores = torch.matmul(q.float(), sink[0, 0].float().T) * scale
    dense_lse = torch.logsumexp(torch.cat((remote_scores, local_scores, sink_scores), dim=-1), dim=-1)
    remote_latent = torch.matmul((remote_scores - dense_lse[..., None]).exp(), leaves[0, 0].float())
    dense = torch.einsum("bhqt,htd->bhqd", remote_latent, uv.float())
    dense += torch.matmul((local_scores - dense_lse[..., None]).exp(), local_values.float())
    sink_values = torch.einsum("bnt,htd->bhnd", sink[:, 0].float(), uv.float())
    dense += torch.matmul((sink_scores - dense_lse[..., None]).exp(), sink_values)

    # The independent dense MLA ablation uses expanded BF16 K/V, rather
    # than absorbed BF16 queries. Quantify that rounding difference too,
    # one head at a time so there is no full expanded history allocation.
    expanded_dense = torch.empty_like(dense)
    for head in range(q.size(1)):
        rk = torch.nn.functional.linear(leaves[0, 0].float(), uk[head].float()).to(q.dtype)
        rv = (leaves[0, 0].float() @ uv[head].float()).to(q.dtype)
        sk = torch.nn.functional.linear(sink[0, 0].float(), uk[head].float()).to(q.dtype)
        sv = (sink[0, 0].float() @ uv[head].float()).to(q.dtype)
        rs = (xq[0, head].float() @ rk.float().T) * scale
        ss = (xq[0, head].float() @ sk.float().T) * scale
        ls = local_scores[0, head]
        denominator = torch.cat((rs, ls, ss), -1).logsumexp(-1)
        expanded_dense[0, head] = (rs - denominator[:, None]).exp() @ rv.float()
        expanded_dense[0, head] += (ls - denominator[:, None]).exp() @ local_values[0, head].float()
        expanded_dense[0, head] += (ss - denominator[:, None]).exp() @ sv.float()

    # Audit raw sum/count invariants independently of the attention kernels.
    sums = torch.zeros((regions, 512), dtype=torch.float32, device=device)
    sums.index_add_(0, owners, leaves[0, 0].float())
    lengths = cache["slot_lengths"][0, 0, :regions]
    count_error = (counts[0, 0, :regions, 0].float() - lengths.float()).abs().max().item()
    sum_error = relative_error(state[0, 0, :regions], sums)

    # Original coarse scores are needed for subtraction even when an oracle
    # chooses the regions. Oracle scores must NEVER replace these scores.
    means = (state[:, 0, :regions].float() / counts[:, 0, :regions].float().clamp_min(1)).to(state.dtype)
    centroid_keys = torch.einsum("bnt,hdt->bhnd", means.float(), uk.float()).to(state.dtype)
    summary_scores = torch.matmul(xq.float(), centroid_keys.float().transpose(-1, -2)) * scale
    # Bias is BF16 in the source-derived CK coarse kernel.
    summary_scores += counts[:, 0, :regions, 0].float().log().to(q.dtype).float()[:, None, None, :]
    summary_values = coarse.mean_v[..., :regions, :].float()
    global_top_owner = owners[remote_scores.argmax(-1)]

    def run_selection(selected, *, region_cache=cache, region_state_len=regions,
                      region_coarse=coarse, selected_scores=None):
        selected = selected.contiguous().to(torch.int32)
        if selected_scores is None:
            selected_scores = summary_scores.gather(-1, selected.clamp_min(0).long())
            selected_scores = selected_scores.masked_fill(selected < 0, -float("inf"))
        replaced = replace(region_coarse, selected_route_scores=selected_scores.contiguous())
        refined, lse = engine._paged_leaf_attention(q, selected, region_cache,
                                                   active_slots=region_state_len, reduce_routes=True)
        refined = project_kimi_head_values(refined, uv)
        output = merge_aiter_mla_prefill_refinement(q, sink, projected_sink, replaced, selected,
            refined, lse, local_out, local_lse, kv_group_size=q.size(1), scale=scale)
        return output

    result = dict(layer=data["layer"], slot=data["slot"], heads=q.size(1), queries=rows,
        remote_leaves=n, local_tokens=local.size(2), regions=regions,
        captured_query_stride=list(q.stride()), captured_query_contiguous=q.is_contiguous(),
        directory_covers_all_leaves=True, count_max_absolute_error=count_error,
        state_sum_error=sum_error, normal_output_vs_dense=relative_error(data["normal_output"], dense),
        absorbed_vs_expanded_dense=relative_error(dense, expanded_dense))
    local_reference = local_scores.softmax(-1) @ local_values.float()
    summary_reference = summary_scores.softmax(-1) @ summary_values
    result["components"] = dict(
        local_output=relative_error(local_out, local_reference),
        local_lse_max_error=(local_lse - local_scores.logsumexp(-1)).abs().max().item(),
        coarse_output=relative_error(coarse.output_0.permute(0, 2, 1, 3), summary_reference),
        coarse_lse_max_error=(coarse.lse_0 - summary_scores.logsumexp(-1)).abs().max().item(),
        coarse_lse_shape=list(coarse.lse_0.shape), coarse_lse_stride=list(coarse.lse_0.stride()))
    normal_replay = run_selection(slots, selected_scores=coarse.selected_route_scores)
    result["normal_replay_vs_captured"] = relative_error(normal_replay, data["normal_output"])

    # Collapse only this control into eight superregions. Exposing all eight
    # still uses the actual indexed leaf consumer + actual replacement kernel;
    # original directory coverage was checked above to avoid hiding lost KV.
    control_owners = (owners % 8).view(1, 1, -1).to(torch.int32)
    control_counts = torch.bincount(control_owners.flatten().long(), minlength=8).float().view(1, 1, 8, 1)
    control_state = torch.zeros((8, 512), dtype=torch.float32, device=device)
    control_state.index_add_(0, control_owners.flatten().long(), leaves[0, 0].float())
    control_state = control_state.view(1, 1, 8, 512).to(q.dtype)
    control_cache = engine._new_page_cache(leaves, leaves, control_owners,
        state_capacity=8, sequence_capacity=n, virtual_k=leaves, virtual_v=leaves)
    control_slots, control_coarse = projected_route_coarse(xq, control_state, control_counts, uk, uv,
        state_len=8, scale=scale, slot_lengths=control_cache["slot_lengths"],
        max_open_leaf_tokens=None, buffers={})
    all_refined, all_refined_lse = engine._paged_leaf_attention(q, control_slots, control_cache,
        active_slots=8, reduce_routes=True)
    remote_reference = torch.matmul(remote_scores.softmax(-1), leaves[0, 0].float())
    result["components"].update(
        all_remote_latent_output=relative_error(all_refined, remote_reference),
        all_remote_lse_max_error=(all_refined_lse - remote_scores.logsumexp(-1)).abs().max().item(),
        control_slots=control_slots[0, 0, 0].tolist())
    control_output = run_selection(control_slots, region_cache=control_cache, region_state_len=8,
        region_coarse=control_coarse, selected_scores=control_coarse.selected_route_scores)
    result["all_exposed_vs_dense"] = relative_error(control_output, dense)
    result["all_exposed_vs_expanded_dense"] = relative_error(control_output, expanded_dense)

    result["selections"] = {}
    for name, selection in [("normal_capped", slots),
            ("normal_uncapped", summary_scores.topk(8, dim=-1).indices),
            ("leaf_max_oracle", region_oracle(remote_scores, owners, regions, mode="max").topk(8, dim=-1).indices),
            ("leaf_mass_oracle", region_oracle(remote_scores, owners, regions, mode="mass").topk(8, dim=-1).indices)]:
        # Literal FP32 mixed exact/coarse reference independently checks the
        # kernel arithmetic on each oracle's original region directory.
        opened = selection_mask(selection, regions)
        exact_mask = opened[..., owners]
        exact_scores = remote_scores.masked_fill(~exact_mask, -float("inf"))
        remaining_scores = summary_scores.masked_fill(opened, -float("inf"))
        mixed_lse = torch.logsumexp(torch.cat((exact_scores, remaining_scores, local_scores, sink_scores), -1), -1)
        mixed_latent = torch.matmul((exact_scores - mixed_lse[..., None]).exp(), leaves[0, 0].float())
        mixed = torch.einsum("bhqt,htd->bhqd", mixed_latent, uv.float())
        mixed += torch.matmul((remaining_scores - mixed_lse[..., None]).exp(), summary_values)
        mixed += torch.matmul((local_scores - mixed_lse[..., None]).exp(), local_values.float())
        mixed += torch.matmul((sink_scores - mixed_lse[..., None]).exp(), sink_values)
        output = normal_replay if name == "normal_capped" else run_selection(selection)
        result["selections"][name] = dict(
            output_vs_dense=relative_error(output, dense), fp32_mixed_vs_dense=relative_error(mixed, dense),
            kernel_vs_fp32_mixed=relative_error(output, mixed),
            top_remote_leaf_owner_open_fraction=opened.gather(-1, global_top_owner[..., None]).float().mean().item(),
            opened_remote_mass_of_full_attention=((remote_scores - dense_lse[..., None]).exp() * exact_mask).sum(-1).mean().item(),
            opened_leaf_tokens_mean=lengths[selection.clamp_min(0).long()].mul(selection >= 0).sum(-1).float().mean().item())
    return result


def analyze_worker(worker, *, save_dir=None):
    """One RPC after quality generation; no collectives or outputs modified."""
    from vllm.distributed import get_tp_group
    rank = get_tp_group().rank_in_group
    layers = [layer for layer in worker.model_runner.get_model().modules()
              if getattr(layer, "_vllm_lod_glm53", False)]
    # Persist the entire cohort first, so a diagnostic-only replay failure
    # never requires another huge model load or another generation pass.
    if save_dir:
        directory = Path(save_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for layer in layers:
            snapshot = getattr(layer._vllm_lod_pool.engine, "_glm53_diagnostic_snapshot", None)
            if snapshot is None:
                raise AssertionError("a requested real-model MLA snapshot was not captured")
            torch.save(snapshot, directory / f"rank{rank}-{snapshot['layer']}.pt")
    results = []
    for layer in layers:
        engine = layer._vllm_lod_pool.engine
        snapshot = getattr(engine, "_glm53_diagnostic_snapshot", None)
        if snapshot is None:
            raise AssertionError("a requested real-model MLA snapshot was not captured")
        original_buffers = getattr(engine, "_lod_prefill_attention_buffers", None)
        engine._lod_prefill_attention_buffers = {}
        try:
            with torch.no_grad():
                report = analyze_snapshot(engine, snapshot)
            results.append(report)
            print("GLM53_DIAGNOSTIC " + str(dict(layer=report["layer"], rank=rank,
                all_exposed=report["all_exposed_vs_dense"]["relative_l2"],
                normal=report["normal_output_vs_dense"]["relative_l2"])), flush=True)
        finally:
            engine._lod_prefill_attention_buffers = original_buffers
            del engine._glm53_diagnostic_snapshot
    return dict(rank=rank, layers=results, scope="16 final-prefill queries of logical request slot 1; not oracle generation")


def main():
    """Replay a saved node-local snapshot without loading any model weights."""
    import argparse
    import json
    from lod_attention._config import ModelFamily, LODMode
    from lod_attention._engines import KernelTwoLevelLODAttention
    from lod_attention._profile import configure_engine
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("snapshot", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    snapshot = torch.load(args.snapshot, map_location="cpu", weights_only=True)
    engine = KernelTwoLevelLODAttention(query_heads=snapshot["q"].size(1), key_value_heads=1,
                                       scale=snapshot["scale"])
    engine.head_dim = 512
    configure_engine(engine, family=ModelFamily.GLM53_FLASH, mode=LODMode.TWO_TIER,
                     request_capacity=65536, has_query_norm=True, has_key_norm=False)
    engine._lod_prefill_attention_buffers = {}
    with torch.no_grad():
        report = analyze_snapshot(engine, snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
