"""Run the full-model K3 decode panel sequentially on a resident-weight node.

Each LoD measurement must contain four global 256-token catch-ups per row.
This is benchmark orchestration only: it does not change serving cadence,
attention math, or the 16K prefill scheduler budget. Run with eight GPUs and
the existing kimi-k3-shared-int4-v6 weight daemon. No cluster runner required.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/kimi-k3-full-model-current"
GIB = 1 << 30
# B8 LoD is already known not to fit a 256K-capacity engine. Do not relabel
# that warmup failure as a decode result or modify attention to force a fit.
PLANS = [
    ("two-tier", 1, (16_384, 32_768, 65_536, 131_072, 262_144), 1),
    ("full", 1, (16_384, 32_768, 65_536, 131_072, 262_144), 1),
    ("two-tier", 8, (16_384, 32_768, 65_536), 3),
    ("full", 8, (16_384, 32_768, 65_536), 3),
    ("two-tier", 8, (131_072,), 1),
    ("full", 8, (131_072,), 5),
    ("full", 8, (262_144, 524_288), 17),
]
LOD_ENV = {
    "LOD_KIMI_SUBTILE64": "score",
    "LOD_KIMI_CHUNK_TILE_PACK": "1",
    "LOD_KIMI_TILE_PACK_QUERY_BLOCK": "1024",
    "LOD_KIMI_SORT_LEAF_ROUTES": "1",
    "LOD_KIMI_LEAF_BLOCK_M": "64",
    "LOD_KIMI_LEAF_WARPS": "1",
    "LOD_KIMI_PREFILL_RECLAIM_INTERVAL": "0",
    "LOD_KIMI_REUSE_PREFILL_ALLOCATOR": "1",
    "LOD_KIMI_PREFILL_MIN_FREE_GIB": "4",
    "LOD_KIMI_TILE_REFINE": "1",
    "LOD_KIMI_DIRECT_LEAF_RESULT": "1",
}


def output_path(mode: str, batch: int, lengths: tuple[int, ...]) -> Path:
    label = "lod" if mode == "two-tier" else "full"
    if batch == 1:
        suffix = "power2"
    else:
        suffix = f"{min(lengths) // 1024}k{max(lengths) // 1024}k"
    if mode == "two-tier":
        # The old nonreplicated-query adapter skipped the DCP history merge.
        # Keep its artifacts, but never resume/render them as valid timings.
        return RESULTS / f"oct6-{label}-b{batch}-decode-{suffix}-correct-dcp-four-updates.json"
    return RESULTS / f"oct4-{label}-b{batch}-decode-{suffix}-four-updates.json"


def load_result(path: Path) -> dict | None:
    """Retain validated completed points even if a later warmup failed."""
    if not path.exists():
        path = path.with_suffix(".partial.json")
        if not path.exists():
            return None
    data = json.loads(path.read_text())
    if "decode_tokens" not in data:
        argv = data["argv"]
        for key in ("decode_tokens", "tensor_parallel_size", "decode_context_parallel_size", "batch_size"):
            data[key] = int(argv[argv.index("--" + key.replace("_", "-")) + 1])
        data["mode"] = argv[argv.index("--mode") + 1]
        data["dummy_attention"] = "--dummy-attention" in argv
    data["_source_file"] = path.name
    validate_result(data)
    return data


def sources_for(mode: str, batch: int, lengths: tuple[int, ...]):
    path = output_path(mode, batch, lengths)
    for source in (path, path.with_name(path.stem + "-remaining.json")):
        data = load_result(source)
        if data is not None:
            yield data


def validate_result(data: dict, *, outputs: int = 1_026) -> None:
    """Validate real work and counters, without source-fingerprint guards."""
    assert data["decode_tokens"] == outputs
    assert data["mode"] == "full" or outputs == 1_026
    assert data["tensor_parallel_size"] == data["decode_context_parallel_size"] == 8
    assert not data["dummy_attention"]
    workers = data["worker_attention_audit"]
    assert len(workers) == 8
    for worker in workers:
        assert worker["cudagraph_mode"] == "FULL_DECODE_ONLY"
        assert worker["dummy_attention_layers"] == 0
        assert worker["dense_gluon_decode_installed"] == (data["mode"] == "full")
    batch = data["batch_size"]
    for context, measured in data["measurements"].items():
        assert len(measured["prompts"]) == batch
        assert all(p["trace_tokens"] == outputs for p in measured["prompts"])
        assert measured["greedy_output_identical"]
        mean_window = sum(measured["decode_timings_seconds"]) / len(
            measured["decode_timings_seconds"]
        )
        assert abs(measured["decode_ms_per_batch_step"] - mean_window / (outputs - 1) * 1_000) < 1e-8
        for repetition in measured["measured_batch_timings"]:
            for timing in repetition:
                assert not any(timing["request_num_preemptions"])
                assert not any(timing["request_num_cached_tokens"])
                assert timing["last_token_spread_seconds"] == 0
                assert timing["all_requests_live_overlap_seconds"] == timing["decode_window_seconds"]
        if data["mode"] != "full":
            if data.get("request_owner_prefill"):
                assert batch == 8
                assert measured["active_attention_owner_ranks"] == list(range(8))
                assert all(replays == [outputs - 1] * 8
                           for replays in measured["owner_decode_graph_replay_deltas"])
                for repetition in measured["owner_decode_update_deltas"]:
                    assert len(repetition) == 8
                    for worker in repetition:
                        assert len(worker) == 24
                        assert all(v == {"updates": 4, "tokens": outputs - 1}
                                   for v in worker.values()), "expected four owner updates"
                for audit in measured["owner_decode_graph_audits"]:
                    assert audit["owner_layer_count"] == 24
                    assert audit["local_decode_heads"] == [96] * 24
                    assert audit["local_decode_world_sizes"] == [1] * 24
                    assert audit["active_global_cadences"] == [256] * 24
                    assert any(g["num_tokens"] == 8 and g["graph_instantiated"]
                               for g in audit["captured_graphs"])
                continue
            for repetition in measured["measured_decode_update_counters"]:
                assert len(repetition) == 8
                for worker in repetition:
                    assert len(worker) == 24
                    assert all(
                        v == {"catch_up_batches": 4, "catch_up_rows": 4 * batch}
                        for v in worker.values()
                    ), f"expected four updates per row: B{batch}, context={context}"


def load_sharded_result(path: Path, dense: dict) -> dict | None:
    """Read completed B1 sharded-prefill points without rerunning dense.

    The old long dense control emitted 1,025 tokens. Verify that entire trace
    and the prompt, then allow only its documented one-token extension for
    the four-update LoD protocol. Normalize the sweep's timing schema solely
    for this renderer; the raw measurement is never rewritten.
    """
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    points = {context: point for context, point in data.get("measurements", {}).items()
              if point.get("measurement_status") == "complete"}
    if not points:
        return None
    assert data["global_centroid_sharded_leaf_prefill"]
    assert not any(data.get(flag) for flag in (
        "capacity_only", "diagnostic_only", "request_owner_prefill",
        "transient_reconstructed_prefill_workspace"))
    assert data["mode"] == "two-tier" and data["batch_size"] == 1
    from benchmarks.prolong import token_digest

    normalized = dict(data, dummy_attention=False, measurements={}, _source_file=path.name)
    for context, point in points.items():
        assert point["worker_attention_audit_status"] == "passed"
        reference = dense["measurements"][context]
        archived = reference["prompts"][0]
        assert archived["trace_tokens"] == 1_025
        assert point["prompt_token_sha256"] == [archived["token_sha256"]]
        generated = point["generated_token_ids"]
        assert len(generated) == 1 and len(generated[0]) == 1_026
        assert token_digest(generated[0][:1_025]) == archived["trace_token_sha256"]
        extension, = data["reference_trace_extensions"][context]
        assert extension["archived_outputs"] == 1_025 and extension["measured_outputs"] == 1_026
        assert extension["archived_trace_sha256"] == archived["trace_token_sha256"]
        assert extension["extended_trace_sha256"] == token_digest(generated[0])
        normalized["measurements"][context] = dict(
            point, prompts=[dict(archived, trace_tokens=1_026)], greedy_output_identical=True,
            measured_batch_timings=[[timing] for timing in point["measured_batch_timings"]],
            decode_timings_seconds=[seconds * 1_025 for seconds in
                                   point["decode_samples_seconds_per_batch_step"]])
    validate_result(normalized)
    return normalized


def load_router_result(path: Path, reference: dict, *, owner: bool = False) -> dict | None:
    """Overlay a matched decode follow-up without rerunning dense or old points."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    points = {context: point for context, point in data.get("measurements", {}).items()
              if point.get("measurement_status") == "complete"}
    if not points:
        return None
    assert data["mode"] == "two-tier" and data["batch_size"] == reference["batch_size"]
    assert not any(data.get(flag) for flag in ("capacity_only", "diagnostic_only"))
    assert bool(data.get("request_owner_prefill")) == owner
    from benchmarks.prolong import token_digest

    normalized = dict(data, dummy_attention=False, measurements={}, _source_file=path.name,
                      router_four_waves=True)
    for context, point in points.items():
        archived = reference["measurements"][context]
        assert point["worker_attention_audit_status"] == "passed"
        assert point["prompt_token_sha256"] == [p["token_sha256"] for p in archived["prompts"]]
        hashes = [token_digest(row) for row in point["generated_token_ids"]]
        assert hashes == [p["trace_token_sha256"] for p in archived["prompts"]]
        normalized["measurements"][context] = dict(
            point, prompts=archived["prompts"], output_token_sha256=[hashes],
            greedy_output_identical=True,
            measured_batch_timings=[[timing] for timing in point["measured_batch_timings"]],
            decode_timings_seconds=[seconds * 1_025 for seconds in
                                   point["decode_samples_seconds_per_batch_step"]])
    validate_result(normalized)
    return normalized


def command(mode: str, batch: int, lengths: tuple[int, ...], cache_gib: int, *, output: Path | None = None) -> list[str]:
    return [
        str(ROOT / "benchmarks/run_kimi_k3_v10_direct.sh"), "-m", "benchmarks.prolong",
        "--checkpoint", "/tmp/dan-agent-kimi-k3-f831ab66814297da540d832a5235f8e904f29d06",
        "--mode", mode, "--measure", "speed", "--lengths", ",".join(map(str, lengths)),
        "--batch-size", str(batch), "--speed-samples", str(batch),
        "--tensor-parallel-size", "8", "--decode-context-parallel-size", "8",
        "--dcp-comm-backend", "ag_rs", "--decode-tokens", "1026", "--fixed-decode-trace",
        "--synchronized-decode", "--repeats", "1", "--seed", "0",
        "--gpu-memory-utilization", "0.8", "--kv-cache-memory-bytes", str(cache_gib * GIB),
        "--kimi-gfx942-int4-moe", "--weight-cache", "--weight-cache-id", "kimi-k3-shared-int4-v6",
        "--allow-experimental-environment", "--retain-warmup-allocator",
        "--output", str(output or output_path(mode, batch, lengths)),
    ]


def current_command(batch: int, lengths: tuple[int, ...], *, output: Path) -> list[str]:
    """Reproduce current kernels/layout, with the already frozen dense traces."""
    dense_paths = [data["_source_file"]
                   for mode, row_batch, planned, _ in PLANS
                   if mode == "full" and row_batch == batch
                   for data in sources_for(mode, row_batch, planned)
                   if set(map(str, lengths)) & set(data["measurements"])]
    if not dense_paths:
        raise RuntimeError("current LoD measurement needs an archived dense cohort")
    maximum = (263186 if batch == 1 else max(lengths) + 1042)
    return [
        str(ROOT / "benchmarks/run_kimi_k3_v10_direct.sh"), "-m",
        "benchmarks.kimi_k3_prefill_sweep", "--checkpoint",
        "/tmp/dan-agent-kimi-k3-f831ab66814297da540d832a5235f8e904f29d06",
        "--mode", "two-tier", "--batch-size", str(batch),
        "--lengths", *map(str, lengths), "--max-model-len", str(maximum),
        "--decode-tokens", "1026", "--reference-decode-trace",
        "--tensor-parallel-size", "8", "--decode-context-parallel-size", "8",
        "--gpu-memory-utilization", ".8", "--kv-cache-memory-bytes",
        str((1 if batch == 1 else 3) * GIB), "--repeats", "1",
        "--weight-cache-id", "kimi-k3-shared-int4-v6", "--real-token-cache",
        str(RESULTS / "prolong-kimi-k3-speed-token-cache.pt"),
        "--reference-baselines", *(str(RESULTS / path) for path in dense_paths),
        "--output", str(output),
    ]


def render() -> dict:
    sources = {}
    rows = {}
    for mode, batch, lengths, _ in PLANS:
        for data in sources_for(mode, batch, lengths):
            sources[data["_source_file"]] = data
            for context, measured in data["measurements"].items():
                rows[(batch, int(context), mode)] = (data, measured)
    for batch, label in ((1, "b1"), (8, "dcp-b8")):
        reference = rows.get((batch, 65_536, "two-tier"))
        if reference:
            path = RESULTS / f"oct6-router-optimized-{label}-64k.json"
            improved = load_router_result(path, reference[0])
            if improved:
                sources[path.name] = improved
                for context, measured in improved["measurements"].items():
                    rows[(batch, int(context), "two-tier")] = (improved, measured)
    for batch, filename in (
        (1, "oct6-parallel-decode-b1-64k.json"),
        (8, "oct6-parallel-decode-b8-split16-64k.json"),
        (1, "oct6-cached-means-decode-b1-64k.json"),
        (8, "oct6-cached-means-decode-b8-64k.json"),
        (1, "oct6-cached-means-decode-b1-16k128k.json"),
    ):
        reference = rows.get((batch, 65_536, "two-tier"))
        if reference:
            path = RESULTS / filename
            # A previous 64K-only overlay has just one measurement. Match a
            # multi-context follow-up against each archived cohort, not that
            # overlay's incomplete measurement map.
            cohorts = {str(context): measured
                       for (row_batch, context, mode), (_, measured) in rows.items()
                       if row_batch == batch and mode == "two-tier"}
            improved = load_router_result(path, dict(reference[0], measurements=cohorts))
            if improved:
                sources[path.name] = improved
                for context, measured in improved["measurements"].items():
                    rows[(batch, int(context), "two-tier")] = (improved, measured)
    # Match owners against the archived ordinary cohorts before discarding
    # obsolete LoD rows. Layout changes are explicit, never silently relabeled.
    owner_references = {str(context): measured
                        for (batch, context, mode), (_, measured) in rows.items()
                        if batch == 8 and mode == "two-tier"}
    current_lod_rows = {
        key: value for key, value in rows.items()
        if key[2] == "full" or (key[0] == 1 and value[0]["_source_file"] in {
            "oct6-cached-means-decode-b1-64k.json",
            "oct6-cached-means-decode-b1-16k128k.json",
        })
    }
    owner_files = (
        "oct6-cached-means-decode-owner-b8-16k.json",
        "oct6-cached-means-decode-owner-b8-64k.json",
        "oct6-cached-means-decode-owner-b8-128k-memory-r2.json",
    )
    for filename in owner_files:
        path = RESULTS / filename
        improved = load_router_result(
            path, dict(batch_size=8, measurements=owner_references), owner=True)
        if improved:
            sources[path.name] = improved
            for context, measured in improved["measurements"].items():
                current_lod_rows[(8, int(context), "two-tier")] = (improved, measured)
    long_dense_path = RESULTS / "oct4-full-b1-512k1020k-decode1025.json"
    if long_dense_path.exists():
        data = json.loads(long_dense_path.read_text())
        data["_source_file"] = long_dense_path.name
        validate_result(data, outputs=1_025)
        sources[long_dense_path.name] = data
        for context, measured in data["measurements"].items():
            rows[(1, int(context), "full")] = (data, measured)
        for key, value in rows.items():
            if key[2] == "full":
                current_lod_rows[key] = value
    rows = current_lod_rows
    # Newly requested measurements get fresh paths, never overwrite/resume
    # the archived one-wave records merely because their context is covered.
    for batch in (1, 8):
        dense_cohorts = {str(context): measured
                         for (row_batch, context, mode), (_, measured) in rows.items()
                         if row_batch == batch and mode == "full"}
        for path in sorted(RESULTS.glob(f"oct6-current-b{batch}-decode-*.json")):
            improved = load_router_result(path, dict(batch_size=batch,
                measurements=dense_cohorts), owner=batch == 8)
            if improved:
                sources[path.name] = improved
                for context, measured in improved["measurements"].items():
                    rows[(batch, int(context), "two-tier")] = (improved, measured)
    used_sources = {data["_source_file"] for data, _ in rows.values()}
    sources = {name: data for name, data in sources.items() if name in used_sources}
    lines = [
        "# Kimi K3 four-update decode panel", "",
        "October 6: LoD rows use the repaired nonreplicated-query DCP dispatch",
        "and authoritative live-tail cache conversion. The earlier `oct4-lod-*`",
        "decode measurements attended to incomplete history and are invalid;",
        "they are not included or resumed. Existing dense controls are unaffected.", "",
        "Only current LoD kernels are shown: four-wave",
        "routing, parallel split reduction, fused distributed-route bookkeeping",
        "and cached physical centroid means, with 32 splits per physical B1.",
        "B1 uses ordinary DCP8; B8 uses **one request's attention per GPU**",
        "(all 96 heads on its owner, native TP8 projections/KDA/MoE).",
        "Old one-wave and ordinary-B8 LoD timings are removed from this table.",
        "No current run exists at the blank contexts; no gains are extrapolated.",
        "See [cached routing](CACHED_ROUTING.md) for matching audits and commands.", "",
        "Full model, TP8/DCP8/EP8, real ProLong traces, seed 0, one full-shape warmup",
        "and one measured pass. 1,026 output tokens produce 1,025 timed decode steps.",
        "Each LoD layer/rank must record exactly four catch-ups per request.",
        "Times are end-to-end ms per batched decode step, including state updates.",
        "A blank LoD cell means the current-kernel run has not completed, not the old",
        "invalid measurement. All prompts/forced continuations must match dense.", "",
        "| Context | B1 dense | B1 LoD | Dense / LoD | B8 dense | B8 row-per-GPU LoD | Dense / LoD |",
        "|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for context in (16_384, 32_768, 65_536, 131_072, 262_144, 524_288, 1_044_480):
        cells = [f"{context // 1024}K"]
        for batch in (1, 8):
            pair = [rows.get((batch, context, mode)) for mode in ("full", "two-tier")]
            for item in pair:
                cells.append(f"{item[1]['decode_ms_per_batch_step']:.3f}" if item else "—")
            if all(pair):
                full_data, full = pair[0]
                lod_data, lod = pair[1]
                if not lod_data.get("global_centroid_sharded_leaf_prefill"):
                    assert full["prompts"] == lod["prompts"], (batch, context, "prompts")
                    assert full["output_token_sha256"] == lod["output_token_sha256"]
                if "timing_protocol" in full_data and "timing_protocol" in lod_data:
                    assert full_data["timing_protocol"] == lod_data["timing_protocol"]
                cells.append(f"{full['decode_ms_per_batch_step'] / lod['decode_ms_per_batch_step']:.3f}x")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "A dash is not an estimate. The replicated archive previously failed",
              "B1/512K and B8/256K warmup on VRAM. B1 dense 512K/1020K values reuse",
              "the already validated 1,024-step controls from README.md; dense has",
              "no LoD catch-ups to amortize. Their old LoD counterparts are not",
              "current-kernel measurements and therefore remain blank here.",
              "The B8/128K memory-safe retry completed all audits, but its prefill",
              "was 202.915 s versus dense's 155.750 s, including allocator",
              "reclamation. Its decode speedup does not imply a prefill speedup."]
    compact = [f"B{batch}/{context // 1024}K" for (batch, context, mode), (data, _) in rows.items()
               if mode == "two-tier"
               and data.get("benchmark_environment", {}).get("LOD_KIMI_COMPACT_SELECTED_PROJECTION") == "1"]
    if compact:
        lines += ["", "The " + ", ".join(compact) + " endpoint uses compact selected-leaf projection",
                  "during prefill only, to retry the earlier launch-resource failure.",
                  "Its decode kernel/math, top-eight selection, global cadence and",
                  "four-update timing protocol are unchanged. Memory snapshots run",
                  "only in untimed warmup; no observer runs in measured generation."]
    lines += ["", "## Reproduction", "",
              "With the resident full-model weight daemon and local checkpoint paths",
              "configured in `benchmarks/kimi_k3_decode_power2.py`, from the repo root:", "",
              "```bash", "bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_decode_power2 --run",
              "```", "", "The driver skips completed audited points and reuses unaffected dense controls.",
              "To refresh the Markdown from completed/partial results without GPU work:", "",
              "```bash", "python -m benchmarks.kimi_k3_decode_power2", "```",
              "", "## Sources", ""]
    lines.extend(f"- [{name}]({name})" for name in sources)
    (RESULTS / "DECODE_POWER2.md").write_text("\n".join(lines) + "\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run missing measurements sequentially.")
    args = parser.parse_args()
    current = render()
    if not args.run:
        return
    for mode, batch, lengths, cache_gib in PLANS:
        path = output_path(mode, batch, lengths)
        completed = {context for row_batch, context, row_mode in current
                     if row_batch == batch and row_mode == mode}
        missing = tuple(n for n in lengths if n not in completed)
        if not missing:
            continue
        if mode == "two-tier":
            path = RESULTS / f"oct6-current-b{batch}-decode-{min(missing)//1024}k{max(missing)//1024}k.json"
        elif completed:
            # The present partial B1 run completed through 128K and failed
            # at 256K. Its missing endpoint has the same max-prefix cohort.
            if max(missing) != max(lengths):
                raise RuntimeError("resumption would change the shared maximum-prefix cohort")
            path = path.with_name(path.stem + "-remaining.json")
        env = os.environ.copy()
        for name in LOD_ENV:
            env.pop(name, None)
        env.update(
            LOD_BENCHMARK_SYNC_PREFILL_CACHE="1", TRITON_CACHE_AUTOTUNING="1",
            VLLM_USE_TRITON_AWQ="1",
            PROLONG_SPEED_TOKEN_CACHE=str(RESULTS / "prolong-kimi-k3-speed-token-cache.pt"),
            AITER_CONFIG_FMOE=str(RESULTS / "kimik3_i4_tuned_fmoe_b2x16k_merged.csv"),
        )
        if mode == "two-tier":
            env.update(LOD_ENV)
        if max(lengths) >= 524_288:
            env["HSA_NO_SCRATCH_RECLAIM"] = "0"
        print(f"K3 decode panel: {mode}, B{batch}, lengths={missing}", flush=True)
        argv = (current_command(batch, missing, output=path) if mode == "two-tier"
                else command(mode, batch, missing, cache_gib, output=path))
        process = subprocess.run(argv, cwd=ROOT, env=env, check=False)
        if process.returncode:
            path.with_suffix(".failure.json").write_text(json.dumps({
                "returncode": process.returncode, "argv": argv,
                "status": "failed; no timing inferred for unfinished points",
            }, indent=2) + "\n")
        elif mode == "full":
            validate_result(json.loads(path.read_text()))
        current = render()
    print("K3 decode panel jobs finished; missing timings stay blank", flush=True)


if __name__ == "__main__":
    main()
