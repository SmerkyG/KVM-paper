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
            for repetition in measured["measured_decode_update_counters"]:
                assert len(repetition) == 8
                for worker in repetition:
                    assert len(worker) == 24
                    assert all(
                        v == {"catch_up_batches": 4, "catch_up_rows": 4 * batch}
                        for v in worker.values()
                    ), f"expected four updates per row: B{batch}, context={context}"


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


def render() -> None:
    sources = {}
    rows = {}
    for mode, batch, lengths, _ in PLANS:
        for data in sources_for(mode, batch, lengths):
            sources[data["_source_file"]] = data
            for context, measured in data["measurements"].items():
                rows[(batch, int(context), mode)] = (data, measured)
    long_dense_path = RESULTS / "oct4-full-b1-512k1020k-decode1025.json"
    if long_dense_path.exists():
        data = json.loads(long_dense_path.read_text())
        validate_result(data, outputs=1_025)
        sources[long_dense_path.name] = data
        for context, measured in data["measurements"].items():
            rows[(1, int(context), "full")] = (data, measured)
    lines = [
        "# Kimi K3 four-update decode panel", "",
        "Full model, TP8/DCP8/EP8, real ProLong traces, seed 0, one full-shape warmup",
        "and one measured pass. 1,026 output tokens produce 1,025 timed decode steps.",
        "Each LoD layer/rank must record exactly four catch-ups per request.",
        "Times are end-to-end ms per batched decode step, including state updates.", "",
        "| Context | B1 dense | B1 LoD | Dense / LoD | B8 dense | B8 LoD | Dense / LoD |",
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
                assert full["prompts"] == lod["prompts"], (batch, context, "prompts")
                assert full["output_token_sha256"] == lod["output_token_sha256"]
                if "timing_protocol" in full_data and "timing_protocol" in lod_data:
                    assert full_data["timing_protocol"] == lod_data["timing_protocol"]
                cells.append(f"{full['decode_ms_per_batch_step'] / lod['decode_ms_per_batch_step']:.3f}x")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "A dash is not an estimate. B8 LoD at 256K+ and B1 LoD at 512K",
              "previously failed warmup on VRAM. B1 dense 512K/1020K values reuse",
              "the already validated 1,024-step controls from README.md; dense has",
              "no LoD catch-ups to amortize. They are not paired with a LoD result.",
              "A partial source contributes only its completed, individually validated",
              "points; a later warmup failure does not create another timing point.", "", "## Sources", ""]
    lines.extend(f"- [{name}]({name})" for name in sources)
    (RESULTS / "DECODE_POWER2.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run missing measurements sequentially.")
    args = parser.parse_args()
    render()
    if not args.run:
        return
    for mode, batch, lengths, cache_gib in PLANS:
        path = output_path(mode, batch, lengths)
        completed = {int(n) for data in sources_for(mode, batch, lengths)
                     for n in data["measurements"]}
        missing = tuple(n for n in lengths if n not in completed)
        if not missing:
            continue
        if completed:
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
        argv = command(mode, batch, missing, cache_gib, output=path)
        process = subprocess.run(argv, cwd=ROOT, env=env, check=False)
        if process.returncode:
            path.with_suffix(".failure.json").write_text(json.dumps({
                "returncode": process.returncode, "argv": argv,
                "status": "failed; no timing inferred for unfinished points",
            }, indent=2) + "\n")
        else:
            validate_result(json.loads(path.read_text()))
        render()
    print("K3 decode panel jobs finished; missing timings stay blank", flush=True)


if __name__ == "__main__":
    main()
