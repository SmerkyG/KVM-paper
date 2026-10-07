"""Render only freshly audited, matched K3 serving measurements.

Reads atomic partial-result snapshots as a long sweep progresses. Never mixes
old dense controls, fixture timings, warmup durations, or failed capacity runs
into the current speedup tables. No GPU work is performed by this renderer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/kimi-k3-full-model-current"
LENGTHS = (16384, 32768, 65536, 131072, 262144, 524288, 1044480)


def validate_point(data, point):
    if point.get("measurement_status") != "complete":
        return False
    if data.get("capacity_only") or data.get("diagnostic_only"):
        return False
    if (data.get("decode_tokens") != 1026 or data.get("repeats") != 1
            or data.get("kda_prefill") != "gluon_paged"
            or not data.get("weight_cache_id") or not data.get("real_token_cache")):
        raise ValueError("not the current trained, four-update, one-pass protocol")
    if point.get("worker_attention_audit_status") != "passed":
        raise ValueError("missing loaded-attention audit")
    batch = data["batch_size"]
    outputs = point["generated_token_ids"]
    if len(outputs) != batch or any(len(row) != 1026 for row in outputs):
        raise ValueError("wrong live request/output count")
    if len(point["prefill_samples_seconds"]) != 1 or len(point["decode_samples_seconds_per_batch_step"]) != 1:
        raise ValueError("one measured pass is required")
    for timing in point["measured_batch_timings"]:
        if any(timing["request_num_preemptions"]) or any(timing["request_num_cached_tokens"]):
            raise ValueError("preemption or prefix reuse invalidates this speed panel")
        if batch > 1 and (timing["last_token_spread_seconds"] != 0
                or timing["all_requests_live_overlap_seconds"] != timing["decode_window_seconds"]):
            raise ValueError("decode did not keep the entire cohort live")
    if data["mode"] == "two-tier":
        if data["request_owner_prefill"]:
            if point["owner_decode_graph_replay_deltas"] != [[1025] * 8]:
                raise ValueError("the B8 model graph did not execute every step")
            for worker in point["owner_decode_update_deltas"][0]:
                if len(worker) != 24 or any(v != {"updates": 4, "tokens": 1025} for v in worker.values()):
                    raise ValueError("wrong owner update cadence")
        else:
            for worker in point["measured_decode_update_counters"][0]:
                if len(worker) != 24 or any(v != {"catch_up_batches": 4, "catch_up_rows": 4 * batch}
                                           for v in worker.values()):
                    raise ValueError("wrong global DCP update cadence")
    return True


def matched(dense, lod):
    if dense["prompt_token_sha256"] != lod["prompt_token_sha256"]:
        raise ValueError("dense/LoD prompts differ")
    if dense["generated_token_ids"] != lod["generated_token_ids"]:
        raise ValueError("dense/LoD forced continuations differ")


def render(directory=RESULTS):
    points, sources, progress = {}, [], []
    for path in sorted(directory.glob("oct7-current-*.json")):
        data = json.loads(path.read_text())
        sources.append(path.name)
        progress.append(f"- `{path.name}`: {data.get('measurement_status', 'unknown')}; "
                        f"{data.get('current_phase', {})}")
        for context, point in data.get("measurements", {}).items():
            if validate_point(data, point):
                key = (data["mode"], data["batch_size"], int(context))
                if key in points:
                    raise ValueError(f"ambiguous current source for {key}; archive the superseded JSON")
                points[key] = point
    prefill, decode = [], []
    for batch in (1, 8):
        for length in LENGTHS:
            full, lod = (points.get((mode, batch, length)) for mode in ("full", "two-tier"))
            if full and lod:
                matched(full, lod)
            row = [f"B{batch}", f"{length // 1024}K"]
            for target, field, digits in ((prefill, "prefill_seconds", 3),
                                          (decode, "decode_ms_per_batch_step", 3)):
                cells = [f"{point[field]:.{digits}f}" if point else "—" for point in (full, lod)]
                ratio = f"{full[field] / lod[field]:.3f}×" if full and lod else "—"
                target.append("| " + " | ".join(row + cells + [ratio]) + " |")
    note = (
        "Full trained K3; TP8/DCP8/EP8, eight MI325X GPUs, frozen real ProLong prompts "
        "and identical teacher-forced continuations. Both arms use the approved G8 "
        "direct-state-I/O KDA prefill baseline and the same packed INT4 MoE weights. "
        "Attention caches remain BF16. Dense decode uses the improved Gluon kernel. "
        "LoD B1 uses DCP8; B8 uses one request's attention per GPU.\n\n"
        "One exact-shape untimed warmup and one measured pass, 1,026 output tokens / "
        "1,025 decode steps, four global-256 catch-ups per LoD request/layer. "
        "No prefix hits, preemptions, profiling events, or warmup times in these cells. "
        "A dash means no completed fresh measurement—not a reused older result.\n\n"
    )
    columns = "| Batch | Context | Dense | Two-tier LoD | Dense / LoD |\n|:--|--:|--:|--:|--:|\n"
    path = directory / "CURRENT_TIMINGS.md"
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text("# Current K3 matched speed sweep\n\n" + note
        + "## Prefill (seconds per batch)\n\n" + columns + "\n".join(prefill)
        + "\n\n## Decode (milliseconds per batch step)\n\n" + columns + "\n".join(decode)
        + "\n\n## Sweep status\n\n" + "\n".join(progress)
        + "\n\n## Raw sources\n\n" + "\n".join(f"- [{name}]({name})" for name in sources) + "\n")
    temporary.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=RESULTS)
    args = parser.parse_args()
    print(render(args.directory))


if __name__ == "__main__":
    main()
