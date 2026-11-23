"""Check matched GLM decode graphs, per-layer dispatch, timing scope, and fixed output tokens."""

import json
from pathlib import Path
from statistics import mean

dest = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def layers(batch, confirm):
    suffix = "-confirm" if confirm else ""
    rows = [
        read(dest / f"glm53-gqa-port-d{doc}-b{batch}{suffix}.json")["results"]
        for doc in (1, 2)
    ]
    for row in rows:
        for variant in row.values():
            assert all(
                row["original"][field] == variant[field]
                for field in ("exact_leaves", "union_loaded_leaves", "open_regions")
            )
    return {name: mean(row[name]["median_us"] for row in rows) for name in rows[0]}


def serving(batch):
    baseline = read(dest.parent / f"serving-b{batch}-c16.json")["measurements"][
        "65536"
    ]["prompts"]
    values = {}
    for arm in ("c16", "ported"):
        r = read(dest / f"serving-b{batch}-{arm}.json")
        assert (
            r["status"] == "complete"
            and not r["random_weights"]
            and r["decode_tokens"] == 1026
        )
        m = r["measurements"]["65536"]
        assert len(baseline) == batch and baseline == m["prompts"]
        assert len(m["decode_timings_seconds"]) == 2
        hashes = [a["trace_token_sha256"] for a in baseline]
        assert (
            m["output_token_sha256"] == [hashes, hashes]
            and m["warmup_output_token_sha256"] == hashes
        )
        assert len(r["post_speed_workers"]) == 4
        for worker in r["post_speed_workers"]:
            assert len(worker["pools"]) == 11 and worker["daemon_resident_bytes"] > 0
            for p in worker["pools"]:
                assert (
                    p["decode_threshold"] == 16
                    and p["threshold_dispatch"]
                    and p["prefill_topk"] == 8
                )
                assert p["deferred_merge"] == (arm == "ported")
        for rep in m["measured_batch_timings"]:
            assert len(rep) == 1
            assert (
                rep[0]["all_requests_live_overlap_seconds"]
                == rep[0]["decode_window_seconds"]
            )
            assert (
                rep[0]["request_num_preemptions"] == [0] * batch
                and rep[0]["request_num_cached_tokens"] == [0] * batch
            )
        for rep in m["measured_decode_update_counters"]:
            assert len(rep) == 4
            for worker in rep:
                assert len(worker) == 11 and all(
                    v["catch_up_rows"] == 4 * batch for v in worker.values()
                )
        values[arm] = {
            "ms": m["decode_ms_per_batch_step"],
            "repeats": [1000 * v / 1025 for v in m["decode_timings_seconds"]],
        }
    return values


if __name__ == "__main__":
    summary = {
        f"b{b}": dict(
            sweep_us=layers(b, False),
            confirmed_us=layers(b, True),
            full_model=serving(b),
        )
        for b in (1, 8)
    }
    (dest / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
