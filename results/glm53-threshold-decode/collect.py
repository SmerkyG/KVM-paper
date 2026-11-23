"""Validate matched full and layer runs and summarize the measured decode windows."""

import json
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parent


def read(name):
    return json.loads((root / name).read_text())


def layer_summary(batch):
    rows = [read(f"confirm-d{doc}-b{batch}.json") for doc in (1, 2)]
    result = {}
    for arm in ("top8", "c8", "c16", "c32"):
        arms = [row["results"][arm] for row in rows]
        result[arm] = {
            "median_us_by_capture": [a["median_us"] for a in arms],
            "mean_layer_us": mean(a["median_us"] for a in arms),
            "mean_opened_leaves": mean(a["exact_leaves"] for a in arms),
            "mean_shared_leaves": mean(a["union_loaded_leaves"] for a in arms),
        }
    return result


def serving(batch):
    ref = None
    results = {}
    for name in ("top8", "c16", "native"):
        file = f"serving-b{batch}-{name}.json"
        raw = read(file)
        assert raw["status"] == "complete" and raw["random_weights"] is False
        assert len(raw["post_speed_workers"]) == 4
        for worker in raw["post_speed_workers"]:
            assert len(worker["mla"]) == 11
            assert worker["daemon_resident_bytes"] > 0
            if name == "native":
                assert not worker["pools"]
                assert all(
                    layer["sparse"] and layer["backend"] == "ROCMAiterMLASparseImpl"
                    for layer in worker["mla"]
                )
                continue
            assert len(worker["pools"]) == 11
            for layer in worker["pools"]:
                assert layer["threshold_dispatch"] == (name == "c16")
                assert layer["decode_threshold"] == (16 if name == "c16" else None)
                assert layer["prefill_topk"] == 8
        m = raw["measurements"]["65536"]
        assert len(m["prompts"]) == batch
        if ref is None:
            ref = m["prompts"]
        assert m["prompts"] == ref
        prior_file = f"../glm53-flash-full-model/packed-query-fix-two-tier-tp4-b{batch}-65536.json"
        assert ref == read(prior_file)["measurements"]["65536"]["prompts"]
        assert len(m["measured_decode_update_counters"]) == 2
        for rep in m["measured_decode_update_counters"]:
            assert len(rep) == 4
            for rank in rep:
                if name == "native":
                    assert not rank
                    continue
                assert len(rank) == 11 and all(
                    v["catch_up_rows"] == 4 * batch for v in rank.values()
                )
        assert len(m["decode_timings_seconds"]) == len(m["measured_batch_timings"]) == 2
        expected_trace = [row["trace_token_sha256"] for row in ref]
        assert m["output_token_sha256"] == [expected_trace, expected_trace]
        assert m["warmup_output_token_sha256"] == expected_trace
        for rep in m["measured_batch_timings"]:
            assert len(rep) == 1
            assert rep[0]["request_num_preemptions"] == [0] * batch
            assert rep[0]["request_num_cached_tokens"] == [0] * batch
            assert (
                rep[0]["all_requests_live_overlap_seconds"]
                == rep[0]["decode_window_seconds"]
            )
        results[name] = {
            "ms_per_step": m["decode_ms_per_batch_step"],
            "repeat_ms_per_step": [
                1000 * v / 1025 for v in m["decode_timings_seconds"]
            ],
            "prefill_s": m["prefill_seconds"],
            "source": file,
        }
    results["percent_lower"] = 100 * (
        1 - results["c16"]["ms_per_step"] / results["top8"]["ms_per_step"]
    )
    return results


if __name__ == "__main__":
    result = {
        f"b{b}": {"attention": layer_summary(b), "full_model": serving(b)}
        for b in (1, 8)
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
