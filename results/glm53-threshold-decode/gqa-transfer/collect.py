"""Aggregate checked replay and serving results for the MI325X transfer study."""
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
WAVES = ("worker", "reduce", "routing", "wide", "combine", "ordered", "stats", "donor", "ablate", "confirm", "profile")
WORK = ("open_regions", "exact_leaves", "union_loaded_leaves")


def read(path):
    return json.loads(path.read_text())


def effective_config(batch, config):
    options = dict(splits=256 if batch == 1 else 64, entry=32, score_tile=64, selector=64,
                   strip=512, merge=32, pv=0, cg=False, base2=False, precise=False, warps=4,
                   static=False, first=False, combine=batch == 1, contiguous=batch == 1, backend={},
                   guarded=False, union=False, ordered=False, half_masks=False, pair_stats=False,
                   cutoff=False, merge_cg=False, split_dim=False)
    options.update(config)
    if not options["combine"]:
        options["static"] = options["first"] = False
    return json.dumps(options, sort_keys=True)


def replay(batch):
    waves, completed, unsupported = {}, set(), set()
    for wave in WAVES:
        captures = [read(ROOT / f"{wave}-d{i}-b{batch}.json") for i in (1, 2)]
        assert all(c["batch"] == batch and c["heads"] == 16 for c in captures)
        result = {}
        for name, config in captures[0]["transfer_cases"].items():
            assert config == captures[1]["transfer_cases"][name]
            arms = [c["results"][name] for c in captures]
            key = effective_config(batch, config)
            if any("unsupported" in a for a in arms):
                unsupported.add(key)
                result[name] = dict(config=config, errors=[a.get("unsupported") for a in arms])
                continue
            completed.add(key)
            for capture, arm in zip(captures, arms):
                if not config.get("union"):
                    assert all(arm[t] == capture["results"]["control"][t] for t in WORK)
                assert len(arm["timings_us"]) == 11
            result[name] = dict(
                config=config,
                us=mean(a["median_us"] for a in arms),
                capture_us=[a["median_us"] for a in arms],
                relative_l2_max=max(a["relative_l2"] for a in arms),
                leaves_per_head=mean(a["exact_leaves"] for a in arms),
            )
            if wave == "profile":
                result[name]["stage_activity_counts"] = [
                    a.get("profile_samples", dict.fromkeys(a["kernel_medians_us"], 50)) for a in arms
                ]
                result[name]["stage_us"] = {
                    k: mean(a["kernel_medians_us"][k] for a in arms)
                    for k in arms[0]["kernel_medians_us"]
                }
        waves[wave] = result
    return dict(waves=waves, distinct_timed=len(completed),
                distinct_resource_limited=len(unsupported - completed))


def serving(batch, *, earlier_b8=False):
    expected = read(ROOT.parent / "gqa-ports" / f"serving-b{batch}-ported.json")["measurements"]["65536"]["prompts"]
    directory = ROOT / "b8-audit" if batch == 8 and not earlier_b8 else ROOT
    arms = {}
    for arm in ("ported", "transfer"):
        record = read(directory / f"serving-b{batch}-{arm}.json")
        assert record["status"] == "complete" and not record["random_weights"] and record["decode_tokens"] == 1026
        m = record["measurements"]["65536"]
        assert m["prompts"] == expected and len(expected) == batch
        hashes = [p["trace_token_sha256"] for p in expected]
        assert m["warmup_output_token_sha256"] == hashes and m["output_token_sha256"] == [hashes, hashes]
        assert len(record["post_speed_workers"]) == 4
        selected = read(ROOT / f"confirm-d1-b{batch}.json")["transfer_cases"]["tuned"]
        for worker in record["post_speed_workers"]:
            assert len(worker["pools"]) == 11 and worker["daemon_resident_bytes"] > 0
            for pool in worker["pools"]:
                assert pool["decode_threshold"] == 16 and pool["threshold_dispatch"] and pool["deferred_merge"]
                assert pool["prefill_topk"] == 8
                if arm == "ported":
                    assert pool["gqa_tuning"] == {}
                elif batch == 1:
                    assert pool["gqa_tuning"] == selected
                elif earlier_b8:
                    # The original worker diagnostic retained only the last Python dispatch.
                    assert pool["gqa_tuning"] == read(ROOT / "confirm-d1-b1.json")["transfer_cases"]["tuned"]
                else:
                    assert pool["gqa_tuning_by_batch"]["8"] == selected
                    assert pool["gqa_captured_batches"]["8"] == selected
        for round_timing in m["measured_batch_timings"]:
            rep, = round_timing
            assert rep["all_requests_live_overlap_seconds"] == rep["decode_window_seconds"]
            assert rep["request_num_preemptions"] == [0] * batch and rep["request_num_cached_tokens"] == [0] * batch
        for counters in m["measured_decode_update_counters"]:
            assert len(counters) == 4
            for layers in counters:
                assert len(layers) == 11 and all(layer["catch_up_rows"] == 4 * batch for layer in layers.values())
        arms[arm] = dict(ms=m["decode_ms_per_batch_step"], repeats_ms=[1000 * x / 1025 for x in m["decode_timings_seconds"]])
    return arms


if __name__ == "__main__":
    summary = {str(b): dict(**replay(b), serving=serving(b)) for b in (1, 8)}
    summary["8"]["earlier_serving_pair"] = serving(8, earlier_b8=True)
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
