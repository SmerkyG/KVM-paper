"""Render the Llama H100 operating-point comparisons from completed artifacts."""

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, median
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from benchmarks.h100_threshold_next_report import TASKS, score_files


BATCH_CONTEXTS = {1: (16384, 32768, 65536, 128000), 4: (8192, 16384, 32768, 65536)}
QUALITY_CONTEXTS = (32768, 65536)
SLOTS = {8192: 1448, 16384: 2048, 32768: 2896, 65536: 4096, 128000: 5724}
STYLES = {
    "Dense BF16": ("#242424", "o"),
    "Dense FP16": ("#777777", "s"),
    "LoD two-tier": ("#006aa6", "o"),
    "LoD adaptive three-tier": ("#c05b00", "D"),
    "RetroInfer": ("#00866a", "o"),
    "QUEST": ("#993d83", "s"),
    "ClusterKV": ("#87651c", "D"),
    "Multipole": ("#31583b", "^"),
    "Squeezed": ("#ab4355", "v"),
}
CONTROL = ("Dense BF16", "Dense FP16", "LoD two-tier", "LoD adaptive three-tier")
AUTHOR = {"Multipole": "multipole-512", "Squeezed": "gpu-squeezed-p80"}
NATIVE = {
    "Dense BF16": ("dense", "dense"),
    "Dense FP16": ("dense-fp16", "dense-fp16"),
    "LoD two-tier": ("two-tier", "two-tier"),
    "LoD adaptive three-tier": ("adaptive", "adaptive"),
    "RetroInfer": ("retro", "retro"),
    "QUEST": ("quest", "quest"),
    "ClusterKV": ("clusterkv-fixed", "clusterkv"),
}
MATCHED_POLICIES = {
    "Multipole": "512 remote exact tokens",
    "Squeezed": "independent fit with p80 cutoff",
    "RetroInfer": "1.8% retrieval and 23.2% estimation",
}
MODES = {"paper": "Published settings", "lod-matched": "LoD-matched settings"}
FONT_SCALE = 1.5


def read(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rel(path, directory):
    return os.path.relpath(path, directory)


def matched_budgets(directory):
    budgets = {}
    for length in SLOTS:
        path = directory / f"llama-lod-exact-audit-{length}.json"
        if not path.is_file():
            continue
        measured, = read(path)["records"]
        samples = measured["lod_exact_attention_audit"]
        assert [row["step"] for row in samples] == list(range(319, 1280, 64))
        values = [x for row in samples for x in row["tokens_per_query_head"]]
        assert len(values) == 16 * 30 * 32 and all(v > 0 for v in values)
        target = mean(values)
        # QUEST uses full archived pages and one current page. On average
        # 8.5 of the last page's 16 entries are populated during replay.
        quest = max(32, round((target + 7.5) / 16) * 16)
        # ClusterKV's configured budget includes the current token. Its
        # sink and generated tokens from the active window are additional.
        recent = mean(step % 320 for step in range(256, 1280))
        clusterkv = max(2, round(target - 16 - recent))
        budgets[length] = (target, quest, clusterkv)
    return budgets


def native_path(directory, paper_points, mode, batch, length, method):
    if method == "ClusterKV" and batch == 4:
        return None
    if method == "RetroInfer" and length == 8192:
        return None
    native, label = NATIVE[method]
    if mode == "lod-matched" and method not in CONTROL:
        # The published 64K RetroInfer timing already starts at 4,096.
        if method != "RetroInfer" or length != 65536:
            return directory / f"llama-lodmatch-b{batch}-{length}-{label}.json"
    if batch == 1:
        if length in QUALITY_CONTEXTS:
            return directory / paper_points[length, method]["timing_source"]
        if native == "retro" and length == 16384:
            native = "retro-deferred-index"
        return directory / f"llama-context-{length}-{native}.json"
    return directory.parent / "llama-final-checks" / f"llama{length // 1024}k-b4-final-predecode-{native}.json"


def author_path(directory, mode, batch, length):
    if mode == "paper" and batch == 1 and length in QUALITY_CONTEXTS:
        return directory / f"llama-paper-b1-{length}-paired.json"
    prefix = "paperscale" if mode == "paper" else "lodmatch"
    return directory / f"llama-{prefix}-b{batch}-{length}-paired.json"


def plot_rows(directory, paper_points, budgets):
    prompts = read(directory / "prompts-context-scaling.json")["contexts"]
    rows = []
    for mode in MODES:
        for batch, lengths in BATCH_CONTEXTS.items():
            for length in lengths:
                ids = prompts[str(length)]["prompts"][0]["prompt_token_ids"]
                raw_hash = hashlib.sha256(struct.pack(f"<{length}I", *ids)).hexdigest()
                json_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
                for method in STYLES:
                    timing = (author_path(directory, mode, batch, length) if method in AUTHOR else
                              native_path(directory, paper_points, mode, batch, length, method))
                    decode = prefill = ""
                    timing_source = prefill_source = ""
                    count = SLOTS[length] if method.startswith("LoD") else ""
                    budget = ""
                    if timing is not None and timing.is_file():
                        data = read(timing)
                        assert data["batch_size"] == batch and data["timed_decode_steps"] == 1024
                        if method in AUTHOR:
                            assert data["stage"] == "complete" and data["prefill"] == "dense causal"
                            assert data["matched_full_model_timing"] and data["warmup_steps"] == 256
                            assert data["timing_prefix_length"] == length
                            record, = data["records"]
                            assert record["input_token_sha256"] == raw_hash
                            replay_hash = hashlib.sha256(json.dumps(ids[-1280:], separators=(",", ":")).encode()).hexdigest()
                            assert record["teacher_forced_token_sha256"] == replay_hash
                            arm = record["decoders"][AUTHOR[method]]
                            reps = arm["repetitions"]
                            assert [row["warmup"] for row in reps] == [True, False, False]
                            decode = arm["median_decode_ms"]
                            for row in reps:
                                win, = row["windows"]
                                assert row["graph_check"]["max_abs"] == 0 and win["steps"] == 1024
                                if method == "Multipole":
                                    expected = SLOTS[length] if mode == "lod-matched" else ((length - 10) // 128 - 1) * 8
                                    assert row["initial_centroids_per_kv_head"] == expected
                                    assert win["cache_update_events"] == 8
                                    count, budget = expected, 512
                                else:
                                    ix = row["squeezed_index"]
                                    expected = SLOTS[length] if mode == "lod-matched" else (length - 100) // 20
                                    assert ix["clusters"] == expected and set(ix["threshold"]) == {"0.8"}
                                    assert not row["reused_initial_clusters"]
                                    count = expected
                            assert abs(decode - median(row["median_decode_ms"] for row in reps[1:])) < 1e-9
                        else:
                            assert data["prompt_token_sha256"] == json_hash
                            assert data["prompt_tokens"] == length and data["decode_warmup_steps"] == 256
                            assert data["replay_prompt_tokens"]
                            assert [r["warmup"] for r in data["records"]] == [True, False, False]
                            decode = data["median_decode_ms_per_batch_step"]
                            prefill = data["median_prefill_s"]
                            prefill_source = rel(timing, directory)
                            if method == "QUEST":
                                budget = data["quest_budget_tokens"]
                                assert budget == (2048 if mode == "paper" else budgets[length][1])
                            if method == "ClusterKV":
                                budget = data["clusterkv_settings"]["token_budget"]
                                assert budget == (1024 if mode == "paper" else budgets[length][2])
                                count = data["clusterkv_settings"]["prefill_centroids"]
                            if method == "RetroInfer" and mode == "lod-matched":
                                count = {16384: 2048, 32768: 2896, 65536: 4096, 128000: 5728}[length]
                        timing_source = rel(timing, directory)
                    if method in AUTHOR:
                        label = "paper" if mode == "paper" else "matched"
                        prep = directory / f"llama-figure-prefill-{label}-b{batch}-{length}-{method.lower()}.json"
                        if prep.is_file():
                            data = read(prep)
                            assert data["stage"] == "complete" and data["prefill"] == "dense causal"
                            assert data["batch_size"] == batch and data["prompt_tokens"] == length
                            assert data["input_token_sha256"] == raw_hash
                            assert [r["warmup"] for r in data["records"]] == [True, False, False]
                            expected = SLOTS[length] if mode == "lod-matched" else (
                                ((length - 10) // 128 - 1) * 8 if method == "Multipole" else (length - 100) // 20)
                            assert {r["initial_centroids"] for r in data["records"]} == {expected}
                            count = expected
                            prefill = data["median_prefill_s"]
                            prefill_source = rel(prep, directory)
                    if method == "RetroInfer" and length == 16384:
                        prefill = prefill_source = ""
                        label = "paper" if mode == "paper" else "matched"
                        prep = directory / f"llama-figure-prefill-{label}-b{batch}-{length}-retro.json"
                        if prep.is_file():
                            data = read(prep)
                            assert data["batch_size"] == batch and data["prompt_token_sha256"] == json_hash
                            assert data["decode_warmup_steps"] == 256 and data["timed_decode_steps"] == 1024
                            reps = data["records"]
                            assert [row["warmup"] for row in reps] == [True, False, False]
                            assert all(row["retro_eager_warmup_steps"] == 69 for row in reps)
                            assert all(row["deferred_retro_build_s"] > 0 for row in reps)
                            prefill = median(row["prefill_s"] + row["deferred_retro_build_s"] for row in reps[1:])
                            prefill_source = rel(prep, directory)
                    availability = ("batch-one adapter" if method == "ClusterKV" and batch == 4 else
                                    "dense fallback" if method == "RetroInfer" and length == 8192 else "pending")
                    rows.append(dict(settings=mode, batch=batch, context_tokens=length, method=method,
                                     initial_centroids_per_kv_head=count, configured_token_budget=budget,
                                     target_exact_entries_per_query_head=(
                                         budgets[length][0]
                                         if mode == "lod-matched" and method in ("QUEST", "ClusterKV")
                                         and length in budgets else ""),
                                     decode_ms_per_batch_step=decode, prefill_s_per_batch=prefill,
                                     decode_status="measured" if timing_source else availability,
                                     prefill_status="measured" if prefill_source else availability,
                                     decode_source=timing_source, prefill_source=prefill_source))
    return rows


def quality_for(directory, budgets, length):
    k = length // 1024
    root = directory.parent
    tokenizer = root / "ruler-32k-panel/llama-tokenizer.json"
    raw = gzip.decompress((root / f"quality-ruler-{k}k-frozen-13x32.json.gz").read_bytes())
    samples = {(r["task"], r["source_row"]): r for r in json.loads(raw)["samples"]}
    assert len(samples) == 416
    reference = {(r["task"], r["source_row"]): r
                 for path in directory.glob(f"llama-paper-{length}-ruler-*.json")
                 for r in read(path)["records"]}
    assert reference.keys() == samples.keys()
    output = {}
    sources = {}
    for method in (*AUTHOR, "RetroInfer", "QUEST", "ClusterKV"):
        suffix = ("" if method in AUTHOR else
                  "-" + {"RetroInfer": "retro", "QUEST": "quest", "ClusterKV": "clusterkv"}[method])
        paths = [directory / f"llama-lodmatch-{length}{suffix}-ruler-shard{i}.json" for i in range(4)]
        if not all(p.is_file() for p in paths):
            continue
        if method == "ClusterKV":
            check_path = directory / f"llama-lodmatch-{length}-clusterkv-graph-check.json"
            if not check_path.is_file():
                continue
            check_data = read(check_path)
            assert check_data["clusterkv_settings"]["token_budget"] == budgets[length][2]
            check_record, = check_data["records"]
            checks = check_record["graph_same_cache_checks"]
            assert checks and all(x["max_abs"] == 0 and x["graph_argmax"] == x["eager_argmax"]
                                  for x in checks.values())
        scores = defaultdict(list)
        seen = set()
        native = None
        if method not in AUTHOR:
            _, native = score_files(paths, tokenizer)
            assert native.keys() == samples.keys()
        for path in paths:
            data = read(path)
            assert data["stage"] == "complete"
            assert data["model_name"] == "meta-llama/Llama-3.1-8B-Instruct"
            if method in AUTHOR:
                assert data["cohort_sha256"] == hashlib.sha256(raw).hexdigest()
                assert data["prefill"] == "dense causal" and data["stop_at_eos"]
                assert data["multipole_fixed_centroids_per_kv_head"] == data["squeezed_fixed_centroids_per_kv_head"] == SLOTS[length]
            elif method == "RetroInfer":
                assert data["retroinfer_settings"]["prefill_centroids_override"] == SLOTS[length]
            elif method == "QUEST":
                assert data["quest_settings"]["budget"] == budgets[length][1]
            elif method == "ClusterKV":
                assert data["clusterkv_settings"]["budget"] == budgets[length][2]
            for row in data["records"]:
                key = row["task"], row["source_row"]
                assert key not in seen
                seen.add(key)
                assert row["prompt_sha256"] == samples[key]["prompt_sha256"]
                assert row["answers"] == samples[key]["answers"]
                assert row["input_token_sha256"] == reference[key]["input_token_sha256"]
                if method in AUTHOR:
                    arm = row["decoders"][AUTHOR[method]]
                    assert arm["graph_check"]["max_abs"] == 0
                    assert row["prefill_token_id"] == reference[key]["prefill_token_id"]
                    if method == "Multipole":
                        assert arm["initial_centroids_per_kv_head"] == SLOTS[length]
                        assert max(arm["last_layer_selected_prefix_counts"]) <= 512
                    else:
                        assert arm["squeezed_index"]["clusters"] == SLOTS[length]
                        assert set(arm["squeezed_index"]["threshold"]) == {"0.8"}
                    value = arm["score"]
                else:
                    if method == "RetroInfer":
                        state = row["retroinfer_state"]
                        alignment = math.lcm(8, state["prefill_segments"])
                        assert state["prefill_centroids"] % alignment == 0
                        assert abs(state["prefill_centroids"] - SLOTS[length]) <= alignment / 2
                    value = native[key]["score"]
                assert 0 <= value <= 1
                scores[key[0]].append(value * 100)
        assert seen == samples.keys()
        assert set(scores) == set(TASKS) and {len(values) for values in scores.values()} == {32}
        output[method] = {task: mean(scores[task]) for task in TASKS}
        sources[method] = [path.name for path in paths]
    return output, sources


def quality_rows(directory, paper_points, rows, budgets):
    selected = {(r["context_tokens"], r["method"]): r for r in rows
                if r["settings"] == "lod-matched" and r["batch"] == 1}
    points = []
    lines = ["# LoD-matched RULER", "",
             "Llama 3.1 8B Instruct on H100, one request at a time, with 32 frozen examples per task. "
             "The 13-task mean weights tasks equally. Timing and quality use the same adjusted settings."]
    for length in QUALITY_CONTEXTS:
        qualities, sources = quality_for(directory, budgets, length)
        methods = list(qualities)
        lines.extend(("", f"## {length // 1024}K RULER", ""))
        if methods:
            lines.append("| Task | " + " | ".join(methods) + " |")
            lines.append("|---|" + "---:|" * len(methods))
            for task in TASKS:
                lines.append(f"| {task} | " + " | ".join(f"{qualities[m][task]:.2f}%" for m in methods) + " |")
            averages = {m: mean(qualities[m].values()) for m in methods}
            lines.append("| **13-task mean** | " + " | ".join(f"**{averages[m]:.2f}%**" for m in methods) + " |")
            lines.append("")
            for method in methods:
                links = []
                for i, name in enumerate(sources[method]):
                    command, = (directory / "provenance").glob(f"*-{Path(name).stem}/run.json")
                    links.append(f"[Shard {i}]({name}) · [Command]({command.relative_to(directory)})")
                lines.append(f"- {method}. " + " · ".join(links))
        else:
            averages = {}
        pending = [m for m in STYLES if m not in CONTROL and m not in methods]
        if pending:
            lines.extend(("", "Pending quality or validation runs for " + ", ".join(pending) + "."))
        for method in STYLES:
            point = dict(paper_points[length, method])
            row = selected[length, method]
            if method not in CONTROL:
                if method not in averages or not row["decode_source"]:
                    continue
                point.update(
                    initial_centroid_slots_per_kv_head=row["initial_centroids_per_kv_head"],
                    attention_policy=(
                        f'{row["initial_centroids_per_kv_head"]:,} initial slots per KV head, '
                        f'{MATCHED_POLICIES[method]}'
                        if method in MATCHED_POLICIES else
                        f'{row["configured_token_budget"]:,} configured exact-token budget'),
                    decode_ms_per_token=row["decode_ms_per_batch_step"],
                    ruler_13_task_mean_pct=averages[method],
                    timing_source=row["decode_source"], quality_source="RULER_LOD_MATCHED.md",
                    quality_section=f"{length // 1024}K RULER",
                )
            points.append(point)
    (directory / "RULER_LOD_MATCHED.md").write_text("\n".join(lines) + "\n")
    write_csv(directory / "decode-speed-quality-lod-matched.csv", points)
    return points


def legend(fig, methods):
    handles = [Line2D([], [], label=m, color=STYLES[m][0], marker=STYLES[m][1], linewidth=1.5)
               for m in STYLES if m in methods]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=3, frameon=False, borderaxespad=0, fontsize=9 * FONT_SCALE)


def save(fig, directory, name, preview):
    fig.savefig(directory / (name + ".pdf"), bbox_inches="tight")
    if preview:
        fig.savefig(preview / (name + ".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def curves(rows, directory, preview):
    decodes = [float(r["decode_ms_per_batch_step"]) for r in rows if r["decode_source"]]
    ylim = (6.0, max(decodes) * 1.08)
    for mode in MODES:
        fig, axs = plt.subplots(1, 2, figsize=(10.1, 4.8), sharey=True)
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.34, top=0.82, wspace=0.19)
        methods = set()
        for ax, (batch, lengths) in zip(axs, BATCH_CONTEXTS.items(), strict=True):
            for method, (color, marker) in STYLES.items():
                sub = {r["context_tokens"]: r["decode_ms_per_batch_step"] for r in rows
                       if r["settings"] == mode and r["batch"] == batch and r["method"] == method
                       and r["decode_source"]}
                if sub:
                    methods.add(method)
                    ax.plot(range(len(lengths)), [sub.get(n, float("nan")) for n in lengths],
                            color=color, marker=marker, linewidth=1.5)
            ax.set(xticks=range(len(lengths)), xticklabels=[f"{round(n/1024)}K" if n < 128000 else "128,000" for n in lengths],
                   title=f"Batch {batch}", xlabel="Context tokens", ylim=ylim)
            ax.grid(alpha=0.3)
        axs[0].set_ylabel("Decode latency (ms/batch step)")
        legend(fig, methods)
        save(fig, directory, f"decode-latency-{mode}", preview)

    prefill = [float(r["prefill_s_per_batch"]) for r in rows if r["prefill_source"]]
    fig, axs = plt.subplots(2, 2, figsize=(10.1, 6.8), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.245, top=0.85, hspace=0.46, wspace=0.19)
    methods = set()
    for row_axes, (mode, title) in zip(axs, MODES.items(), strict=True):
        for ax, (batch, lengths) in zip(row_axes, BATCH_CONTEXTS.items(), strict=True):
            for method, (color, marker) in STYLES.items():
                sub = {r["context_tokens"]: r["prefill_s_per_batch"] for r in rows
                       if r["settings"] == mode and r["batch"] == batch and r["method"] == method
                       and r["prefill_source"]}
                if sub:
                    methods.add(method)
                    ax.plot(range(len(lengths)), [sub.get(n, float("nan")) for n in lengths],
                            color=color, marker=marker, linewidth=1.5)
            ax.set(xticks=range(len(lengths)), xticklabels=[f"{round(n/1024)}K" if n < 128000 else "128,000" for n in lengths],
                   title=f"{title} · batch {batch}", yscale="log", ylim=(min(prefill)*0.8, max(prefill)*1.25))
            ax.grid(alpha=0.3, which="both")
        row_axes[0].set_ylabel("Prefill latency (s/batch)")
    for ax in axs[-1]:
        ax.set_xlabel("Context tokens")
    legend(fig, methods)
    save(fig, directory, "prefill-latency", preview)


def quality_plots(points, paper, directory, preview):
    all_points = [*points, *paper]
    xmin = min(float(p["decode_ms_per_token"]) for p in all_points)
    xmax = max(float(p["decode_ms_per_token"]) for p in all_points)
    scores = sorted({float(p["ruler_13_task_mean_pct"]) for p in all_points})
    gaps = list(zip(scores, scores[1:]))
    lower, upper = max(gaps, key=lambda span: span[1] - span[0])
    broken = upper - lower >= 7
    ranges = ([(upper - 1.4, scores[-1] + 1), (scores[0] - 2.4, lower + 1.7)]
              if broken else [(scores[0] - 1, scores[-1] + 1)])
    for mode, data in (("paper", paper), ("lod-matched", points)):
        fig, axs = plt.subplots(len(ranges), 2, figsize=(10.1, 5.15), sharex="col", squeeze=False,
                                gridspec_kw={"height_ratios": [4, 1]} if broken else None)
        partial = len(data) < len(paper)
        fig.subplots_adjust(top=0.79, bottom=0.36 if partial else 0.325, hspace=0.10, wspace=0.15)
        for col, length in enumerate(QUALITY_CONTEXTS):
            axs[0, col].set_title(f"{length//1024}K context", loc="left")
            for i, (lo, hi) in enumerate(ranges):
                ax = axs[i, col]
                ax.set(xlim=(xmin - 0.7, xmax + 0.8), ylim=(lo, hi))
                if col:
                    ax.tick_params(labelleft=False)
                for point in data:
                    score = float(point["ruler_13_task_mean_pct"])
                    if int(point["context_tokens"]) == length and lo <= score <= hi:
                        method = point["method"]
                        color, marker = STYLES[method]
                        ax.scatter(float(point["decode_ms_per_token"]), score, marker=marker,
                                   edgecolor=color, facecolor="none" if method == "Dense FP16" else color,
                                   s=38, zorder=3)
                ax.grid(alpha=0.28)
            if broken:
                top, bottom = axs[:, col]
                top.spines["bottom"].set_visible(False)
                top.tick_params(axis="x", bottom=False, labelbottom=False)
                dash = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=8, color="#343434",
                            linestyle="none", clip_on=False)
                top.plot([0], [0], transform=top.transAxes, **dash)
                bottom.plot([0], [1], transform=bottom.transAxes, **dash)
            axs[-1, col].set_xlabel("Decode latency (ms/token)")
        fig.supylabel("RULER 13-task mean (%)", x=0.037, fontsize=11 * FONT_SCALE)
        legend(fig, {p["method"] for p in data})
        if partial:
            fig.text(0.5, 0.18, "Points appear after both timing and quality finish.",
                     ha="center", fontsize=8 * FONT_SCALE)
        name = "decode-speed-quality" if mode == "paper" else "decode-speed-quality-lod-matched"
        save(fig, directory, name, preview)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, default=Path("results/h100-baselines/llama-context-scaling"))
    p.add_argument("--preview", type=Path)
    a = p.parse_args()
    if a.preview:
        a.preview.mkdir(parents=True, exist_ok=True)
    paper = read_csv(a.directory / "decode-speed-quality.csv")
    paper_points = {(int(r["context_tokens"]), r["method"]): r for r in paper}
    assert len(paper_points) == len(QUALITY_CONTEXTS) * len(STYLES)
    budgets = matched_budgets(a.directory)
    rows = plot_rows(a.directory, paper_points, budgets)
    write_csv(a.directory / "comparison-timings.csv", rows)
    points = quality_rows(a.directory, paper_points, rows, budgets)
    with plt.rc_context({"font.size": FONT_SCALE * plt.rcParamsDefault["font.size"],
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}):
        curves(rows, a.directory, a.preview)
        quality_plots(points, paper, a.directory, a.preview)
    for mode in MODES:
        r = [row for row in rows if row["settings"] == mode and
             not (row["method"] == "ClusterKV" and row["batch"] == 4) and
             not (row["method"] == "RetroInfer" and row["context_tokens"] == 8192)]
        print(mode, "decode", sum(bool(row["decode_source"]) for row in r), "prefill",
              sum(bool(row["prefill_source"]) for row in r), "of", len(r))
    print("LoD-matched speed-quality points", len(points))


if __name__ == "__main__":
    main()
