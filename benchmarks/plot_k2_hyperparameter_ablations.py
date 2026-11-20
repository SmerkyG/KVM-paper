#!/usr/bin/env python3
"""Render the completed K2 Horizon ablations from their audited result JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator


CONTROL = "#495465"
DENSE = "#6B3E76"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#008564"
GRID = "#E1E6ED"
INK = "#182638"
META = {"Creator": "Matplotlib", "CreationDate": None, "ModDate": None}
BASE = Path(__file__).resolve().parents[1] / "results/amd-threshold-ports"
REPORTS = {
    "dense": "quality32-final-v1/REPORT.json",
    "threshold": "k2-threshold-ablation-v1/REPORT.json",
    "c4": "k2-threshold-c4-v1/REPORT.json",
    "c64": "k2-threshold-c64-v1/REPORT.json",
    "topk": "k2-fixed-region-budget-v1/REPORT.json",
    "topk_speed": "k2-fixed-region-budget-v1/timing/REPORT.json",
    "geometry": "k2-geometry-ablation-v2/REPORT.json",
    "b128": "k2-bswa128-ablation-v1/REPORT.json",
    "b512": "k2-bswa512-ablation-v1/REPORT.json",
    "prefill": "k2-prefill-block-ablation-v2/REPORT.json",
    "pages": "k2-page-budget-ablation-v1/REPORT.json",
}
FOLLOWUPS = {
    "threshold_speed": "k2-threshold-ablation-v1/timing/REPORT.json",
    "c2": "k2-threshold-c2-v1/REPORT.json",
    "c2_speed": "k2-threshold-c2-v1/timing/REPORT.json",
    "prefill_small": "k2-prefill-block-small-v1/REPORT.json",
    "prefill_short": "k2-prefill-block-short-v1/REPORT.json",
    "prefill_512_fixed": "k2-prefill-block-512-fixed-v1/REPORT.json",
    "prefill_256": "k2-prefill-block-256-v1/REPORT.json",
    "centroid_dense_comparison": "k2-geometry-ablation-v2/DENSE_COMPARISON.json",
    "pages_fixed1": "k2-page-budget-fixed1-v1/REPORT.json",
    "pages_full": "k2-page-budget-ruler-v1/REPORT.json",
    "dense_comparison": "k2-pareto-dense-v1/REPORT.json",
}
MULTIPLIERS = (4, 8, 16, 32, 64)
DENSE_NOTE = "Dense full attention uses the same RULER prompts."

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titleweight": "bold",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.labelcolor": INK,
    "axes.edgecolor": CONTROL,
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "mathtext.fontset": "dejavusans",
    "text.color": INK,
})


def canvas(note, *, cols=2, rows=1, width=8, height=3.7, **margins):
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False)
    fig.subplots_adjust(**{
        "left": 0.10, "right": 0.96, "bottom": 0.28, "top": 0.85,
        "wspace": 0.40, "hspace": 0.65, **margins,
    })
    fig.text(0.055, 0.042, note, fontsize=7.5, color=CONTROL, va="bottom", linespacing=1.45)
    for ax in axes.flat:
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.yaxis.set_major_locator(MaxNLocator(5))
    return fig, list(axes.flat)


def xlabels(ax, labels, title):
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-0.48, len(labels) - 0.52)
    ax.set_xlabel(title)


def zero(ax, *, vertical=False, label=None):
    line = ax.axvline if vertical else ax.axhline
    line(0, color=CONTROL, linewidth=0.85, linestyle=(0, (3, 2)), zorder=1, label=label)


def dense_reference(ax, score):
    ax.axhline(score, color=DENSE, linewidth=1.15, linestyle=(0, (1, 2.2)), zorder=1.5,
               label="Dense full attention")


def figure_legend(fig, *axes):
    entries = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        entries.update(zip(labels, handles, strict=True))
    fig.legend(list(entries.values()), list(entries), loc="upper right",
               bbox_to_anchor=(0.985, 0.98), ncol=len(entries))


def interval(ax, x, bounds, color, *, cap=0.045):
    if bounds is not None:
        lo, hi = bounds
        ax.vlines(x, lo, hi, color=color, linewidth=1.05, zorder=2)
        ax.hlines((lo, hi), x - cap, x + cap, color=color, linewidth=1.05, zorder=2)


def series(ax, xs, ys, *, intervals=None, color=BLUE, marker="o", label=None, linestyle="-"):
    if intervals is not None:
        for x, bounds in zip(xs, intervals, strict=True):
            interval(ax, x, bounds, color)
    ax.plot(xs, ys, color=color, marker=marker, markersize=4.8, linewidth=1.4,
            linestyle=linestyle, label=label, zorder=3)


def quality_axis(ax, title="RULER quality", ylabel="RULER change (pp)"):
    ax.set_title(title, loc="left", pad=8)
    ax.set_ylabel(ylabel)
    zero(ax)


def evidence(data, variant, baseline):
    return (data["means"][variant] - data["means"][baseline],
            data["paired_bootstrap_95_intervals"].get(variant))


def shift_bounds(bounds, shift):
    return None if bounds is None else tuple(b + shift for b in bounds)


def threshold_key(phase, c):
    return f"p{c}-d16" if phase == "prefill" else f"p16-d{c}"


def threshold_figure(reports):
    main = reports["threshold"]
    baseline = main["means"]["p16-d16"]
    multipliers = (2, *MULTIPLIERS) if "c2" in reports else MULTIPLIERS
    for name in (f"c{c}" for c in multipliers if f"c{c}" in reports):
        if not math.isclose(baseline, reports[name]["means"]["p16-d16"], abs_tol=1e-10):
            raise ValueError("Threshold sweep and supplements have different controls")
    dense_note = DENSE_NOTE
    if "c2" in reports:
        dense_note += " The c=2 prefill arm used a smaller internal query tile."
    fig, (ax,) = canvas(
        f"Each curve varies one phase while the other stays at c=16. The shared c=16 control scored {baseline:.2f}%."
        f"\nRULER macro-average over 13 tasks. Bars are paired 95% bootstrap intervals.\n{dense_note}",
        cols=1, width=8.6,
    )
    for phase, color, marker, offset in (("prefill", BLUE, "o", -0.08), ("decode", ORANGE, "s", 0.08)):
        values, bounds = [], []
        for c in multipliers:
            if c == 16:
                values.append(0)
                bounds.append(None)
                continue
            data = reports.get(f"c{c}", main)
            key = threshold_key(phase, c)
            row = data["comparisons"][key]
            values.append(row["delta_points"])
            bounds.append(row["paired_bootstrap_95_interval_points"])
        series(ax, [i + offset for i in range(len(multipliers))], values,
               intervals=bounds, color=color, marker=marker, label=f"Vary {phase}")
    quality_axis(ax, "Sensitivity to the region threshold")
    dense_reference(ax, reports["dense_score"] - baseline)
    xlabels(ax, multipliers, "Threshold multiplier c")
    figure_legend(fig, ax)
    return fig


def topk_figure(reports):
    quality, timing = reports["topk"], reports["topk_speed"]
    control = quality["means"]["threshold"]
    if not math.isclose(control, timing["quality_control"], abs_tol=1e-10):
        raise ValueError("Fixed-budget quality and timing reports have different controls")
    fig, axes = canvas(
        "Top-K uses the same normalized region scores as the c=16 control. Only the plotted phase changes."
        f"\nTime deltas use same-GPU medians. Quality bars are paired 95% bootstrap intervals.\n{DENSE_NOTE}",
        left=0.12, wspace=0.38,
    )
    for ax, phase, color, marker, cap in zip(
        axes, ("prefill", "decode"), (BLUE, ORANGE), ("o", "s"), (0.20, 0.045), strict=True
    ):
        quality_axis(ax, f"{phase.capitalize()} budget")
        dense_reference(ax, reports["dense_score"] - control)
        zero(ax, vertical=True)
        ax.scatter(0, 0, marker="D", s=27, color=CONTROL, zorder=4)
        ax.annotate("c=16", (0, 0), xytext=(6, 4), textcoords="offset points", fontsize=8)
        for k in (4, 8, 16):
            key = f"{phase}-top{k}"
            score = quality["comparisons"][key]
            time = timing["results"][key]
            if not math.isclose(time["ruler_mean"], quality["means"][key], abs_tol=1e-10):
                raise ValueError(f"Quality mismatch for {key}")
            x = time[f"{phase}_change_percent"]
            y = score["delta_points"]
            interval(ax, x, score["paired_bootstrap_95_interval_points"], color, cap=cap)
            ax.scatter(x, y, color=color, marker=marker, s=27, zorder=4)
            offset = (-6, -10) if phase == "decode" and k == 8 else (6, 4)
            ax.annotate(f"Top-{k}", (x, y), xytext=offset, textcoords="offset points",
                        fontsize=8, ha="right" if offset[0] < 0 else "left")
        ax.set_xlabel(f"{phase.capitalize()} latency change (%)")
        ax.set_xlim((-4.0, 18.0) if phase == "prefill" else (-0.45, 4.5))
        ax.set_ylim(-5.0, 0.85)
        if phase == "decode":
            ax.set_ylabel("")
    figure_legend(fig, axes[0])
    return fig


def centroid_figure(data, dense, dense_comparison=None):
    keys = ["m8-w512", "m16-w512", "m32-w512"]
    baseline = keys[1]
    dense_note = DENSE_NOTE
    if dense_comparison is not None:
        if dense_comparison["dense"] != dense or dense_comparison["lod"] != data["means"][keys[2]]:
            raise ValueError("Mismatched centroid dense comparison")
        lo, hi = dense_comparison["paired_bootstrap_95_interval_points"]
        delta = dense_comparison["delta_points"]
        dense_note += f" 32√N vs. dense is {delta:+.2f} pp (paired 95% CI {lo:+.2f} to {hi:+.2f})."
    fig, axes = canvas(
        "The window and c=16 routing are fixed. Bars show paired 95% bootstrap intervals versus 16√N LoD."
        f"\nLatency panels show the single warmed timings collected on separate GPUs.\n{dense_note}",
        cols=3, width=9.6, left=0.085, wspace=0.52, bottom=0.29,
    )
    xs = range(len(keys))
    labels = [rf"${m}\sqrt{{N}}$" for m in (8, 16, 32)]
    y, bounds = zip(*(evidence(data, variant, baseline) for variant in keys), strict=True)
    quality_axis(axes[0])
    dense_reference(axes[0], dense - data["means"][baseline])
    series(axes[0], xs, y, intervals=bounds)
    for ax, key, title, units in (
        (axes[1], "prefill_seconds", "Prefill time", "Time to first token (s)"),
        (axes[2], "decode_ms_per_batch_step", "Decode time", "Latency (ms/token)"),
    ):
        ax.set_title(title, loc="left", pad=8)
        ax.set_ylabel(units)
        series(ax, xs, [data["serving_checks"][v][key] for v in keys], color=ORANGE, marker="s")
    for ax in axes:
        xlabels(ax, labels, "Centroid schedule")
    figure_legend(fig, axes[0])
    return fig


def window_figure(data, dense):
    keys = ["m16-w256", "m16-w512", "m16-w1024"]
    fig, (a, b) = canvas(
        "The decode block and cache catch-up interval stay at 256 tokens. Quality bars are paired 95% bootstrap intervals."
        f"\nDecode latency comes from one warmed timing per setting on separate GPUs.\n{DENSE_NOTE}",
        left=0.11,
    )
    xs = range(len(keys))
    y, bounds = zip(*(evidence(data, variant, keys[1]) for variant in keys), strict=True)
    quality_axis(a)
    dense_reference(a, dense - data["means"][keys[1]])
    series(a, xs, y, intervals=bounds)
    b.set_title("Decode latency", loc="left", pad=8)
    b.set_ylabel("Latency (ms/token)")
    series(b, xs, [data["serving_checks"][v]["decode_ms_per_batch_step"] for v in keys],
           color=ORANGE, marker="s")
    b.set_ylim(38.7, 39.3)
    for ax in (a, b):
        xlabels(ax, ["256", "512", "1,024"], "Configured decode window (tokens)")
    figure_legend(fig, a)
    return fig


def bswa_figure(reports):
    fig, (a, b) = canvas(
        "Each series uses its own 256-token block and 512-token window control."
        "\nQuality bars are paired 95% bootstrap intervals shifted onto the absolute score axis."
        f"\nSame-GPU decode timings include cache catch-up. {DENSE_NOTE}",
        left=0.11, bottom=0.28, width=9.2, height=3.8,
    )
    for block, color, marker, offset in ((128, BLUE, "o", -0.07), (512, ORANGE, "s", 0.07)):
        report = reports[f"b{block}"]
        variants = [f"b{block}-w{block*n}" for n in range(1, 5)]
        xs = [n + offset for n in range(4)]
        baseline = report["means"]["b256-w512"]
        y = [report["means"][v] for v in variants]
        bounds = [shift_bounds(report["paired_bootstrap_95_intervals"][v], baseline) for v in variants]
        series(a, xs, y, intervals=bounds, color=color, marker=marker)
        change = [bswa_latency_change(report, v) for v in variants]
        series(b, xs, change, color=color, marker=marker, label=f"{block}-token blocks")
    a.set_title("RULER quality", loc="left", pad=8)
    a.set_ylabel("RULER macro-average (%)")
    dense_reference(a, reports["dense_score"])
    b.set_title("Decode latency", loc="left", pad=8)
    b.set_ylabel("Latency change (%)")
    zero(b)
    b.set_ylim(-1, 1)
    for ax in (a, b):
        xlabels(ax, [0, 1, 2, 3], "Complete preceding blocks kept exact")
    figure_legend(fig, b, a)
    return fig


def bswa_latency_change(report, variant):
    metric = "decode_ms_per_batch_step"
    return 100 * (report["serving_checks"][variant][metric]
                  / report["paired_speed_controls"][variant][metric] - 1)


def quality_speed_points(ax, points, dense, phase, color, marker):
    # At tied latencies, consider the best score first.
    measurements = [(x, y) for x, y, *_ in points] + [dense]
    latencies = [x for x, _ in measurements]
    cap = (max(latencies) - min(latencies)) * 0.009
    frontier = []
    best = -math.inf
    for x, y in sorted(measurements, key=lambda point: (point[0], -point[1])):
        if y > best:
            frontier.append((x, y))
            best = y
    xs, ys = zip(*frontier, strict=True)
    ax.plot(xs, ys, color=INK, linewidth=1.25, drawstyle="steps-post", zorder=2.5,
            marker="o", markersize=8.2, markerfacecolor="none", label="Measured Pareto frontier")
    for x, y, bounds, label, offset in points:
        interval(ax, x, bounds, color, cap=cap)
        control = bounds is None
        ax.scatter(x, y, marker="D" if control else marker, s=27,
                   color=CONTROL if control else color, zorder=4)
        ax.annotate(label, (x, y), xytext=offset, textcoords="offset points",
                    ha="right" if offset[0] < 0 else "left", fontsize=8)
    ax.scatter(*dense, marker="X", s=39, color=DENSE, label="Dense full attention", zorder=4)
    ax.set_xlabel("Time to first token (s)" if phase == "prefill" else "Decode latency (ms/token)")
    ax.set_ylabel("RULER macro-average (%)")
    ax.margins(x=0.17, y=0.15)


def threshold_speed_figure(reports):
    control = reports["threshold"]["means"]["p16-d16"]
    dense = reports["dense_score"]
    dense_speed = reports["dense_comparison"]["speed"]["threshold"]
    timing = reports["threshold_speed"]
    if not math.isclose(control, timing["quality_control"], abs_tol=1e-10):
        raise ValueError("Threshold quality and timing reports have different controls")
    supplement = reports.get("c2_speed") if "c2" in reports else None
    multipliers = (2, *MULTIPLIERS) if supplement else MULTIPLIERS
    timing_note = "The c=16 point uses the median of the paired control timings. Steps trace the measured frontier."
    if supplement:
        timing_note = "The c=2 prefill arm uses a smaller internal tile. The c=16 point uses the original tile."
    fig, axes = canvas(
        "Absolute latency on the same 64K ProLong prompt. Decode includes 1,024 steps and cache updates."
        f"\n{timing_note}"
        f"\nDense scored {dense:.2f}% on the same RULER prompts. Bars show paired 95% intervals for quality changes.",
        left=0.11, right=0.95, width=8.8, height=4.1, wspace=0.43, bottom=0.27,
    )
    for ax, phase, color, marker in zip(
        axes, ("prefill", "decode"), (BLUE, ORANGE), ("o", "s"), strict=True
    ):
        metric = "prefill_s" if phase == "prefill" else "decode_ms"
        control_latency = median(timing["results"][threshold_key(phase, c)]["control"][metric]
                                 for c in MULTIPLIERS if c != 16)
        points = []
        for c in multipliers:
            if c == 16:
                points.append((control_latency, control, None, "c=16", (5, 6)))
                continue
            key = threshold_key(phase, c)
            report = reports.get(f"c{c}", reports["threshold"])
            score = report["means"][key]
            row = (supplement if c == 2 else timing)["results"][key]
            if not math.isclose(score, row["ruler_mean"], abs_tol=1e-10):
                raise ValueError(f"Quality mismatch for {key}")
            bounds = shift_bounds(report["comparisons"][key]["paired_bootstrap_95_interval_points"], control)
            offset = {8: (6, -10), 32: (-6, 7), 64: (-6, 5)}.get(c, (6, 4))
            if c == 2 and phase == "decode":
                offset = (6, -11)
            points.append((row["candidate"][metric], score, bounds, f"c={c}", offset))
        quality_speed_points(ax, points, (dense_speed[metric], dense), phase, color, marker)
        ax.set_title(f"Vary {phase} threshold", loc="left", pad=8)
    ylim = (min(ax.get_ylim()[0] for ax in axes), max(ax.get_ylim()[1] for ax in axes))
    for ax in axes:
        ax.set_ylim(ylim)
    axes[1].set_ylabel("")
    figure_legend(fig, axes[0])
    return fig


def bswa_speed_figure(reports):
    dense = reports["dense_score"]
    dense_speed = reports["dense_comparison"]["speed"]["bswa"]["decode_ms"]
    fig, axes = canvas(
        "Absolute decode latency on the same 64K ProLong prompt, including cache catch-up."
        "\nLabels count complete preceding blocks kept exact. The 256-token control is the median of the paired timings."
        f"\nDense scored {dense:.2f}% on the same RULER prompts. Bars show paired 95% intervals for quality changes.",
        left=0.10, right=0.95, width=9.6, height=4.1, wspace=0.42, bottom=0.27,
    )
    for ax, block, color, marker in zip(
        axes, (128, 512), (BLUE, ORANGE), ("o", "s"), strict=True
    ):
        report = reports[f"b{block}"]
        control = report["means"]["b256-w512"]
        metric = "decode_ms_per_batch_step"
        control_latency = median(row[metric] for row in report["paired_speed_controls"].values())
        control_offset = (6, -16) if block == 128 else (6, 6)
        points = [(control_latency, control, None, "256 control", control_offset)]
        offsets = [(-6, -12), (-6, 7), (6, 6), (7, -5)]
        for n in range(4):
            variant = f"b{block}-w{block * (n + 1)}"
            y = report["means"][variant]
            bounds = shift_bounds(report["paired_bootstrap_95_intervals"][variant], control)
            x = report["serving_checks"][variant][metric]
            points.append((x, y, bounds, str(n), offsets[n]))
        quality_speed_points(ax, points, (dense_speed, dense), "decode", color, marker)
        ax.set_title(f"{block}-token blocks", loc="left", pad=8)
    xlim = (min(ax.get_xlim()[0] for ax in axes), max(ax.get_xlim()[1] for ax in axes))
    ylim = (min(ax.get_ylim()[0] for ax in axes), max(ax.get_ylim()[1] for ax in axes))
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    axes[1].set_ylabel("")
    figure_legend(fig, axes[0])
    return fig


def prefill_figure(data, *followups, dense_loss=None):
    reports = {"16k": data}
    reference = [doc["token_sha256"] for doc in data["quality"]["16k"]["samples"]]
    for report in (data, *(study for study in followups if study is not None)):
        if [doc["token_sha256"] for doc in report["quality"]["16k"]["samples"]] != reference:
            raise ValueError("Mismatched ProLong cohorts across prefill studies")
        for block in report["speed"]:
            if block in reports:
                raise ValueError(f"Duplicate prefill block {block}")
            reports[block] = report
    sizes = sorted(reports, key=lambda block: reports[block]["quality"][block]["memory"][
        "configured_prefill_block_tokens"])
    control_x = sizes.index("16k")
    loss_note = "Loss uses eight paired ProLong documents."
    block_note = "The first exact block, scheduler chunk and prefill cache catch-up change together."
    if "256" in reports:
        block_note += " The 256 and 512 results use the repaired handoff."
    elif "512" in reports:
        block_note += " The 512 results use the corrected cache handoff."
    if dense_loss is not None:
        if [doc["token_sha256"] for doc in dense_loss["samples"]] != reference:
            raise ValueError("Mismatched dense ProLong loss inputs")
        loss_delta = dense_loss["loss"] - dense_loss["lod_16k_loss"]
        loss_note += f" Dense loss changes by {loss_delta:+.6f} nats/token versus 16K LoD."
    fig, (a, b, c) = canvas(
        f"{block_note}"
        f"\n{loss_note}"
        "\nSpeed includes the first decode step. Peak memory is PyTorch reserved. Deltas use matched 16K controls.",
        cols=3, width=10.6, height=4.05, left=0.09, wspace=0.47, bottom=0.30,
    )
    latency, memory = [], []
    for x, block in enumerate(sizes):
        report = reports[block]
        loss = report["quality"]
        refs = loss["16k"]["samples"]
        per_doc = []
        for sample, ref in zip(loss[block]["samples"], refs, strict=True):
            if sample["token_sha256"] != ref["token_sha256"]:
                raise ValueError("Mismatched ProLong loss inputs")
            per_doc.append(1000 * (sample["loss"] - ref["loss"]))
        dx = [(i - (len(per_doc) - 1) / 2) * 0.037 for i in range(len(per_doc))]
        a.scatter([x + j for j in dx], per_doc, color=BLUE, alpha=0.47, s=15, zorder=3,
                  label="Document" if x == 0 else None)
        avg = 1000 * (loss[block]["loss"] - loss["16k"]["loss"])
        a.scatter(x, avg, marker="D", color=INK, s=23, zorder=4,
                  label="Mean" if x == 0 else None)
        if block == "16k":
            latency.append(0)
            memory.append(0)
        else:
            pair = report["speed"][block]
            latency.append(100 * (pair["two_token_candidate_s"] / pair["two_token_control_s"] - 1))
            memory.append((pair["memory"]["peak_reserved_bytes"] -
                           pair["control_memory"]["peak_reserved_bytes"]) / 2**30)
    a.set_title("Prompt loss", loc="left", pad=8)
    a.set_ylabel("Change (×10⁻³ nats/token)")
    if dense_loss is not None:
        dense_reference(a, 1000 * loss_delta)
    for ax, values, title, units in (
        (b, latency, "Latency through two tokens", "Latency change (%)"),
        (c, memory, "Peak reserved memory", "Memory change (GiB)"),
    ):
        ax.set_title(title, loc="left", pad=8)
        ax.set_ylabel(units)
        colors = [BLUE if x < control_x else CONTROL if x == control_x else ORANGE
                  for x in range(len(sizes))]
        ax.bar(range(len(sizes)), values, width=0.46, color=colors)
        ax.scatter(control_x, 0, s=19, marker="D", color=CONTROL, zorder=4)
    for ax in (a, b, c):
        zero(ax, label="16K LoD control")
        xlabels(ax, [x.upper() for x in sizes], "Prefill block")
    figure_legend(fig, a)
    return fig


def pages_figure(data, dense_tasks, full_panel=None):
    labels_by_key = {
        "two-tier": ("Two-tier", CONTROL), "fixed-1": ("Fixed 1", "#527A9F"),
        "fixed-2": ("Fixed 2", BLUE), "fixed-4": ("Fixed 4", ORANGE),
        "adaptive-0.5": ("Adaptive 0.5", GREEN),
    }
    keys = [key for key in labels_by_key if key in data["retrieval"]]
    labels, colors = zip(*(labels_by_key[key] for key in keys), strict=True)
    quality_note = "Targeted retrieval on Multikey-2/3, with top-8 prefill and region c=16 in decode."
    work_note = "Work averages the last decode step across requests, layers and heads."
    if full_panel is not None:
        quality_note = "All 13 RULER tasks, with top-8 prefill and region c=16 in decode. Bars are paired 95% intervals."
        work_note = "Exact work uses last-step Multikey-2/3 diagnostics."
    fig, (a, b, c, d) = canvas(
        f"{quality_note}"
        f"\n{work_note} Latency uses same-GPU two-tier controls."
        f"\n{DENSE_NOTE}",
        rows=2, width=9.6, height=6.15, left=0.11, bottom=0.195, top=0.89, hspace=0.65,
    )
    xs = list(range(len(keys)))
    tasks = ["niah_multikey_2", "niah_multikey_3"]
    retrieval = data["retrieval"]
    if full_panel is None:
        for task, label, color, marker, offset in zip(
            tasks, ["Multikey-2", "Multikey-3"], [BLUE, ORANGE], ["o", "s"], [-0.10, 0.10], strict=True
        ):
            series(a, [x + offset for x in xs], [retrieval[v][task]["score"] for v in keys],
                   color=color, marker=marker, label=label, linestyle="None")
        for score in sorted({dense_tasks[task] for task in tasks}):
            dense_reference(a, score)
        a.set_title("Retrieval quality", loc="left", pad=8)
        floor = min(retrieval[v][t]["score"] for v in keys for t in tasks)
        a.set_ylim(min(85, floor - 2.5), 102.4)
    else:
        results = full_panel["results"]
        control = results["two-tier"]["ruler_mean"]
        for x, key, color in zip(xs, keys, colors, strict=True):
            row = results[key]
            bounds = shift_bounds(row["paired_bootstrap_95_interval_points"], control)
            interval(a, x, bounds, color)
            a.scatter(x, row["ruler_mean"], color=color, s=25, zorder=3)
        dense_reference(a, full_panel["dense_score"])
        a.set_title("RULER macro-average", loc="left", pad=8)
    a.set_ylabel("RULER score (%)")
    figure_legend(fig, a)
    for ax, metric, title, ylabel in (
        (b, "exact_pages_per_head", "Opened pages", "Exact pages per head"),
        (c, "exact_entries_per_head", "Opened remote entries", "Exact entries per head"),
    ):
        for x, key, color in zip(xs, keys, colors, strict=True):
            values = [retrieval[key][t][metric] for t in tasks]
            if values[0] is None:
                ax.text(x, 0.05, "—", transform=ax.get_xaxis_transform(), ha="center")
            else:
                ax.bar(x, mean(values), width=0.53, color=color)
        ax.set_title(title, loc="left", pad=8)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
    d.bar(xs[1:], [data["speed"][v]["decode_change_percent"] for v in keys[1:]],
          width=0.53, color=colors[1:])
    d.scatter(0, 0, color=CONTROL, marker="D", s=20, zorder=4)
    zero(d)
    d.set_ylim(bottom=min(-0.5, d.get_ylim()[0]))
    d.set_title("Decode latency", loc="left", pad=8)
    d.set_ylabel("Latency change (%)")
    for ax in (a, b, c, d):
        xlabels(ax, labels, "Decode refinement")
    return fig


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=BASE)
    parser.add_argument("--output-dir", type=Path, default=BASE / "k2-ablation-figures")
    parser.add_argument("--preview-dir", type=Path, help="Optional raster previews for visual QA")
    parser.add_argument("--allow-incomplete", action="store_true", help="Render the available follow-ups")
    args = parser.parse_args()
    paths = {**REPORTS, **FOLLOWUPS}
    sources = {name: args.input_dir / relative for name, relative in paths.items()
               if not args.allow_incomplete or name in REPORTS or (args.input_dir / relative).is_file()}
    data = {name: json.loads(path.read_text()) for name, path in sources.items()}
    if any(report.get("stage") != "complete" for report in data.values()):
        raise ValueError("An input is missing its completed results")
    archived = data["dense"]
    if archived["examples_per_task"] != 32 or len(archived["models"]["k2"]["tasks"]) != 13:
        raise ValueError("Unexpected dense RULER cohort")
    dense = archived["models"]["k2"]
    data["dense_score"] = dense["mean"]["archived_dense"]
    dense_tasks = {row["task"]: row["archived_dense"] for row in dense["tasks"]}
    dense_comparison = data.get("dense_comparison", {})
    prefill_short = data.get("prefill_short")
    if prefill_short is not None:
        # The original 512 run read stale K/V. Retain the unaffected 1K run.
        prefill_short = {**prefill_short, "speed": {"1k": prefill_short["speed"]["1k"]}}
    figures = [
        ("01-region-threshold", threshold_figure(data)),
        ("02-fixed-region-budget", topk_figure(data)),
        ("03-centroid-schedule", centroid_figure(data["geometry"], data["dense_score"],
                                               data.get("centroid_dense_comparison"))),
        ("04-decode-window", window_figure(data["geometry"], data["dense_score"])),
        ("05-bswa-block-size", bswa_figure(data)),
        ("06-prefill-block-length", prefill_figure(data["prefill"], data.get("prefill_small"),
                                                 prefill_short, data.get("prefill_512_fixed"),
                                                 data.get("prefill_256"),
                                                 dense_loss=dense_comparison.get("prompt_loss"))),
        ("07-page-refinement", pages_figure(data.get("pages_fixed1", data["pages"]), dense_tasks,
                                           data.get("pages_full"))),
    ]
    if dense_comparison and "threshold_speed" in data:
        figures.append(("08-threshold-quality-speed", threshold_speed_figure(data)))
    if dense_comparison:
        figures.append(("09-bswa-quality-speed", bswa_speed_figure(data)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    bundle = args.output_dir / "all-ablation-plots.pdf"
    with PdfPages(bundle, metadata={**META, "Title": "K2 Horizon hyperparameter ablations at 64K"}) as pages:
        for name, fig in figures:
            output = args.output_dir / f"{name}.pdf"
            fig.savefig(output, metadata=META)
            pages.savefig(fig)
            if args.preview_dir:
                fig.savefig(args.preview_dir / f"{name}.png", dpi=135)
            plt.close(fig)
    outputs = [bundle, *(args.output_dir / f"{name}.pdf" for name, _ in figures)]
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "inputs": {paths[name]: digest(path) for name, path in sources.items()},
        "exports": {path.name: digest(path) for path in outputs},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
