"""Plot the completed wave evaluations from the report CSVs."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
from matplotlib.ticker import PercentFormatter
import numpy as np


ROOT = Path(__file__).resolve().parent
LENGTHS = (4096, 8192, 16384, 32768)
TICK_LABELS = ("4K", "8K", "16K", "32K")
COLORS = ("#13526d", "#ba501d", "#357c55", "#71549c", "#ac3f70", "#946b16", "#202830")
MARKERS = ("o", "s", "^", "D", "P", "v", "X")
DASHES = ("-", "--", "-.", ":", (0, (5, 2, 1, 2)), (0, (6, 2, 1, 2, 1, 2)), "-")
BENCHMARKS = ("ruler_4k_pct", "longbench_pct", "short_mean_pct")
BENCHMARK_LABELS = ("RULER 4K", "LongBench", "Short mean")


def read_rows(name):
    with (ROOT / name).open() as source:
        return list(csv.DictReader(source))


def values(row, keys):
    return [float(row[key]) for key in keys]


def ruler_8k_scores():
    with (ROOT.parent / "ruler-8k-ablations-20261113/ruler_8k_scores.csv").open() as file:
        return {
            row["model"]: float(row["ruler_8k_pct"])
            for row in csv.DictReader(file) if row["status"] == "complete"
        }


def save(fig, name):
    fig.savefig(ROOT / name, dpi=190, bbox_inches="tight", pad_inches=0.14, facecolor="white")
    plt.close(fig)


def retrieval(ax, rows, runs, task, title, legend=False, x_label=True):
    keys = [f"s{task}_{length}_pct" for length in LENGTHS]
    for i, (run, label) in enumerate(runs):
        ax.plot(TICK_LABELS, values(rows[run], keys), label=label,
                color=COLORS[i], marker=MARKERS[i], linestyle=DASHES[i],
                linewidth=2.2 if i == 0 else 1.7, markersize=4.5)
    ax.set(title=title, ylim=(-3, 104), yticks=[0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    if x_label:
        ax.set_xlabel("Context length")
    ax.grid(axis="y")
    if legend:
        ax.legend(loc="upper right")


def benchmark_bars(ax, rows, runs, *, include_ruler_8k=False):
    keys = BENCHMARKS
    labels = BENCHMARK_LABELS
    if include_ruler_8k:
        keys = (keys[0], "ruler_8k_pct", *keys[1:])
        labels = (labels[0], "RULER 8K", *labels[1:])
    x = np.arange(len(keys))
    width = 0.74 / len(runs)
    for i, (run, label) in enumerate(runs):
        shift = (i - (len(runs) - 1) / 2) * width
        bars = ax.bar(x + shift, values(rows[run], keys), width=width * 0.92,
                      color=COLORS[i], label=label)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set(title="Broader benchmarks", xticks=x, xticklabels=labels, ylim=(0, 50))
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_axisbelow(True)
    ax.grid(axis="y")


def routing_and_budget(rows):
    append = [
        ("K0", "Least redundant"), ("E1", "Random"),
        ("E2", "Evenly spaced"), ("B1", "Bipartite"),
    ]
    schedule = [("K0", "Square root"), ("S1", "Linear cap at 4K"), ("S2", "Linear cap at 8K")]
    fig, axes = plt.subplots(2, 2, figsize=(10.1, 6.25), layout="constrained")
    for row, runs, title in zip(axes, (append, schedule), ("Routing policies", "State budgets")):
        for task, ax in zip((2, 3), row):
            retrieval(ax, rows, runs, task, f"{title} · NIAH-S{task}",
                      legend=task == 3, x_label=title == "State budgets")
    save(fig, "routing-and-budgets.png")


def ablation_heatmap(
    rows, specs=None, *, output="kvm-component-ablations.png",
    figure_title=None, boundaries=(2.5, 4.5, 7.5), include_ruler_8k=False,
    color_limit=None, font_scale=1,
):
    if specs is None:
        specs = [
            ("K0", "K0  Square-root reference", None),
            ("A1", "A1  No protected sink", "K0"),
            ("A2", "A2  Unit head temperatures", "K0"),
            ("A4", "A4  Unit merge weights", "K0"),
            ("A6", "A6  Merge into old slots only", "K0"),
            ("A3", "A3  Corrected count", "K0"),
            ("A5", "A5  Gate-weight denominator", "K0"),
            ("A7", "A7  Learned per-head radius", "K0"),
            ("F0", "F0  Fixed-256 reference", None),
            ("F1", "F1  Corrected count at 256", "F0"),
        ]
    benchmark_keys = BENCHMARKS
    benchmark_labels = BENCHMARK_LABELS
    if include_ruler_8k:
        benchmark_keys = (BENCHMARKS[0], "ruler_8k_pct", *BENCHMARKS[1:])
        benchmark_labels = (BENCHMARK_LABELS[0], "RULER 8K", *BENCHMARK_LABELS[1:])
    panels = [
        (f"NIAH-S{task}", [f"s{task}_{length}_pct" for length in LENGTHS], TICK_LABELS)
        for task in (1, 2, 3)
    ] + [("Broader benchmarks", benchmark_keys, benchmark_labels)]
    references = [i for i, (_, _, control) in enumerate(specs) if control is None]
    displayed = []
    for _, keys, _ in panels:
        matrix = np.array([values(rows[run], keys) for run, _, _ in specs])
        for i, (_, _, control) in enumerate(specs):
            if control:
                matrix[i] -= values(rows[control], keys)
        displayed.append(matrix)
    if color_limit is None:
        differences = np.concatenate(displayed, axis=1)
        maximum = np.max(np.abs(np.delete(differences, references, axis=0)))
        color_limit = 10 * math.ceil(maximum / 10)
    limit = color_limit
    norm = CenteredNorm(halfrange=limit)

    height = max(4.1, 1.75 + 0.38 * len(specs)) + 0.9 * (font_scale - 1)
    width = 15.2 if include_ruler_8k else 14.3
    benchmark_width = len(benchmark_keys) + 0.4 + 5 * (font_scale - 1)
    fig, axes = plt.subplots(1, 4, figsize=(width, height), sharey=True,
                             width_ratios=(4, 4, 4, benchmark_width),
                             layout="constrained")
    vector_pdf = Path(output).suffix.lower() == ".pdf"
    if figure_title:
        fig.suptitle(figure_title, fontsize=15 * font_scale, fontweight="semibold")
    for i, (ax, panel, matrix) in enumerate(zip(axes, panels, displayed)):
        title, _, ticks = panel
        colored = matrix.copy()
        colored[references] = np.nan
        if vector_pdf:
            x_edges = np.arange(len(ticks) + 1) - 0.5
            y_edges = np.arange(len(specs) + 1) - 0.5
            image = ax.pcolormesh(x_edges, y_edges, colored, cmap="RdBu", norm=norm)
            ax.set_ylim(y_edges[-1], y_edges[0])
        else:
            image = ax.imshow(colored, aspect="auto", cmap="RdBu", norm=norm)
        for y in references:
            ax.axhspan(y - 0.5, y + 0.5, color="#eaf0f4")
        for (y, x), number in np.ndenumerate(matrix):
            if y in references or abs(number) < 0.05:
                label = f"{abs(number):.1f}"
            else:
                label = f"{number:+.1f}".replace("-", "−")
            ax.text(x, y, label, ha="center", va="center", fontsize=9 * font_scale,
                    color="white" if y not in references and abs(number) > limit * 0.68 else "#182832",
                    fontweight="semibold" if y in references else "normal")
        ax.set_title(title, fontsize=11 * font_scale, pad=8)
        ax.set_xticks(range(len(ticks)), ticks, fontsize=9 * font_scale)
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
        ax.set_yticks(range(len(specs)), [label for _, label, _ in specs], fontsize=10 * font_scale)
        ax.tick_params(axis="y", labelleft=i == 0, length=0, pad=7)
        for boundary in np.arange(0.5, len(specs) - 0.5, 1):
            ax.axhline(boundary, color="white", linewidth=1)
        for boundary in boundaries:
            ax.axhline(boundary, color="#adbbc3", linewidth=1)
        for boundary in np.arange(0.5, len(ticks) - 0.5, 1):
            ax.axvline(boundary, color="white", linewidth=1)
        ax.set_xlim(-0.5, len(ticks) - 0.5)
        ax.spines[:].set_visible(False)
    scale = fig.colorbar(image, ax=axes, orientation="horizontal", shrink=0.36, pad=0.09, aspect=32)
    if vector_pdf:
        scale.solids.set_rasterized(False)
    scale.set_label("Change from reference (percentage points)")
    scale.ax.xaxis.label.set_fontsize(scale.ax.xaxis.label.get_fontsize() * font_scale)
    scale.set_ticks([-limit, -limit / 2, 0, limit / 2, limit])
    tick_size = scale.ax.get_xticklabels()[0].get_fontsize()
    scale.ax.tick_params(axis="x", labelsize=tick_size * font_scale)
    save(fig, output)


def hybrids(rows):
    runs = [
        ("K0", "K0  All KVM-sqrt"), ("H1", "H1  SWA / full"),
        ("H2", "H2  RWKV-7 / full"), ("H3", "H3  SWA / KVM-sqrt"),
    ]
    if "H4" in rows:
        runs.append(("H4", "H4  RWKV-7 / KVM-sqrt"))
    fig, axes = plt.subplots(2, 2, figsize=(10.1, 6.25), layout="constrained")
    for task, ax in enumerate(axes.flat, 1):
        if task <= 3:
            retrieval(ax, rows, runs, task, f"NIAH-S{task}", legend=task == 3, x_label=task == 3)
        else:
            benchmark_bars(ax, rows, runs)
    save(fig, "hybrids-3-to-1.png")


def qwen(rows, rates):
    runs = [
        ("Q0", "Q0 full · 2e-3"), ("Q0 tuned", "Q0 full · 4e-3"),
        ("Q1", "Q1 KVM-sqrt · 4e-3"),
        ("120M GPTA-2", "GPTA-2 full · paper"),
        ("120M KVM sqrt", "GPTA-2 KVM-sqrt · paper"),
    ]
    rows = rows | {row["model"]: row for row in read_rows("published_metrics.csv")}
    fig = plt.figure(figsize=(10.8, 6.6), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=(0.15, 1, 1))
    legend = fig.add_subplot(grid[0, :])
    legend.axis("off")
    sweep, s1, s3, benchmarks = (
        fig.add_subplot(grid[y, x]) for y in (1, 2) for x in (0, 1)
    )
    fig.suptitle("120M Qwen3-style and GPTA-2 backbones", fontsize=15, fontweight="semibold")
    learning_rates = values_by_row(rates, "peak_lr")
    losses = values_by_row(rates, "final_validation_loss")
    selected = next(i for i, rate in enumerate(rates) if rate["selected"] == "True")
    sweep.plot(learning_rates, losses, color=COLORS[0], marker="o")
    sweep.scatter(learning_rates[selected], losses[selected], color=COLORS[1], zorder=3, s=55)
    sweep.annotate("Selected 4e-3", (learning_rates[selected], losses[selected]),
                   xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9)
    sweep.set(title="Q0 learning-rate sweep", xlabel="Peak learning rate (×10⁻³)",
              ylabel="Final DCLM NLL", xscale="log", xticks=learning_rates,
              xticklabels=[f"{rate * 1000:g}" for rate in learning_rates])
    sweep.minorticks_off()
    sweep.grid(axis="y")
    retrieval(s1, rows, runs, 1, "NIAH-S1", x_label=False)
    retrieval(s3, rows, runs, 3, "NIAH-S3")
    benchmark_bars(benchmarks, rows, runs)
    for label in benchmarks.texts:
        label.set_fontsize(7)
    handles, labels = s1.get_legend_handles_labels()
    legend.legend(handles, labels, ncols=len(runs), loc="center")
    save(fig, "qwen-controls.png")


def s2_endpoint_comparison(rows):
    status_path = ROOT.parent / "S2-sqrt-matched-eval-20261112" / "status.json"
    status = json.loads(status_path.read_text())
    if status["state"] != "complete":
        raise ValueError(f"S2 matched-cap evaluation is incomplete: {status_path}")
    metrics = {stage["suite"]: stage.get("metrics", {}) for stage in status["stages"]}
    matched = {}
    for length in LENGTHS:
        suite = "niah-32k" if length == 32768 else "niah-4k-8k-16k"
        for task in (1, 2, 3):
            matched[f"s{task}_{length}_pct"] = 100 * metrics[suite][f"niah_single_{task}/{length},none"]
    matched["ruler_4k_pct"] = 100 * metrics["ruler-4k"]["ruler/4096,none"]
    matched["longbench_pct"] = 100 * metrics["longbench-fewshot"]["longbench_fewshot/score,none"]
    short_metrics = (
        "arc_challenge/acc_norm,none", "arc_easy/acc,none", "hellaswag/acc_norm,none",
        "lambada_openai/acc,none", "piqa/acc,none", "winogrande/acc,none",
    )
    matched["short_mean_pct"] = 100 * sum(metrics["short"][key] for key in short_metrics) / len(short_metrics)
    rows = rows | {row["model"]: row for row in read_rows("published_metrics.csv")} | {"S2 matched": matched}
    runs = (
        ("S2 matched", "S2 (matched cap)"), ("S2", "S2 (original cap)"),
        ("K0", "KVM-sqrt (K0)"), ("120M KVM sqrt", "KVM-sqrt (paper)"),
        ("120M KVM 256", "KVM-256 (paper)"),
    )

    fig = plt.figure(figsize=(11.6, 6.4), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=(0.14, 1, 1))
    legend = fig.add_subplot(grid[0, :])
    legend.axis("off")
    s1, s2, s3, benchmarks = (fig.add_subplot(grid[y, x]) for y in (1, 2) for x in (0, 1))
    for task, axis in enumerate((s1, s2, s3), 1):
        retrieval(axis, rows, runs, task, f"NIAH-S{task}", x_label=task == 3)
        axis.set_ylabel("Accuracy")
    s1.set(ylim=(90, 101), yticks=(90, 92, 94, 96, 98, 100))
    benchmark_bars(benchmarks, rows, runs)
    benchmarks.set_ylabel("Score")
    for label in benchmarks.texts:
        label.set_fontsize(7)
    handles, labels = s1.get_legend_handles_labels()
    legend.legend(handles, labels, ncols=len(runs), loc="center", columnspacing=1.7)
    plt.rcParams["pdf.fonttype"] = 42
    fig.savefig(ROOT / "s2-endpoint-comparison.pdf", bbox_inches="tight", pad_inches=0.14, facecolor="white")
    save(fig, "s2-endpoint-comparison.png")


def values_by_row(rows, key):
    return [float(row[key]) for row in rows]


def configure_plot_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#c2cbd1", "grid.color": "#e2e8eb",
        "legend.frameon": False, "legend.fontsize": 9, "text.color": "#182832",
        "axes.labelcolor": "#182832", "xtick.color": "#2c3f49", "ytick.color": "#2c3f49",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s2-endpoint", action="store_true", help="render the S2 matched-cap comparison")
    args = parser.parse_args()
    configure_plot_style()
    rows = {row["run"]: row for row in read_rows("wave_metrics.csv")}
    if args.s2_endpoint:
        s2_endpoint_comparison(rows)
        return
    routing_and_budget(rows)
    ablation_heatmap(rows)
    hybrids(rows)
    qwen(rows, read_rows("qwen_lr.csv"))


if __name__ == "__main__":
    main()
