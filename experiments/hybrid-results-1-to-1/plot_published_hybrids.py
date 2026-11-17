"""Plot the published 1:1 hybrid results from Tables 2 and 4."""

import csv
import sys
from pathlib import Path


OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUTPUT.parent / "wave-results-20261111"))
from plot_report import benchmark_bars, plt, retrieval


def plot(scale, rows):
    runs = [(name, row["label"]) for name, row in rows.items()]
    fig = plt.figure(figsize=(10.3, 6.5), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=(0.15, 1, 1))
    legend = fig.add_subplot(grid[0, :])
    legend.axis("off")
    axes = [fig.add_subplot(grid[y, x]) for y in (1, 2) for x in (0, 1)]
    fig.suptitle(f"Published {scale} KVM-sqrt and 1:1 hybrids", fontsize=15, fontweight="semibold")
    for task, ax in enumerate(axes[:3], 1):
        retrieval(ax, rows, runs, task, f"NIAH-S{task}", x_label=task == 3)
    benchmark_bars(axes[3], rows, runs)
    for label in axes[3].texts:
        label.set_fontsize(7)
    handles, labels = axes[0].get_legend_handles_labels()
    legend.legend(handles, labels, ncols=len(runs), loc="center")
    fig.savefig(OUTPUT / f"hybrids-1-to-1-{scale}.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#c2cbd1", "grid.color": "#e2e8eb",
        "legend.frameon": False, "legend.fontsize": 9, "text.color": "#182832",
    })
    with (OUTPUT / "published_1_to_1_metrics.csv").open() as file:
        metrics = list(csv.DictReader(file))
    for scale in ("120M", "350M"):
        plot(scale, {row["model"]: row for row in metrics if row["scale"] == scale})
