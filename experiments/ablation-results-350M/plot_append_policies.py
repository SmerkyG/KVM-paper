"""Compare KVM-sqrt append and merge policies at 120M and 350M."""

import argparse
import csv
import sys
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
EXPERIMENTS = OUTPUT.parent
sys.path.insert(0, str(EXPERIMENTS / "hybrid-results-350M-20261111"))
from plot_hybrids import BENCHMARK_KEYS, evaluation_metrics, load_json
from plot_report import benchmark_bars, configure_plot_style, plt, read_rows, retrieval, ruler_8k_scores


RUNS = [
    ("published", "KVM-sqrt"),
    ("E1", "E1 · Random append"),
    ("E2", "E2 · Evenly spaced append"),
]
BIPARTITE = ("B1", "B1 · Bipartite")
RUNS_TRITON_K0 = [("K0", "K0 · KVM-sqrt"), *RUNS[1:], BIPARTITE]
RUNS_120M = [
    ("K0", "K0 · Least-redundant append"),
    *RUNS[1:],
    ("B1", "B1 · Bipartite merge"),
]


def collect_rows(runs=RUNS):
    rows = {}
    routing = EXPERIMENTS / "routing-350M-retries-20261112"
    for name, _ in runs:
        if name == "published":
            published = EXPERIMENTS / "hybrid-results-1-to-1/published_1_to_1_metrics.csv"
            with published.open() as file:
                rows[name] = next(
                    row for row in csv.DictReader(file)
                    if row["scale"] == "350M" and row["model"] == "KVM sqrt"
                )
            continue
        run = EXPERIMENTS / "kvm-sqrt-triton-350M-20261111" if name == "K0" else routing / name
        status = run / "eval/status.json"
        if load_json(run / "pipeline_status.json")["state"] != "complete":
            raise ValueError(f"{name} has incomplete results")
        rows[name] = evaluation_metrics(status, name)
    plotted_keys = {key for key in BENCHMARK_KEYS if key.endswith("_pct")}
    for name, row in rows.items():
        if not plotted_keys <= row.keys():
            raise ValueError(f"Missing plot metrics for {name}")
    return rows


def plot_comparison(rows, runs, output, *, benchmark_max=55, include_ruler_8k=False):
    configure_plot_style()
    fig = plt.figure(figsize=(12.2 if include_ruler_8k else 10.8, 6.6), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=(0.15, 1, 1))
    legend = fig.add_subplot(grid[0, :])
    legend.axis("off")
    axes = [fig.add_subplot(grid[y, x]) for y in (1, 2) for x in (0, 1)]
    for task, ax in enumerate(axes[:3], 1):
        retrieval(ax, rows, runs, task, f"NIAH-S{task}", x_label=task == 3)
    benchmark_bars(axes[3], rows, runs, include_ruler_8k=include_ruler_8k)
    axes[3].set_ylim(0, benchmark_max)
    handles, labels = axes[0].get_legend_handles_labels()
    legend.legend(handles, labels, ncols=len(runs), loc="center")
    fig.savefig(output, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--b1", action="store_true", help="include the 350M bipartite merge run")
    modes.add_argument("--k0", action="store_true", help="plot all 350M runs against Triton K0")
    modes.add_argument("--120m", dest="small_model", action="store_true",
                       help="plot the 120M runs against original K0")
    parser.add_argument("--ruler-8k", action="store_true", help="include RULER 8K in the --k0 plot")
    parser.add_argument("--pdf", action="store_true", help="export a vector PDF")
    args = parser.parse_args()
    if args.ruler_8k and not args.k0:
        parser.error("--ruler-8k requires --k0")
    if args.small_model:
        names = {name for name, _ in RUNS_120M}
        rows = {row["run"]: row for row in read_rows("wave_metrics.csv") if row["run"] in names}
        output = EXPERIMENTS / "wave-results-20261111/kvm-append-policies-120M-with-B1.png"
        plot_comparison(rows, RUNS_120M, output.with_suffix(".pdf") if args.pdf else output)
        return
    k0_filename_label = "K0" if args.pdf else "triton-K0"
    if args.k0:
        runs = RUNS_TRITON_K0
        filename = f"kvm-append-policies-350M-vs-{k0_filename_label}.png"
    elif args.b1:
        runs = [*RUNS, BIPARTITE]
        filename = "kvm-append-policies-350M-with-B1.png"
    else:
        runs = RUNS
        filename = "kvm-append-policies-350M.png"
    rows = collect_rows(runs)
    if args.ruler_8k:
        scores = ruler_8k_scores()
        ruler_runs = {"K0": "350M-KVMsqrt-K0", "E1": "350M-E1", "E2": "350M-E2", "B1": "350M-B1"}
        pending = [name for name, run in ruler_runs.items() if run not in scores]
        if pending:
            parser.exit(1, f"RULER 8K results pending for {', '.join(pending)}\n")
        for name, run in ruler_runs.items():
            rows[name]["ruler_8k_pct"] = scores[run]
        filename = f"kvm-append-policies-350M-vs-{k0_filename_label}-ruler-8k.png"
    output = OUTPUT / filename
    plot_comparison(rows, runs, output.with_suffix(".pdf") if args.pdf else output,
                    include_ruler_8k=args.ruler_8k)


if __name__ == "__main__":
    main()
