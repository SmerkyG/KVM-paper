"""Refresh the 350M hybrid plot and CSV from saved evaluation results."""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from statistics import mean


OUTPUT = Path(__file__).resolve().parent
EXPERIMENTS = OUTPUT.parent
sys.path.insert(0, str(EXPERIMENTS / "wave-results-20261111"))
from plot_report import BENCHMARKS, LENGTHS, benchmark_bars, plt, retrieval


RUNS = {
    "K0": ("All KVM-sqrt", EXPERIMENTS / "kvm-sqrt-triton-350M-20261111"),
    "H1": ("SWA / full", EXPERIMENTS / "wave5-350M-20261111/H1"),
    "H2": ("RWKV-7 / full", EXPERIMENTS / "wave5-350M-20261111/H2"),
    "H3": ("SWA / KVM-sqrt", EXPERIMENTS / "wave5-350M-triton-20261111/H3"),
    "H4": ("RWKV-7 / KVM-sqrt", EXPERIMENTS / "wave5-350M-triton-20261111/H4"),
    "H5": ("KVM-256 / full", EXPERIMENTS / "kvm-full-hybrids-350M-20261112/H5"),
    "H6": ("KVM-sqrt / full", EXPERIMENTS / "kvm-full-hybrids-350M-20261112/H6"),
}
SHORT_METRICS = {
    "lambada_openai": "acc,none",
    "piqa": "acc,none",
    "winogrande": "acc,none",
    "arc_easy": "acc,none",
    "arc_challenge": "acc_norm,none",
    "hellaswag": "acc_norm,none",
}
NIAH_KEYS = [f"s{task}_{length}_pct" for task in (1, 2, 3) for length in LENGTHS]
BENCHMARK_KEYS = [*NIAH_KEYS, *BENCHMARKS, "textbook_32k_nll"]


def load_json(path):
    return json.loads(path.read_text())


def evaluation_metrics(status_path, name):
    if not status_path.is_file():
        return {}
    stages = {
        row["suite"]: load_json(Path(row["result_file"]))
        for row in load_json(status_path)["stages"]
        if row["model"] == name and row["state"] == "complete" and row["exit_code"] == 0
    }
    metrics = {}
    for length in LENGTHS:
        suite = "niah-32k" if length == 32768 else "niah-4k-8k-16k"
        if suite in stages:
            for task in (1, 2, 3):
                values = stages[suite]["results"][f"niah_single_{task}"]
                score = values[f"{length},none"]
                assert 0 <= score <= 1
                metrics[f"s{task}_{length}_pct"] = 100 * score
    if "ruler-4k" in stages:
        metrics["ruler_4k_pct"] = 100 * stages["ruler-4k"]["groups"]["ruler"]["4096,none"]
    if "longbench-fewshot" in stages:
        group = stages["longbench-fewshot"]["groups"]["longbench_fewshot"]
        metrics["longbench_pct"] = 100 * group["score,none"]
    if "short" in stages:
        results = stages["short"]["results"]
        metrics["short_mean_pct"] = 100 * mean(results[task][metric] for task, metric in SHORT_METRICS.items())
    if "textbook-32k" in stages:
        metrics["textbook_32k_nll"] = stages["textbook-32k"]["avg_nll"]
    return metrics


def collect_rows():
    rows = {}
    validation = {}
    for name, (label, directory) in RUNS.items():
        train_log = directory / "runs" / name / "train.log"
        text = train_log.read_text()
        validation[name] = {
            int(step): float(loss)
            for step, loss in re.findall(r"^step:(\d+)/15000 val_loss:([0-9.]+)", text, re.M)
        }
        latest_step = max(int(step) for step in re.findall(r"^step:(\d+)/15000 ", text, re.M))
        status = load_json(directory / "pipeline_status.json")
        eval_status = directory / "eval/status.json"
        rows[name] = {
            "run": name, "variant": label, "state": status["state"], "latest_step": latest_step,
            "final_dclm_nll": status.get("final_validation_loss"),
            **evaluation_metrics(eval_status, name),
            "train_log": str(train_log), "eval_status": str(eval_status) if eval_status.is_file() else "",
        }
    matched_step = max(set.intersection(*(set(losses) for losses in validation.values())))
    for name, row in rows.items():
        row.update(matched_step=matched_step, matched_dclm_nll=validation[name][matched_step])
    return rows


def plot(rows, *, pdf=False):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#c2cbd1", "grid.color": "#e2e8eb",
        "legend.frameon": False, "legend.fontsize": 9, "text.color": "#182832",
        "axes.labelcolor": "#182832", "xtick.color": "#2c3f49", "ytick.color": "#2c3f49",
    })
    runs = [
        (name, f"{name}  {label}") for name, (label, _) in RUNS.items()
        if name != "K0" and any(key in rows[name] for key in [*NIAH_KEYS, *BENCHMARKS])
    ]
    if pdf:
        required_metrics = {*NIAH_KEYS, *BENCHMARKS}
        for name in RUNS:
            row = rows[name]
            if row["state"] != "complete" or not required_metrics <= row.keys():
                raise ValueError(f"Benchmark results are incomplete for {name}")
        runs.append(("K0", "K0  KVM-sqrt (non-hybrid)"))
    metrics = {
        name: {key: row.get(key, float("nan")) for key in BENCHMARK_KEYS}
        for name, row in rows.items()
    }
    fig = plt.figure(figsize=(15.2 if pdf else 12.4, 6.8), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=(0.23, 1, 1))
    legend = fig.add_subplot(grid[0, :])
    legend.axis("off")
    axes = [fig.add_subplot(grid[y, x]) for y in (1, 2) for x in (0, 1)]
    if not pdf:
        fig.suptitle("350M 3:1 hybrids", fontsize=15, fontweight="semibold")
    for task, ax in enumerate(axes[:3], 1):
        retrieval(ax, metrics, runs, task, f"NIAH-S{task}", x_label=task == 3)
    benchmark_bars(axes[3], metrics, runs)
    for label in axes[3].texts:
        label.set_fontsize(7)
    peak = max(row[key] for row in rows.values() for key in BENCHMARKS if key in row)
    axes[3].set_ylim(0, max(50, 5 * math.ceil((peak + 3) / 5)))
    handles, labels = axes[0].get_legend_handles_labels()
    # Matplotlib fills legend columns first. Arrange entries to read by row.
    order = [index for column in range(3) for index in range(column, len(handles), 3)]
    legend.legend([handles[index] for index in order], [labels[index] for index in order],
                  ncols=3, loc="center")
    pending = [f"{name} · {row['state']}" for name, row in rows.items()
               if name != "K0" and row["state"] != "complete"]
    if pending:
        axes[3].set_xlabel(" · ".join(pending), loc="right")
    filename = "hybrids-h1-h6-350M-vs-K0.pdf" if pdf else "hybrids-h1-h6-350M.png"
    with plt.rc_context({"pdf.fonttype": 42}):
        fig.savefig(OUTPUT / filename, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", action="store_true", help="export a title-free PDF with the non-hybrid K0 baseline")
    args = parser.parse_args()
    rows = collect_rows()
    if args.pdf:
        plot(rows, pdf=True)
        return
    columns = [
        "run", "variant", "state", "latest_step", "matched_step", "matched_dclm_nll",
        "final_dclm_nll", *BENCHMARK_KEYS, "train_log", "eval_status",
    ]
    with (OUTPUT / "hybrid_metrics_350M.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows.values())
    plot(rows)


if __name__ == "__main__":
    main()
