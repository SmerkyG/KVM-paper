"""Plot the 120M RoPE ablations against the published KVM-256 scores."""

import csv
import sys
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
EXPERIMENTS = OUTPUT.parent
sys.path.insert(0, str(EXPERIMENTS / "ablation-results-350M"))
from plot_append_policies import plot_comparison
from plot_hybrids import BENCHMARK_KEYS, evaluation_metrics, load_json


RUNS = [
    ("published", "KVM-256"),
    ("P1", "P1 · Local-only RoPE"),
    ("P2", "P2 · RoPE everywhere"),
]


def collect_rows():
    published = EXPERIMENTS / "wave-results-20261111/published_metrics.csv"
    with published.open() as file:
        baseline = next(row for row in csv.DictReader(file) if row["model"] == "120M KVM 256")
    rows = {"published": baseline}
    for name, _ in RUNS[1:]:
        directory = OUTPUT / name
        if load_json(directory / "pipeline_status.json")["state"] != "complete":
            raise ValueError(f"{name} has incomplete results")
        rows[name] = evaluation_metrics(directory / "eval/status.json", name)
    plotted_keys = {key for key in BENCHMARK_KEYS if key.endswith("_pct")}
    for name, row in rows.items():
        if not plotted_keys <= row.keys():
            raise ValueError(f"Missing plot metrics for {name}")
    return rows


if __name__ == "__main__":
    plot_comparison(collect_rows(), RUNS, OUTPUT / "kvm256-rope-ablations-120M.png", benchmark_max=50)
