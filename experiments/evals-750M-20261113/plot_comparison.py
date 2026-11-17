"""Plot the completed 750M evaluation results."""

import csv
from pathlib import Path
import sys


OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUTPUT.parent / "ablation-results-350M"))
from plot_append_policies import plot_comparison


RUNS = [
    ("GPTA2-750M", "GPTA-2"),
    ("RWKV7-750M", "RWKV-7"),
    ("KVM256-750M", "KVM-256"),
    ("KVMsqrt-750M", "KVM-sqrt"),
]


def main():
    with (OUTPUT / "metrics.csv").open(newline="") as file:
        rows = {row["run"]: row for row in csv.DictReader(file)}
    for name, _ in RUNS:
        if rows[name]["eval_state"] != "complete" or not rows[name]["ruler_8k_pct"]:
            raise ValueError(f"Evaluation results are incomplete for {name}")
    plot_comparison(rows, RUNS, OUTPUT / "750M-benchmark-comparison.png", include_ruler_8k=True)


if __name__ == "__main__":
    main()
