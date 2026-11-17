"""Plot completed 350M component ablations against a KVM-sqrt reference."""

import argparse
import csv
import math
import sys
from pathlib import Path


OUTPUT = Path(__file__).resolve().parent
EXPERIMENTS = OUTPUT.parent
sys.path.insert(0, str(EXPERIMENTS / "hybrid-results-350M-20261111"))
from plot_hybrids import BENCHMARK_KEYS, evaluation_metrics, load_json
from plot_report import ablation_heatmap, configure_plot_style, ruler_8k_scores


CONTROL = EXPERIMENTS / "kvm-sqrt-triton-350M-20261111"
ABLATIONS = EXPERIMENTS / "ablations-350M-triton-20261111-r2"
ROPE_ABLATIONS = EXPERIMENTS / "kvm-rope-350M-triton-20261112"
SPECS = [
    ("K0", "K0  Triton KVM-sqrt", None),
    ("A1", "A1  No protected sink", "K0"),
    ("A2", "A2  Unit head temperatures", "K0"),
    ("A4", "A4  Unit merge weights", "K0"),
    ("A6", "A6  Merge into old slots only", "K0"),
    ("A3", "A3  Assignment-count readout", "K0"),
]
METRICS = ["final_dclm_nll", *BENCHMARK_KEYS]
DELTA_FIELDS = {
    key: key.removesuffix("_pct") + "_delta_pp" if key.endswith("_pct") else key + "_delta"
    for key in METRICS
}


def collect_rows():
    protocol = load_json(CONTROL / "eval/manifest.json")["protocol"]
    rows = {}
    for name, label, _ in SPECS:
        directory = CONTROL if name == "K0" else ABLATIONS / name
        pipeline = load_json(directory / "pipeline_status.json")
        status_path = directory / "eval/status.json"
        assert pipeline["state"] == load_json(status_path)["state"] == "complete", name
        assert load_json(directory / "eval/manifest.json")["protocol"] == protocol, name
        scores = evaluation_metrics(status_path, name)
        assert set(scores) == set(BENCHMARK_KEYS), name
        rows[name] = {
            "run": name, "variant": label.split("  ", 1)[1],
            "final_dclm_nll": pipeline["final_validation_loss"], **scores,
            "eval_status": str(status_path),
        }
        assert all(math.isfinite(rows[name][key]) for key in METRICS), name
    baseline = rows["K0"]
    for row in rows.values():
        row.update({delta: round(row[key] - baseline[key], 10) for key, delta in DELTA_FIELDS.items()})
    return rows


def plot_components_and_rope(rows, reference):
    rows = {name: dict(row) for name, row in rows.items()}
    if reference == "published":
        metrics_path = EXPERIMENTS / "hybrid-results-1-to-1/published_1_to_1_metrics.csv"
        with metrics_path.open() as file:
            rows[reference] = next(
                row for row in csv.DictReader(file)
                if row["scale"] == "350M" and row["model"] == "KVM sqrt"
            )
    reference_label = "K0  KVM-sqrt" if reference == "K0" else "KVM-sqrt"
    specs = [(reference, reference_label, None)] + sorted(
        (name, label, reference) for name, label, _ in SPECS[1:] if name != "A6"
    ) + [
        ("P1", "P1  Local-only RoPE", reference),
        ("P2", "P2  RoPE everywhere", reference),
    ]
    for name in ("P1", "P2"):
        directory = ROPE_ABLATIONS / name
        status = directory / "eval/status.json"
        assert load_json(status)["state"] == "complete", name
        rows[name] = evaluation_metrics(status, name)
        assert set(rows[name]) == set(BENCHMARK_KEYS), name
    scores = ruler_8k_scores()
    for name, _, _ in specs:
        run = f"350M-KVMsqrt-{reference}" if name == reference else f"350M-{name}"
        rows[name]["ruler_8k_pct"] = scores[run]
    ablation_heatmap(
        rows, specs,
        output=OUTPUT / f"kvm-component-ablations-350M-vs-{reference}.png",
        figure_title="350M KVM-sqrt ablations", boundaries=(4.5,), include_ruler_8k=True,
        color_limit=100,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    reference_options = parser.add_mutually_exclusive_group()
    reference_options.add_argument("--published", action="store_true",
                                   help="plot against the published 350M KVM-sqrt scores")
    reference_options.add_argument("--k0", action="store_true",
                                   help="plot component and RoPE ablations against K0")
    args = parser.parse_args()
    rows = collect_rows()
    configure_plot_style()
    if args.published or args.k0:
        plot_components_and_rope(rows, "K0" if args.k0 else "published")
    else:
        columns = ["run", "variant", *METRICS, *DELTA_FIELDS.values(), "eval_status"]
        with (OUTPUT / "ablation_metrics_350M.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows.values())
        ablation_heatmap(
            rows, SPECS, output=OUTPUT / "kvm-component-ablations-350M.png",
            figure_title="350M Triton KVM-sqrt ablations", boundaries=(2.5, 4.5),
        )
