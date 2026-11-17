"""Plot 120M square-root state ablations against the published or matched K0 reference."""

import argparse
import sys
from pathlib import Path

from plot_report import ablation_heatmap, configure_plot_style, plt, read_rows, ruler_8k_scores


EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS / "hybrid-results-350M-20261111"))
from plot_hybrids import evaluation_metrics


SPECS = [
    ("published", "KVM-sqrt", None),
    ("A1", "A1  No protected sink", "published"),
    ("A2", "A2  Unit head temperatures", "published"),
    ("A3", "A3  Assignment-count readout", "published"),
    ("A4", "A4  Unit merge weights", "published"),
    ("P1", "P1  Local-only RoPE", "published"),
    ("P2", "P2  RoPE everywhere", "published"),
]
RULER_RUNS = {
    "published": "120M-KVMsqrt-published",
    "A1": "120M-A1",
    "A2": "120M-A2",
    "A3": "120M-A3",
    "A4": "120M-A4",
    "P1": "120M-KVMsqrt-P1",
    "P2": "120M-KVMsqrt-P2",
}
MATCHED_COMPONENTS = EXPERIMENTS / "ablations-120M-triton-20261113"
MATCHED_PDF_LABELS = {
    "triton-K0": "K0  KVM-sqrt",
    "A2": "A2  No head temps",
    "A3": "A3  No v-len normalization",
    "A4": "A4  No merge gate",
}


def collect_rows(reference="published"):
    rows = {}
    if reference != "triton-K0":
        rows.update({
            row["run"]: row for row in read_rows("wave_metrics.csv")
            if row["run"] in ("A1", "A2", "A3", "A4", reference)
        })
    if reference == "published":
        rows[reference], = [
            row for row in read_rows("published_metrics.csv") if row["model"] == "120M KVM sqrt"
        ]
    elif reference == "triton-K0":
        status = EXPERIMENTS / "kvm-sqrt-triton-20261111/eval-r2/status.json"
        rows[reference] = evaluation_metrics(status, "K0")
        for name in ("A1", "A2", "A3", "A4"):
            status = MATCHED_COMPONENTS / name / "eval/status.json"
            rows[name] = evaluation_metrics(status, name)
    if reference in ("published", "triton-K0"):
        for name in ("P1", "P2"):
            status = EXPERIMENTS / "kvm-rope-120M-triton-20261113" / name / "eval/status.json"
            rows[name] = evaluation_metrics(status, name)
    return rows


def main(reference="published", *, pdf=False):
    rows = collect_rows(reference)
    reference_labels = {
        "published": "KVM-sqrt",
        "K0": "K0  Original KVM-sqrt",
        "triton-K0": "K0  Triton KVM-sqrt",
    }
    ablations = {
        "published": SPECS[1:], "K0": SPECS[1:5], "triton-K0": SPECS[1:],
    }
    specs = [(reference, reference_labels[reference], None)] + [
        (name, label, reference) for name, label, _ in ablations[reference]
    ]
    scores = ruler_8k_scores()
    ruler_runs = RULER_RUNS | {
        "K0": "120M-KVMsqrt-K0", "triton-K0": "120M-KVMsqrt-triton-K0",
    }
    if reference == "triton-K0":
        ruler_runs.update({name: f"120M-KVMsqrt-triton-{name}" for name in ("A1", "A2", "A3", "A4")})
    boundary = (4.5,) if reference != "K0" else ()
    filename = f"kvm-sqrt-ablations-120M-vs-{reference}"
    if pdf and reference == "triton-K0":
        rows["A6"] = evaluation_metrics(MATCHED_COMPONENTS / "A6/eval/status.json", "A6")
        ruler_runs["A6"] = "120M-KVMsqrt-triton-A6"
        specs.insert(5, ("A6", "O1  Merge before append", reference))
        specs = [(name, MATCHED_PDF_LABELS.get(name, label), control) for name, label, control in specs]
        boundary = (5.5,)
        filename = "kvm-sqrt-ablations-120M-vs-K0-with-O1"
    for name, _, _ in specs:
        rows[name]["ruler_8k_pct"] = scores[ruler_runs[name]]
    titles = {
        "published": "120M KVM-sqrt ablations",
        "K0": "120M KVM-sqrt ablations · Original implementation",
        "triton-K0": "120M KVM-sqrt ablations · Triton",
    }
    configure_plot_style()
    with plt.rc_context({"pdf.fonttype": 42}):
        ablation_heatmap(
            rows, specs, figure_title=None if pdf else titles[reference],
            boundaries=boundary, include_ruler_8k=True, color_limit=100,
            output=f"{filename}-150pct-text.pdf" if pdf else f"{filename}.png",
            font_scale=1.5 if pdf else 1,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument("--k0", dest="reference", action="store_const", const="K0",
                          help="compare original component ablations with original K0")
    controls.add_argument("--triton", dest="reference", action="store_const", const="triton-K0",
                          help="compare Triton component and RoPE ablations with Triton K0")
    parser.add_argument("--pdf", action="store_true", help="export a title-free vector PDF with larger text")
    parser.set_defaults(reference="published")
    args = parser.parse_args()
    main(args.reference, pdf=args.pdf)
