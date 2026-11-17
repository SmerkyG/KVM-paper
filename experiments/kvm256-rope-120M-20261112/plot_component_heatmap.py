"""Plot 120M fixed-state ablations against the published or F0 reference."""

import argparse

from plot_rope_ablations import EXPERIMENTS, collect_rows
from plot_report import ablation_heatmap, configure_plot_style, read_rows, ruler_8k_scores


# The paper's component ablations are fixed-state. The wave runs with the
# corresponding A1–A4 labels use square-root state growth.
SPECS = [
    ("published", "KVM-256", None),
    ("no sink", "A1  No protected sink", "published"),
    ("no head temps", "A2  Unit head temperatures", "published"),
    ("no v-len normalization", "A3  No value-length normalization", "published"),
    ("no merge gate", "A4  No merge gate", "published"),
    ("P1", "P1  Local-only RoPE", "published"),
    ("P2", "P2  RoPE everywhere", "published"),
]
RULER_RUNS = {
    "published": "120M-KVM256-published",
    "no sink": "120M-A1-published",
    "no head temps": "120M-A2-published",
    "no v-len normalization": "120M-A3-published",
    "no merge gate": "120M-A4-published",
    "P1": "120M-P1",
    "P2": "120M-P2",
}


def main(*, use_f0=False):
    rows = collect_rows()
    rows.update({
        row["model"]: row for row in read_rows("published_metrics.csv")
        if row["long_table"] == "S6.T3" and row["short_table"] == "A5.T5"
    })
    reference, reference_label, reference_ruler = "published", "KVM-256", RULER_RUNS["published"]
    if use_f0:
        reference, reference_label, reference_ruler = "F0", "F0  KVM-256", "120M-KVM256-F0"
        rows[reference], = [row for row in read_rows("wave_metrics.csv") if row["run"] == reference]
    specs = [(reference, reference_label, None)] + [
        (name, label, reference) for name, label, _ in SPECS[1:]
    ]
    scores = ruler_8k_scores()
    rows[reference]["ruler_8k_pct"] = scores[reference_ruler]
    for name, _, _ in specs[1:]:
        run = RULER_RUNS[name]
        rows[name]["ruler_8k_pct"] = scores[run]
    configure_plot_style()
    filename = f"kvm-ablations-120M-vs-{reference}.png"
    ablation_heatmap(
        rows, specs, figure_title="120M KVM-256 ablations",
        boundaries=(4.5,), include_ruler_8k=True, color_limit=100 if use_f0 else None,
        output=EXPERIMENTS / "wave-results-20261111" / filename,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f0", action="store_true", help="use the F0 control as the reference")
    main(use_f0=parser.parse_args().f0)
