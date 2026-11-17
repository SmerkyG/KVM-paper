"""Plot Q0, Q1, and the 350M GPTA-2 references."""

import csv
import sys
from pathlib import Path


OUTPUT = Path(__file__).resolve().parent
EXPERIMENTS = OUTPUT.parent
sys.path.insert(0, str(EXPERIMENTS / "hybrid-results-350M-20261111"))
from plot_hybrids import BENCHMARK_KEYS, evaluation_metrics, load_json
from plot_report import LENGTHS, benchmark_bars, configure_plot_style, plt, retrieval


QWEN = EXPERIMENTS / "qwen-350M-lr-2p52982e-3-20261112"
K0 = EXPERIMENTS / "kvm-sqrt-triton-350M-20261111"
PAPER = "https://arxiv.org/html/2605.09877v5"
RUNS = [
    ("Q0", "Q0 full · 2.53e-3"),
    ("Q1", "Q1 Triton KVM-sqrt · 2.53e-3"),
    ("GPTA-2", "GPTA-2 full · paper"),
    ("K0", "GPTA-2 Triton KVM-sqrt · K0"),
    ("GPTA-2 KVM-sqrt paper", "GPTA-2 KVM-sqrt · paper"),
]
BACKBONE_RUNS = [
    ("Q0", "Qwen3-style full"),
    ("Q1", "Qwen3-style KVM-sqrt"),
    ("GPTA-2", "GPTA-2 full"),
    ("K0", "GPTA-2 KVM-sqrt"),
]


def collect():
    rows = {}
    protocol = load_json(K0 / "eval/manifest.json")["protocol"]
    for name, directory in (("Q0", QWEN / "Q0"), ("Q1", QWEN / "Q1"), ("K0", K0)):
        evaluation = directory / "eval"
        assert load_json(evaluation / "manifest.json")["protocol"] == protocol, name
        scores = evaluation_metrics(evaluation / "status.json", name)
        source = str(evaluation / "status.json")
        textbook_source = source if "textbook_32k_nll" in scores else ""
        retry = directory / "eval-textbook-retry/status.json"
        if retry.is_file() and "textbook_32k_nll" not in scores:
            recovered = evaluation_metrics(retry, name)
            if "textbook_32k_nll" in recovered:
                scores["textbook_32k_nll"] = recovered["textbook_32k_nll"]
                textbook_source = str(retry)
        assert set(BENCHMARK_KEYS) - {"textbook_32k_nll"} <= scores.keys(), name
        pipeline = load_json(directory / "pipeline_status.json")
        rows[name] = {
            "run": name, "dclm_val_nll": pipeline["final_validation_loss"],
            **scores, "benchmark_source": source, "textbook_source": textbook_source,
        }

    # Published 350M GPTA-2 full attention, Tables 2 and 4.
    niah = (
        (100.0, 100.0, 51.8, 22.4),
        (100.0, 99.8, 45.2, 20.8),
        (98.2, 72.2, 37.6, 5.8),
    )
    rows["GPTA-2"] = {
        "run": "GPTA-2",
        **{f"s{task}_{length}_pct": score
           for task, scores in enumerate(niah, 1)
           for length, score in zip(LENGTHS, scores)},
        "ruler_4k_pct": 47.2, "longbench_pct": 25.1, "short_mean_pct": 47.2,
        "benchmark_source": f"{PAPER}#S4.T2 | {PAPER}#A5.T4",
    }
    published = EXPERIMENTS / "hybrid-results-1-to-1/published_1_to_1_metrics.csv"
    with published.open() as file:
        kvm_sqrt = next(
            row for row in csv.DictReader(file)
            if row["scale"] == "350M" and row["model"] == "KVM sqrt"
        )
    rows["GPTA-2 KVM-sqrt paper"] = {
        "run": "GPTA-2 KVM-sqrt paper",
        **{key: float(kvm_sqrt[key]) for key in BENCHMARK_KEYS if key.endswith("_pct")},
        "benchmark_source": rows["GPTA-2"]["benchmark_source"],
    }
    return rows


def plot(rows, runs, filename, *, title=None, font_scale=1):
    configure_plot_style()
    for setting in ("font.size", "axes.titlesize", "legend.fontsize"):
        plt.rcParams[setting] *= font_scale
    extra_height = 1.5 * (font_scale - 1)
    fig = plt.figure(figsize=(10.8, 6.6 + extra_height), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=(0.15 + 0.14 * (font_scale - 1), 1, 1))
    legend = fig.add_subplot(grid[0, :])
    legend.axis("off")
    axes = [fig.add_subplot(grid[y, x]) for y in (1, 2) for x in (0, 1)]
    if title:
        fig.suptitle(title, fontsize=15 * font_scale, fontweight="semibold")
    for task, ax in enumerate(axes[:3], 1):
        retrieval(ax, rows, runs, task, f"NIAH-S{task}", x_label=task == 3)
    benchmark_bars(axes[3], rows, runs)
    axes[3].set_ylim(0, 55)
    for label in axes[3].texts:
        label.set_fontsize(7 * font_scale)
        if font_scale > 1:
            label.set(rotation=90, verticalalignment="top", position=(0, -4), color="white")
    handles, labels = axes[0].get_legend_handles_labels()
    legend.legend(handles, labels, ncols=len(runs), loc="center")
    with plt.rc_context({"pdf.fonttype": 42}):
        fig.savefig(OUTPUT / filename, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    rows = collect()
    with (OUTPUT / "qwen_metrics_350M.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "run", "dclm_val_nll", *BENCHMARK_KEYS, "benchmark_source", "textbook_source",
        ])
        writer.writeheader()
        writer.writerows(rows[name] for name, _ in RUNS)
    plot(rows, RUNS, "qwen-controls-350M.png", title="350M Qwen3-style and GPTA-2 backbones")
    plot(rows, BACKBONE_RUNS, "qwen-gpta2-backbones-350M.png")
    plot(rows, BACKBONE_RUNS, "qwen-gpta2-backbones-350M.pdf")
    plot(rows, BACKBONE_RUNS, "qwen-gpta2-backbones-350M-150pct-text.pdf", font_scale=1.5)
