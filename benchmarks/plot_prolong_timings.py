#!/usr/bin/env python3
"""Plot the current Qwen timing panels from benchmarks/PROLONG.md."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONTEXT_RE = re.compile(r"^(\d+)K$")
HEADING_RE = re.compile(r"^### Qwen3\.8, TP(\d+), batch (\d+), top-8$")
CELL_RE = re.compile(r"([0-9.]+) s / ([0-9.]+) ms")
MODES = ("Full", "Two-tier BF16", "Three-tier BF16", "Three-tier INT4")
COLORS = {
    "Full": "#222222",
    "Two-tier BF16": "#0072B2",
    "Three-tier BF16": "#009E73",
    "Three-tier INT4": "#D55E00",
}
MARKERS = {
    "Full": "o",
    "Two-tier BF16": "s",
    "Three-tier BF16": "^",
    "Three-tier INT4": "D",
}


@dataclass(frozen=True)
class Timing:
    prefill_seconds: float
    decode_ms: float


Panel = dict[str, dict[str, Timing]]


def _parse_table(lines: list[str], start: int) -> tuple[Panel, int]:
    panel: Panel = {}
    index = start
    while index < len(lines) and not lines[index].startswith("| Context"):
        index += 1
    index += 2  # Skip the header and Markdown separator.
    while index < len(lines) and lines[index].startswith("|"):
        cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
        context_match = CONTEXT_RE.match(cells[0])
        if context_match is None or len(cells) != len(MODES) + 1:
            break
        row: dict[str, Timing] = {}
        for mode, cell in zip(MODES, cells[1:], strict=True):
            timing_match = CELL_RE.fullmatch(cell)
            if timing_match is None:
                raise ValueError(f"Cannot parse timing cell: {cell!r}")
            row[mode] = Timing(
                prefill_seconds=float(timing_match.group(1)),
                decode_ms=float(timing_match.group(2)),
            )
        panel[cells[0]] = row
        index += 1
    return panel, index


def load_qwen_panels(path: Path) -> dict[str, dict[str, Panel]]:
    lines = path.read_text().splitlines()
    panels: dict[str, dict[str, Panel]] = {}
    index = 0
    while index < len(lines):
        heading = HEADING_RE.match(lines[index])
        if heading is None:
            index += 1
            continue
        label = f"TP{heading.group(1)} / B{heading.group(2)}"
        end_to_end, index = _parse_table(lines, index + 1)
        while index < len(lines) and not lines[index].startswith(
            "Attention core (real minus dummy)"
        ):
            index += 1
        attention_core, index = _parse_table(lines, index + 1)
        panels[label] = {"end_to_end": end_to_end, "attention_core": attention_core}
    if not panels:
        raise ValueError(f"No current Qwen top-8 timing panels found in {path}")
    return panels


def _plot_measurement(
    panels: dict[str, dict[str, Panel]],
    measurement: str,
    output_path: Path,
) -> None:
    setup_order = ("TP1 / B1", "TP1 / B8", "TP4 / B8")
    missing = set(setup_order) - panels.keys()
    if missing:
        raise ValueError(f"Missing Qwen timing panels: {sorted(missing)}")

    figure, axes = plt.subplots(
        2,
        len(setup_order),
        figsize=(14.2, 7.7),
        sharex=True,
        sharey="row",
        constrained_layout=False,
    )
    for column, setup in enumerate(setup_order):
        panel = panels[setup][measurement]
        contexts = list(panel)
        x_values = range(len(contexts))
        for mode in MODES:
            style = {
                "color": COLORS[mode],
                "marker": MARKERS[mode],
                "linewidth": 2.0,
                "markersize": 5.5,
                "label": mode,
            }
            axes[0, column].plot(
                x_values,
                [panel[context][mode].prefill_seconds for context in contexts],
                **style,
            )
            axes[1, column].plot(
                x_values,
                [panel[context][mode].decode_ms for context in contexts],
                **style,
            )
        axes[0, column].set_title(setup, fontsize=12, fontweight="bold")
        axes[0, column].set_yscale("log")
        axes[0, column].grid(True, which="both", alpha=0.25)
        axes[1, column].grid(True, which="both", alpha=0.25)
        axes[1, column].set_xticks(list(x_values), contexts)
        axes[1, column].set_xlabel("Context length")

    axes[0, 0].set_ylabel("Prefill wall time (s, log scale)")
    axes[1, 0].set_ylabel("Decode latency (ms / batch step)")
    title = (
        "Qwen3.8-27B-FP8 end-to-end timings"
        if measurement == "end_to_end"
        else "Qwen3.8-27B-FP8 inferred attention-core timings"
    )
    figure.suptitle(title, fontsize=15, fontweight="bold", y=0.985)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=4,
        frameon=False,
    )
    bottom = 0.10 if measurement == "end_to_end" else 0.14
    if measurement == "attention_core":
        figure.text(
            0.5,
            0.025,
            "Attention core is inferred by subtracting a full-mode dummy-attention run. "
            "Different cache-maintenance scaling can bias the residual, especially below 1 ms.",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#555555",
        )
    figure.subplots_adjust(top=0.86, bottom=bottom, left=0.075, right=0.985, hspace=0.18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    figure.savefig(output_path.with_suffix(".pdf"))
    figure.savefig(output_path.with_suffix(".svg"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("PROLONG.md"),
        help="PROLONG.md containing the timing tables",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("figures"),
        help="Directory for PNG and SVG outputs",
    )
    args = parser.parse_args()

    panels = load_qwen_panels(args.input)
    _plot_measurement(
        panels,
        "end_to_end",
        args.output_dir / "qwen_top8_end_to_end.png",
    )
    _plot_measurement(
        panels,
        "attention_core",
        args.output_dir / "qwen_top8_attention_core.png",
    )


if __name__ == "__main__":
    main()
