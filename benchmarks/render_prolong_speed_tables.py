#!/usr/bin/env python3
"""Render every current ProLong speed table as paper-ready LaTeX."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


SPEED_SECTION = "## Current routed top-8 speed results"
SECTION_END = "## Reproduce prompt quality"
MODES = ("Full", "Two-tier BF16", "Three-tier BF16", "Three-tier INT4")
END_TO_END_RE = re.compile(r"([0-9.]+) s / ([0-9.]+) ms")
VERIFICATION_RE = re.compile(r"([0-9.]+) ms / ([0-9.]+)")
GEOMETRY_RE = re.compile(r"TP([0-9]+), batch ([0-9]+)")


@dataclass(frozen=True)
class SpeedTable:
    heading: str
    kind: str
    rows: tuple[tuple[str, ...], ...]


def _split_row(line: str) -> tuple[str, ...]:
    return tuple(cell.strip() for cell in line.strip().strip("|").split("|"))


def load_speed_tables(path: Path) -> list[SpeedTable]:
    lines = path.read_text().splitlines()
    try:
        index = lines.index(SPEED_SECTION) + 1
    except ValueError as error:
        raise ValueError(f"Missing speed section in {path}") from error

    heading: str | None = None
    pending_kind: str | None = None
    tables: list[SpeedTable] = []
    while index < len(lines) and lines[index] != SECTION_END:
        line = lines[index]
        if line.startswith("### "):
            heading = line.removeprefix("### ")
        elif line == "End-to-end wall time:":
            pending_kind = "end-to-end"
        elif line.startswith("Attention core (real minus dummy)"):
            pending_kind = "attention-core"
        elif line.startswith("Each next cell is"):
            pending_kind = "verification"
        elif line.startswith("| Context") and pending_kind is not None:
            if heading is None:
                raise ValueError("Speed table appears before a configuration heading")
            headers = _split_row(line)
            if headers != ("Context", *MODES):
                raise ValueError(f"Unexpected speed-table headers: {headers}")
            index += 2
            rows: list[tuple[str, ...]] = []
            while index < len(lines) and lines[index].startswith("|"):
                row = _split_row(lines[index])
                if len(row) != len(headers) or not re.fullmatch(r"[0-9]+K", row[0]):
                    break
                rows.append(row)
                index += 1
            tables.append(SpeedTable(heading, pending_kind, tuple(rows)))
            pending_kind = None
            continue
        index += 1

    if not tables:
        raise ValueError(f"No speed tables found in {path}")
    return tables


def _latex_cell(cell: str, kind: str) -> str:
    if cell == "Does not fit B8":
        return r"\textemdash"
    pattern = VERIFICATION_RE if kind == "verification" else END_TO_END_RE
    match = pattern.fullmatch(cell)
    if match is None:
        raise ValueError(f"Cannot parse {kind} cell: {cell!r}")
    first, second = match.groups()
    if kind == "verification":
        return rf"\({first}\,\mathrm{{ms}}\,/\,{second}\)"
    return rf"\({first}\,\mathrm{{s}}\,/\,{second}\,\mathrm{{ms}}\)"


def _caption(table: SpeedTable) -> str:
    prefix = f"{table.heading}: "
    if table.kind == "end-to-end":
        decode_unit = (
            "milliseconds per output token"
            if "DFlash2" in table.heading
            else "milliseconds per batch step"
        )
        return (
            prefix
            + "end-to-end ProLong timing. Each cell reports prefill seconds / "
            + f"decode {decode_unit}."
        )
    if table.kind == "attention-core":
        return (
            prefix
            + "attention-subsystem timing from matched real-minus-dummy runs. "
            + "Each cell reports prefill seconds / decode milliseconds per batch step."
        )
    return (
        prefix
        + "DFlash2 verification behavior. Each cell reports target verification-cycle "
        + "milliseconds / pooled mean output tokens per cycle."
    )


def _label(table: SpeedTable) -> str:
    model = "qwen38-dflash2" if "DFlash2" in table.heading else (
        "qwen38" if table.heading.startswith("Qwen3.8") else "k2-horizon"
    )
    geometry = GEOMETRY_RE.search(table.heading)
    if geometry is None:
        raise ValueError(f"Cannot parse configuration geometry: {table.heading!r}")
    tp, batch = geometry.groups()
    return f"tab:prolong-{model}-tp{tp}-b{batch}-{table.kind}"


def render_table(table: SpeedTable) -> str:
    rows = [
        " & ".join((row[0], *(_latex_cell(cell, table.kind) for cell in row[1:])))
        + r" \\"
        for row in table.rows
    ]
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            rf"\caption{{{_caption(table)}}}",
            rf"\label{{{_label(table)}}}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            "Context & " + " & ".join(MODES) + r" \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )


def render_document(tables: list[SpeedTable], source: Path) -> str:
    preamble = [
        f"% Generated from benchmarks/{source.name}.",
        r"% Requires \usepackage{booktabs}.",
        "% Cells preserve the units documented in the source benchmark tables.",
        "",
    ]
    return "\n".join(preamble) + "\n\n".join(render_table(table) for table in tables) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("PROLONG.md"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("PROLONG_SPEED_TABLES.tex"),
    )
    args = parser.parse_args()
    tables = load_speed_tables(args.input)
    args.output.write_text(render_document(tables, args.input))
    print(f"Wrote {len(tables)} tables to {args.output}")


if __name__ == "__main__":
    main()
