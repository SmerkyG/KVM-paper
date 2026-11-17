"""Render the batch-32 full-model prefill, decode, and memory sweep."""

import argparse
import csv
import json
import math
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


CONTEXTS = (512, 1024, 2048, 4096, 8192, 16384, 32768)
SERIES = (
    ("fixed256", "KVM-256", "#009E73", "s", "-."),
    ("sqrt", r"KVM-sqrt", "#0072B2", "o", "-"),
    ("gpta2", "GPTA-2", "#D55E00", "^", "--"),
)
CHUNKED_GPTA2 = ("gpta2_chunked", "GPTA-2 (2K chunks)", "#AA4499", "D", ":")
SERIES_2K = tuple(
    (f"{model}_chunked_2048" if model != "gpta2" else "gpta2_chunked", label, color, marker, style)
    for model, label, color, marker, style in SERIES
)


def read_rows(directory, series=SERIES, *, sources=None):
    rows = {}
    environment = set()
    horizons = set()
    sources = sources or {}
    for model, *_ in series:
        for length in CONTEXTS:
            path = sources.get(model, directory) / "raw" / f"{model}-{length}.json"
            data = json.loads(path.read_text())
            if (data["model"], data["context"], data["batch_size"], data["mode"]) != (
                model, length, 32, "graph"
            ):
                raise ValueError(f"Unexpected workload in {path}")
            samples = data["samples"]
            if len(samples) != data["rounds"] or data["rounds"] != 7 or any(
                not math.isfinite(sample[key]) or sample[key] <= 0
                for sample in samples for key in ("prefill_ms", "decode_ms", "wall_ms")
            ):
                raise ValueError(f"Invalid timing samples in {path}")
            environment.add(tuple(data[key] for key in ("gpu", "torch", "rocm", "dtype")))
            prefill = data["prefill_ms"]
            decode = data["decode_ms"]
            steps = data["new_tokens"] - 1
            if steps <= 0:
                raise ValueError(f"Invalid decode horizon in {path}")
            horizons.add(steps)
            chunk_check = data.get("unchunked_prefill_check") or {}
            rows[model, length] = {
                "model": model, "context_tokens": length, "decode_steps": steps,
                "prefill_chunk_size": data.get("prefill_chunk_size"),
                "unchunked_prefill_token_differences": chunk_check.get("first_token_mismatches"),
                "unchunked_prefill_logit_differences": chunk_check.get("first_logit_mismatches"),
                "state_updates_per_layer": data.get("state_updates_per_layer"),
                "decode_graphs": data.get("decode_graphs", 1),
                "prefill_median_ms": prefill["median"],
                "prefill_min_ms": prefill["min"],
                "prefill_max_ms": prefill["max"],
                "decode_median_ms_per_step": decode["median"] / steps,
                "decode_min_ms_per_step": decode["min"] / steps,
                "decode_max_ms_per_step": decode["max"] / steps,
                "end_to_end_median_ms": data["wall_ms"]["median"],
                "peak_reserved_gib": data["peak_reserved_bytes"] / 2**30,
                "setup_peak_reserved_gib": data["setup_peak_reserved_bytes"] / 2**30,
                "replays_with_token_differences": sum(
                    check["greedy_token_mismatches"] > 0 for check in data["output_checks"]
                ),
                "replays_with_logit_differences": sum(
                    check["first_logit_mismatches"] + check["last_logit_mismatches"] > 0
                    for check in data["output_checks"]
                ),
                "checkpoint": data["checkpoint"],
                "timing_source": str(path),
            }
    if len(environment) != 1:
        raise ValueError(f"Inconsistent benchmark environments: {environment}")
    if len(horizons) != 1:
        raise ValueError(f"Inconsistent decode horizons: {horizons}")
    if ("gpta2", CONTEXTS[0]) in rows and ("gpta2_chunked", CONTEXTS[0]) in rows:
        for length in CONTEXTS:
            chunked = rows["gpta2_chunked", length]
            if chunked["prefill_chunk_size"] != 2048 or chunked["checkpoint"] != rows["gpta2", length]["checkpoint"]:
                raise ValueError(f"Expected the same GPTA-2 checkpoint and 2K chunks at {length}")
    return rows, horizons.pop()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows.values() for key in row)))
        writer.writeheader()
        writer.writerows(rows.values())


def draw_panels(rows, panels, output, series, *, heading=None, font_scale=1):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10 * font_scale,
        "axes.titleweight": "semibold",
        "axes.labelcolor": "#20262D",
        "axes.edgecolor": "#7A828D",
        "xtick.color": "#3D4652",
        "ytick.color": "#3D4652",
        "pdf.fonttype": 42,
    })
    panel_count = len(panels)
    left = 0.055 if panel_count == 4 else 0.070
    width = 13.7 if panel_count == 4 else 10.4
    height = (3.45 if heading else 3.1) + 1.5 * (font_scale - 1)
    fig, axes = plt.subplots(1, panel_count, figsize=(width, height))
    fig.subplots_adjust(left=left + 0.023 * (font_scale - 1), right=0.984,
                        top=(0.695 if heading else 0.785) - 0.096 * (font_scale - 1),
                        bottom=(0.166 if heading else 0.18) + 0.055 * (font_scale - 1),
                        wspace=0.38 + 0.14 * (font_scale - 1))
    if heading:
        title, subtitle = heading
        fig.text(left, 0.937, title, fontsize=15 * font_scale, fontweight="semibold", color="#20262D")
        fig.text(left, 0.874, subtitle, fontsize=9.5 * font_scale, color="#4A5563")
    x = [length / 1024 for length in CONTEXTS]
    for ax, (title, ylabel, metric, scale) in zip(axes, panels):
        for name, label, color, marker, linestyle in series:
            points = [rows[name, length] for length in CONTEXTS]
            ax.plot(x, [row[metric] * scale for row in points], label=label, color=color,
                    marker=marker, linestyle=linestyle, linewidth=1.7, markersize=4.4,
                    markerfacecolor="none" if name == "gpta2_chunked" else color)
        if font_scale > 1:
            title = fill(title, width=14)
        ax.set_title(title, loc="left", color="#20262D", fontsize=11 * font_scale, pad=7)
        ax.set_xlabel("Prompt length\n(K tokens)" if font_scale > 1 else "Prompt length (K tokens)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.45, 33.2)
        ax.set_ylim(bottom=0)
        ax.set_xticks((0, 8, 16, 24, 32))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis="y", color="#D9DFE6", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=9 * font_scale, length=3.5)
    handles, labels = axes[0].get_legend_handles_labels()
    legend_top = 0.848 if heading else 0.965 + 0.025 * (font_scale - 1)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(left - 0.009, legend_top),
               ncols=len(series), frameon=False, handlelength=2.4, columnspacing=1.8)
    formats = (output.suffix,) if output.suffix else (".png", ".pdf")
    for extension in formats:
        fig.savefig(output.with_suffix(extension), dpi=400, facecolor="white")
    plt.close(fig)


def read_memory_rows(directory, timing_rows, steps, *, sources=None):
    rows = {}
    sources = sources or {}
    for (model, length), timing in timing_rows.items():
        path = sources.get(model, directory) / "memory-raw" / f"{model}-{length}.json"
        data = json.loads(path.read_text())
        if (data["model"], data["context"], data["batch_size"], data["new_tokens"], data["checkpoint"]) != (
            model, length, 32, steps + 1, timing["checkpoint"]
        ):
            raise ValueError(f"Mismatched memory benchmark in {path}")
        if data.get("prefill_chunk_size") != timing["prefill_chunk_size"]:
            raise ValueError(f"Mismatched prefill chunk size in {path}")
        rows[model, length] = {
            "model": model, "context_tokens": length, "decode_steps": steps,
            "prefill_chunk_size": timing["prefill_chunk_size"],
            "memory_source": str(path),
            **{
                key.removesuffix("_bytes") + "_gib": value / 2**30
                for key, value in data.items() if key.endswith("_bytes")
            },
            "decode_cache_in_graph_pool_gib": data["decode_cache_bytes_in_graph_pool"] / 2**30,
            "prefill_graph_cache_in_pool_gib": data["prefill_graph_cache_bytes_in_pool"] / 2**30,
            "graph_pool_added_by_decode_gib": (
                data["graph_pool_reserved_bytes"] - data["prefill_graph_pool_reserved_bytes"]
            ) / 2**30,
        }
    return rows


def render(directory, memory_breakdown=False, chunked=False, kvm_2k_sweep=None):
    all_chunked = kvm_2k_sweep is not None
    sources = {}
    if all_chunked:
        series = SERIES_2K
        sources = {name: kvm_2k_sweep for name, *_ in series if name != "gpta2_chunked"}
    else:
        series = (*SERIES, CHUNKED_GPTA2) if chunked else SERIES
    rows, steps = read_rows(directory, series, sources=sources)
    if all_chunked and (steps != 256 or any(row["prefill_chunk_size"] != 2048 for row in rows.values())):
        raise ValueError("The 2K sweep requires 2048-token prefill chunks and 256 decode steps")
    peak_memory_key = "peak_reserved_gib"
    if memory_breakdown:
        memory = read_memory_rows(directory, rows, steps, sources=sources)
        peak_memory_key = "replay_peak_reserved_gib"
        for key, row in rows.items():
            row["decode_cache_gib"] = memory[key]["decode_cache_gib"]
            row[peak_memory_key] = memory[key].get(peak_memory_key, memory[key]["replay_total_reserved_gib"])
            row["memory_source"] = memory[key]["memory_source"]
    panels = (
        ("Prefill", "Time to first token (s)", "prefill_median_ms", 0.001),
        ("Decode", "Latency (ms / step)", "decode_median_ms_per_step", 1.0),
        ("Peak GPU memory", "PyTorch reserved (GiB)", peak_memory_key, 1.0),
    )
    if memory_breakdown:
        panels += (("Resident decode cache", "Tensor storage (GiB)", "decode_cache_gib", 1.0),)
    write_csv(directory / "batch32-summary.csv", rows)
    draw_panels(rows, panels, directory / "batch32-graph-inference", series)
    if memory_breakdown:
        write_csv(directory / "batch32-memory.csv", memory)
        panels = (
            ("Resident decode cache", "Tensor storage (GiB)", "decode_cache_gib", 1.0),
            ("Eager prefill peak", "PyTorch allocated (GiB)", "eager_prefill_peak_allocated_gib", 1.0),
            ("CUDA Graph pool", "Reserved excluding cache (GiB)", "graph_pool_reserved_minus_cache_gib", 1.0),
        )
        chunks = "2K prefill chunks  ·  " if all_chunked else ""
        subtitle = f"Batch 32  ·  {chunks}{steps} decode steps  ·  BF16  ·  AMD Instinct MI325X"
        draw_panels(memory, panels, directory / "batch32-memory-breakdown", series,
                    heading=("350M inference memory by phase", subtitle))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).with_name("figure5-b32"))
    parser.add_argument("--memory-breakdown", action="store_true")
    parser.add_argument("--chunked-gpta2", action="store_true")
    parser.add_argument("--kvm-2k-sweep", type=Path, help="plot the 2K prefill sweep for all three models")
    args = parser.parse_args()
    if args.chunked_gpta2 and args.kvm_2k_sweep is not None:
        parser.error("Choose either the GPTA-2 comparison or the all-model 2K sweep")
    render(args.directory, args.memory_breakdown, args.chunked_gpta2, args.kvm_2k_sweep)
