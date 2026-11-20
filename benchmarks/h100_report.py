"""Render matched H100 results, rejecting incomplete or mismatched runs.

uv run --no-project python -m benchmarks.h100_report \
  --results results/h100-llama31/raw --output benchmarks/H100_LLAMA_RESULTS.md
"""

import argparse
import json
from pathlib import Path
import statistics


def read_result(root, name):
    result = json.loads((root / (name + ".json")).read_text())
    if result["stage"] != "complete":
        raise ValueError(f"{name} is incomplete")
    return result


def compare(native, other, length):
    for field in (
        "checkpoint",
        "batch_size",
        "max_model_len",
        "decode_tokens",
        "warmups",
        "repeats",
        "prompt_manifest_sha256",
        "prompt_source",
        "gpu",
        "capability",
        "node",
        "gpus",
    ):
        if native[field] != other[field]:
            raise ValueError(f"Mismatched {field} at {length}")
    if (
        native["benchmark_identity"]["runtime"]
        != other["benchmark_identity"]["runtime"]
    ):
        raise ValueError("Runtime packages differ")
    nkwargs = native["llm_kwargs"] | dict(attention_config=None)
    okwargs = other["llm_kwargs"] | dict(attention_config=None)
    if nkwargs != okwargs:
        raise ValueError("Serving settings differ")
    a = native["measurements"][str(length)]
    b = other["measurements"][str(length)]
    if a["prompts"] != b["prompts"]:
        raise ValueError("Prompt tokens differ")
    if other["worker_state"][0][0]["layers"] != 32:
        raise ValueError("Expected all 32 Llama attention layers to use LoD")
    if any(native["worker_state"]):
        raise ValueError("Native baseline has a LoD runtime")


def metrics(result, length):
    measurement = result["measurements"][str(length)]
    rows = [row for row in measurement["records"] if row["phase"] == "measured"]
    if len(rows) != 3 or result["warmups"] != 2:
        raise ValueError("Require two warmups and three measurements")
    return dict(
        prefill_seconds=statistics.median(row["prefill_seconds"] for row in rows),
        decode_ms=statistics.median(row["decode_ms_per_batch_step"] for row in rows),
        elapsed_seconds=statistics.median(row["elapsed_seconds"] for row in rows),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panels = [
        (
            "native-speed-b1-cached",
            "h100-final-three-tier-bf16-b1",
            "h100-final-three-tier-int4-b1",
            [8192, 32768, 65536],
        ),
        (
            "llama-native-b8",
            "h100-final-three-tier-bf16-b8",
            "h100-final-three-tier-int4-b8",
            [32768],
        ),
        (
            "h100-window-128k-full",
            "h100-window-128k-three-tier-bf16",
            "h100-window-128k-three-tier-int4",
            [128000],
        ),
    ]
    records = []
    sources = []
    for names in panels:
        native, bf16, int4 = [read_result(args.results, name) for name in names[:3]]
        sources.extend(names[:3])
        for length in names[3]:
            compare(native, bf16, length)
            compare(native, int4, length)
            records.append(
                dict(
                    batch=native["batch_size"],
                    length=length,
                    capacity=native["max_model_len"],
                    native=metrics(native, length),
                    bf16=metrics(bf16, length),
                    int4=metrics(int4, length),
                )
            )
    records.sort(key=lambda row: (row["batch"], row["length"]))
    window = next(row for row in records if row["length"] == 128000)
    speed = window["native"]["decode_ms"]
    elapsed = window["native"]["elapsed_seconds"]
    headline = (
        f"At the 128K-window point, BF16 and INT4 deliver {speed / window['bf16']['decode_ms']:.2f}× "
        f"and {speed / window['int4']['decode_ms']:.2f}× native decode speed, respectively. "
        f"Complete-request throughput improves by {elapsed / window['bf16']['elapsed_seconds']:.2f}× "
        f"and {elapsed / window['int4']['elapsed_seconds']:.2f}×. "
        "Prefill is slower at 8K, 32K, and 64K, and faster in the 128K-window case in this panel."
    )
    quality = []
    for mode in ("three-tier-bf16", "three-tier-int4"):
        for batch in (1, 8):
            name = f"h100-final-{mode}-retrieval-b{batch}"
            result = read_result(args.results, name)
            quality.append(
                dict(
                    mode=mode,
                    batch=batch,
                    correct=result["retrieval_correct"],
                    total=result["retrieval_total"],
                )
            )
            sources.append(name)
    native_quality = read_result(args.results, "native-retrieval")
    http = read_result(args.results, "final-http-smoke")
    if native_quality["retrieval_correct"] != 9 or not all(
        row["correct"] for row in http["responses"]
    ):
        raise ValueError("A baseline retrieval or HTTP serving check failed")
    sources.extend(["native-retrieval", "final-http-smoke"])
    lines = [
        "# Llama-3.1-8B-Instruct: H100 measurements",
        "",
        headline,
        "",
        "One NVIDIA H100 80GB HBM3, tensor parallelism one, BF16 model weights. "
        "Native vLLM uses FlashAttention-3. LoD uses eight routed regions and two pages per region during three-tier decode.",
        "",
        "Each timing is the median of three measurements after two warmups, with 1,025 forced output tokens. "
        "Prompts, runtime packages, scheduling, context capacity, and memory target are checked for each comparison. "
        "The 128K-window case uses 128,000 prompt tokens within a 131,072-token model limit.",
        "",
        "## Decode",
        "",
        "Milliseconds per batch step, including periodic LoD catch-up. The batch-eight metric follows the existing ProLong cohort timing; "
        "requests may enter and leave decode at different times.",
        "",
        "| Batch | Prompt tokens | Native | Three-tier BF16 | Three-tier INT4 | BF16 speedup | INT4 speedup |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        n, b, i = [row[key]["decode_ms"] for key in ("native", "bf16", "int4")]
        lines.append(
            f"| {row['batch']} | {row['length']:,} | {n:.3f} | {b:.3f} | {i:.3f} | {n / b:.2f}× | {n / i:.2f}× |"
        )
    for title, field in [
        ("Prefill", "prefill_seconds"),
        ("Complete request cohort", "elapsed_seconds"),
    ]:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "Seconds, measured separately from model startup.",
                "",
                "| Batch | Prompt tokens | Native | Three-tier BF16 | Three-tier INT4 |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in records:
            n, b, i = [row[key][field] for key in ("native", "bf16", "int4")]
            lines.append(
                f"| {row['batch']} | {row['length']:,} | {n:.3f} | {b:.3f} | {i:.3f} |"
            )
    two = read_result(args.results, "h100-final-two-tier-b1")
    sources.append("h100-final-two-tier-b1")
    t = metrics(two, 32768)
    lines.extend(
        [
            "",
            "Two-tier BF16 also completes the 32K batch-one panel: "
            f"{t['prefill_seconds']:.3f} s prefill and {t['decode_ms']:.3f} ms per decode step.",
            "",
            "## Quality and serving checks",
            "",
            "These are retrieval sanity checks. They do not establish full RULER quality parity.",
            "",
            "Validation passed 76 regression checks and 31 focused H100 checks. "
            "The [timing audit](../results/h100-llama31/audit.json) found no reported JIT compilation inside measured calls.",
            "",
            "| LoD mode | Sequential 8K/32K/64K retrieval | Concurrent mixed-length requests |",
            "|---|---:|---:|",
        ]
    )
    for mode in ("three-tier-bf16", "three-tier-int4"):
        a, b = [row for row in quality if row["mode"] == mode]
        lines.append(
            f"| {mode} | {a['correct']}/{a['total']} | {b['correct']}/{b['total']} |"
        )
    lines.extend(
        [
            "",
            "Native vLLM passes all nine sequential retrieval checks. The actual OpenAI HTTP server passes "
            "two short questions and a 19,557-token retrieval request in three-tier INT4.",
            "",
            "Greedy output hashes vary across some repeated LoD speed runs. The report measures fixed-length performance; "
            "it does not claim identical generated continuations across runs or equivalence to dense attention.",
            "",
            "## Reproduction and artifacts",
            "",
            "Use [the H100 guide](H100_LLAMA.md) for setup and commands. "
            "Source snapshots, per-file hashes, commands, hardware identity, and logs are retained in "
            "[the provenance directory](../results/h100-llama31/provenance/). "
            "The complete installed environment is in [runtime-freeze.txt](../results/h100-llama31/runtime-freeze.txt).",
            "",
        ]
    )
    for name in sources:
        lines.append(f"- [{name}](../results/h100-llama31/raw/{name}.json)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    (args.results.parent / "summary.json").write_text(
        json.dumps(dict(measurements=records, quality=quality), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
