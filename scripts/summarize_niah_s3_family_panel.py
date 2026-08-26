#!/usr/bin/env python3
"""Consolidate the 2026-08-20 NIAH-S3 family panel artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("artifacts/niah_s3_family_panel_20260820")
LENGTHS = (8192, 16384, 32768, 65536, 131072)


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def row(name: str, length: int) -> dict:
    return load(name)["results"][str(length)]


def quality_shards(names: tuple[str, ...], length: int) -> dict:
    shards = [row(name, length)["quality"] for name in names]
    return {
        "correct": sum(int(shard["correct"]) for shard in shards),
        "total": sum(int(shard["total"]) for shard in shards),
        "sample_offsets": [int(load(name)["sample_offset"]) for name in names],
        "source_files": list(names),
    }


def paired(quality: dict, speed: dict) -> dict:
    return {"quality": quality, "speed": speed}


def regular(name: str, length: int) -> dict:
    result = row(name, length)
    return paired(result["quality"], result["speed"])


def unavailable(reason: str) -> dict:
    return {"unavailable": reason}


def fmt_quality(value: dict) -> str:
    if "unavailable" in value:
        return "—"
    quality = value["quality"]
    return f'{quality["correct"]}/{quality["total"]}'


def fmt_tps(value: dict) -> str:
    if "unavailable" in value:
        return "—"
    return f'{value["speed"]["prefill_prompt_tokens_per_second"]:,.0f}'


def fmt_ms(value: dict) -> str:
    if "unavailable" in value:
        return "—"
    return f'{value["speed"]["marginal_decode_ms_per_batch_step"]:.1f}'


def main() -> None:
    models: dict[str, dict] = {
        "Gemma-4-26B-A4B-it": {
            "checkpoint": "google/gemma-4-26B-A4B-it",
            "official_context": 262144,
            "tensor_parallel_size": 1,
            "results": {},
        },
        "Qwen3.8-27B-FP8": {
            "checkpoint": "Qwen/Qwen3.8-27B-FP8",
            "official_context": 262144,
            "tensor_parallel_size": 1,
            "results": {},
        },
        "Muse-Glimmer-30B": {
            "checkpoint": "meta-models/Muse-Glimmer-30B",
            "official_context": 131072,
            "tensor_parallel_size": 1,
            "results": {},
        },
        "OLMo-3-1125-32B": {
            "checkpoint": "allenai/Olmo-3-1125-32B",
            "official_context": 65536,
            "tensor_parallel_size": 1,
            "results": {},
        },
        "Phi-4": {
            "checkpoint": "microsoft/phi-4",
            "official_context": 16384,
            "tensor_parallel_size": 5,
            "results": {},
        },
    }

    for length in LENGTHS:
        models["Gemma-4-26B-A4B-it"]["results"][str(length)] = {
            "full": regular("gemma4_full.json", length),
            "lod": regular("gemma4_lod.json", length),
        }
        models["Qwen3.8-27B-FP8"]["results"][str(length)] = {
            "full": regular("qwen38_full.json", length),
            "lod": regular("qwen38_lod.json", length),
        }

    for length in LENGTHS[:-1]:
        models["Muse-Glimmer-30B"]["results"][str(length)] = {
            "full": regular("muse_glimmer_full.json", length),
            "lod": regular("muse_glimmer_lod.json", length),
        }
    models["Muse-Glimmer-30B"]["results"]["131072"] = {
        "full": paired(
            quality_shards(
                (
                    "muse_glimmer_full_128k_q0.json",
                    "muse_glimmer_full_128k_q32.json",
                ),
                131072,
            ),
            row("muse_glimmer_full_128k_speed.json", 131072)["speed"],
        ),
        "lod": paired(
            quality_shards(
                (
                    "muse_glimmer_lod_128k_q0.json",
                    "muse_glimmer_lod_128k_q32.json",
                ),
                131072,
            ),
            row("muse_glimmer_lod_128k_speed.json", 131072)["speed"],
        ),
    }

    for length in LENGTHS[:3]:
        models["OLMo-3-1125-32B"]["results"][str(length)] = {
            "full": regular("olmo3_full.json", length),
            "lod": regular("olmo3_lod.json", length),
        }
    models["OLMo-3-1125-32B"]["results"]["65536"] = {
        "full": regular("olmo3_full_64k.json", 65536),
        "lod": paired(
            quality_shards(
                ("olmo3_lod_64k_q0.json", "olmo3_lod_64k_q32.json"),
                65536,
            ),
            row("olmo3_lod_64k_speed.json", 65536)["speed"],
        ),
    }
    olmo_128_reason = (
        "Checkpoint advertises 65,536 tokens; both full and LOD reproducibly "
        "faulted when forced to extrapolate to 131,072."
    )
    models["OLMo-3-1125-32B"]["results"]["131072"] = {
        "full": unavailable(olmo_128_reason),
        "lod": unavailable(olmo_128_reason),
    }

    for length in LENGTHS[:-1]:
        models["Phi-4"]["results"][str(length)] = {
            "full": regular("phi4_full_tp5.json", length),
            "lod": regular("phi4_lod_tp5.json", length),
        }
    phi_128_reason = (
        "Checkpoint advertises 16,384 tokens; both full and LOD reproducibly "
        "faulted at 81,920 tokens when forced toward 131,072."
    )
    models["Phi-4"]["results"]["131072"] = {
        "full": unavailable(phi_128_reason),
        "lod": unavailable(phi_128_reason),
    }

    consolidated = {
        "date": "2026-08-20",
        "hardware": "AMD Instinct MI325X (gfx942)",
        "batch_size": 8,
        "quality_samples_per_length": 64,
        "quality_decode_tokens": 64,
        "speed_warmups": 1,
        "speed_repeats": 3,
        "max_num_batched_tokens": 16384,
        "lod": {
            "levels": 2,
            "state_factor": 16,
            "open_routes": 8,
            "kv_dtype": "bfloat16",
            "routing_geometry": "auto (spherical/coherence-aware where selected)",
            "dense_leaf_storage": True,
        },
        "models": models,
    }
    (ROOT / "panel.json").write_text(
        json.dumps(consolidated, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# NIAH-S3 family panel (batch 8)",
        "",
        "Each quality cell is `Full / LOD` correct out of 64. Speed uses one "
        "warmup and the median of three runs with a 16,384-token vLLM scheduler "
        "budget; prefill is aggregate prompt tokens/s and decode is milliseconds "
        "per batch-8 step (lower is better).",
        "",
        "## Quality",
        "",
        "| Model | 8K | 16K | 32K | 64K | 128K |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, data in models.items():
        values = []
        for length in LENGTHS:
            result = data["results"][str(length)]
            values.append(
                f'{fmt_quality(result["full"])} / {fmt_quality(result["lod"])}'
            )
        lines.append(f'| {model} | ' + " | ".join(values) + " |")

    lines += [
        "",
        "## Prefill throughput",
        "",
        "Cells are `Full / LOD (LOD speedup)` aggregate prompt tok/s.",
        "",
        "| Model | 8K | 16K | 32K | 64K | 128K |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, data in models.items():
        values = []
        for length in LENGTHS:
            result = data["results"][str(length)]
            if "unavailable" in result["full"] or "unavailable" in result["lod"]:
                values.append("—")
                continue
            full = result["full"]["speed"]["prefill_prompt_tokens_per_second"]
            lod = result["lod"]["speed"]["prefill_prompt_tokens_per_second"]
            values.append(f'{fmt_tps(result["full"])} / {fmt_tps(result["lod"])} ({lod / full:.2f}×)')
        lines.append(f'| {model} | ' + " | ".join(values) + " |")

    lines += [
        "",
        "## Decode latency",
        "",
        "Cells are `Full / LOD (LOD speedup)` ms per batch-8 step.",
        "",
        "| Model | 8K | 16K | 32K | 64K | 128K |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, data in models.items():
        values = []
        for length in LENGTHS:
            result = data["results"][str(length)]
            if "unavailable" in result["full"] or "unavailable" in result["lod"]:
                values.append("—")
                continue
            full = result["full"]["speed"]["marginal_decode_ms_per_batch_step"]
            lod = result["lod"]["speed"]["marginal_decode_ms_per_batch_step"]
            values.append(f'{fmt_ms(result["full"])} / {fmt_ms(result["lod"])} ({full / lod:.2f}×)')
        lines.append(f'| {model} | ' + " | ".join(values) + " |")

    lines += [
        "",
        "## Notes",
        "",
        "- NIAH-S3 uses greedy 64-token generation, proper chat templates for "
        "instruction checkpoints, and raw prompting for base OLMo.",
        "- LOD is the current two-tier BF16 design: state factor 16, top-8 routes, "
        "dense leaf storage, and automatic spherical/coherence-aware routing.",
        "- Qwen weights use the requested FP8 checkpoint; activations and LOD state "
        "are BF16. Other checkpoints use BF16 weights/state.",
        "- Phi-4 uses TP=5 for both modes because it has 10 KV heads. Other models "
        "use one GPU. Ratios are paired within a model; absolute Phi throughput is "
        "not cross-model comparable.",
        "- At advertised context boundaries, the speed prompt reserves the 64 decode "
        "positions: 65,472 input tokens for OLMo 64K and 131,008 for Muse 128K.",
        "- Gemma full attention uses Triton because AITER cannot support its 512-wide "
        "global heads. Other reported full rows use AITER.",
        "- OLMo 128K and Phi 128K are unavailable for both modes after reproducible "
        "forced-extrapolation faults; no scores or timings are imputed.",
        "",
        "Machine-readable consolidated data is in `panel.json`; source JSON files "
        "in this directory retain per-example responses and all timing repetitions.",
        "",
    ]
    (ROOT / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
