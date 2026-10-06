"""Prepare a node-local, genuinely truncated trained K3 checkpoint.

Copy only shards needed by the retained language layers and non-layer weights.
Do not slice a loaded full model or reuse its IPC storage. This is a speed
fixture, not a quality checkpoint or a two-node pipeline simulation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shutil
import struct


LAYER = re.compile(r"^(?:language_model\.)?model\.layers\.(\d+)\.")


def stage_config(original: dict, layers: int) -> dict:
    config = deepcopy(original)
    text = config.get("text_config", config)
    total = int(text["num_hidden_layers"])
    block = int(text.get("attn_res_block_size", 12))
    if not 0 < layers < total or layers % block:
        raise ValueError("retain a proper prefix ending at an AttnRes block boundary")
    text["num_hidden_layers"] = layers
    text["num_nextn_predict_layers"] = 0
    linear = text.get("linear_attn_config", {})
    for name in ("full_attn_layers", "kda_layers"):
        if name in linear:
            # K3 config uses one-based layer indices.
            linear[name] = [i for i in linear[name] if i <= layers]
    return config


def retained_weight(name: str, layers: int) -> bool:
    match = LAYER.match(name)
    return match is None or int(match[1]) < layers


def selected_weight_map(original: dict, layers: int) -> dict[str, str]:
    selected = {name: shard for name, shard in original["weight_map"].items()
                if retained_weight(name, layers)}
    indices = {int(match[1]) for name in selected if (match := LAYER.match(name))}
    if indices != set(range(layers)):
        raise ValueError("checkpoint index does not contain the complete retained prefix")
    if any(Path(shard).name != shard for shard in selected.values()):
        raise ValueError("checkpoint shards must be simple relative filenames")
    return selected


def prepare(source: Path, destination: Path, layers: int, workers: int) -> dict:
    original = json.loads((source / "config.json").read_text())
    index = json.loads((source / "model.safetensors.index.json").read_text())
    config = stage_config(original, layers)
    selected = selected_weight_map(index, layers)
    shards = sorted(set(selected.values()))
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or source.stat().st_dev == destination.stat().st_dev:
        raise ValueError("destination must be a separate node-local filesystem, not the source cache")
    manifest_path = destination / "half-stage-manifest.json"
    identity = dict(source=str(source.resolve()), original_layers=int(
        original.get("text_config", original)["num_hidden_layers"]),
        retained_layers=layers, source_revision=source.name, shards=shards)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if any(previous.get(k) != v for k, v in identity.items()):
            raise ValueError("existing destination belongs to a different checkpoint/stage")
    elif any(destination.iterdir()):
        raise ValueError("refusing to overwrite an unrelated nonempty destination")
    sizes = {name:(source / name).stat().st_size for name in shards}
    missing_bytes = sum(size for name, size in sizes.items()
                        if not (destination / name).is_file()
                        or (destination / name).stat().st_size != size)
    free = shutil.disk_usage(destination).free
    if missing_bytes + 4 * 1024**3 > free:
        raise RuntimeError(f"insufficient local disk: need {missing_bytes} bytes plus reserve, free {free}")
    manifest = identity | dict(status="copying", checkpoint_bytes=sum(sizes.values()),
        selected_tensors=len(selected), missing_copy_bytes=missing_bytes,
        speed_only=True, full_model_daemon_modified=False)
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    print("KIMI_HALF_COPY_PLAN " + json.dumps(manifest), flush=True)

    def copy_shard(name):
        target = destination / name
        if target.is_file() and target.stat().st_size == sizes[name]:
            return name, "reused"
        partial = destination / (name + ".partial")
        shutil.copyfile(source / name, partial)
        if partial.stat().st_size != sizes[name]:
            raise RuntimeError(f"incomplete copy of {name}")
        partial.replace(target)
        return name, "copied"

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(copy_shard, name) for name in shards]
        for completed, future in enumerate(as_completed(futures), 1):
            name, status = future.result()
            print("KIMI_HALF_COPY " + json.dumps(dict(completed=completed,
                total=len(shards), shard=name, status=status)), flush=True)

    # Compute accurate selected tensor bytes from tiny headers, not weight data.
    tensor_bytes = 0
    for name in shards:
        with (destination / name).open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            if header_size > 128 * 1024**2:
                raise ValueError("unreasonable safetensors header size")
            header = json.loads(handle.read(header_size))
        for key, record in header.items():
            if key in selected and selected[key] == name:
                a, b = record["data_offsets"]
                tensor_bytes += b-a
    for asset in source.iterdir():
        if (asset.is_file() and asset.suffix in (".json", ".py", ".txt", ".model", ".tiktoken")
                and asset.name not in ("config.json", "model.safetensors.index.json")):
            shutil.copyfile(asset, destination / asset.name)
    (destination / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    metadata = dict(index.get("metadata", {}), total_size=tensor_bytes)
    (destination / "model.safetensors.index.json").write_text(
        json.dumps(dict(metadata=metadata, weight_map=selected))+"\n")
    manifest.update(status="ready", selected_tensor_bytes=tensor_bytes,
                    copied_shards=len(shards))
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    print("KIMI_HALF_CHECKPOINT_READY " + json.dumps(manifest), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--copy-workers", type=int, default=4)
    parser.add_argument("--serve-cache-id", help="serve a new weight broker after the copy")
    args = parser.parse_args()
    if args.copy_workers < 1:
        parser.error("positive copy worker count required")
    prepare(args.source, args.destination, args.layers, args.copy_workers)
    if args.serve_cache_id:
        wrapper = Path(__file__).with_name("run_kimi_k3_v10_direct.sh")
        os.execvp("bash", ["bash", str(wrapper), "-m",
            "vllm_lod_plugin.weight_cache_daemon", "serve",
            "--cache-id", args.serve_cache_id, "--max-cache-gb-per-gpu", "150",
            "--load-timeout", "3600"])


if __name__ == "__main__":
    main()
