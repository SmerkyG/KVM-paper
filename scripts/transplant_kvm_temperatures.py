#!/usr/bin/env python3
"""Create a checkpoint with selected KVM head temperatures from another model."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from safetensors.torch import load_file, save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("state", "front", "both"), required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    target_file = args.target / "model.safetensors"
    source_file = args.source / "model.safetensors"
    target = load_file(str(target_file), device="cpu")
    source = load_file(str(source_file), device="cpu")

    suffixes = {
        "state": (".attn.state_head_temp",),
        "front": (".attn.front_head_temp",),
        "both": (".attn.state_head_temp", ".attn.front_head_temp"),
    }[args.mode]
    selected = sorted(
        key for key in target if any(key.endswith(suffix) for suffix in suffixes)
    )
    expected = 24 if args.mode == "both" else 12
    if len(selected) != expected:
        raise ValueError(f"expected {expected} temperature tensors, found {len(selected)}")
    for key in selected:
        if key not in source:
            raise KeyError(f"source is missing {key}")
        if source[key].shape != target[key].shape:
            raise ValueError(f"shape mismatch for {key}")
        target[key] = source[key].to(dtype=target[key].dtype)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{args.output.name}-", dir=args.output.parent)
    )
    try:
        for path in args.target.iterdir():
            if path.name != "model.safetensors":
                destination = temp_dir / path.name
                if path.is_dir():
                    shutil.copytree(path, destination)
                else:
                    shutil.copy2(path, destination)
        save_file(target, str(temp_dir / "model.safetensors"))
        temp_dir.rename(args.output)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print(f"created {args.output} with {len(selected)} transplanted tensors")


if __name__ == "__main__":
    main()
