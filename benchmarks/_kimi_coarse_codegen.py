"""Isolated CK coarse-tile experiment; never edits installed AITER.

Wrap the canonical generator and change gfx9 BF16 D192/V128 query tile,
query-axis warp count, or feature-axis step. The 128-key token tile and all
LoD score math stay unchanged. The caller must use a distinct JIT module name.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-generator", type=Path, required=True)
    parser.add_argument("--query-tile", type=int, choices=(64, 128), required=True)
    parser.add_argument("--key-step", type=int, choices=(32, 64), default=32)
    parser.add_argument("-d", "--direction", default="fwd")
    parser.add_argument("--receipt", type=int, default=104)
    parser.add_argument("--filter", default="*")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--optdim", default="192")
    parser.add_argument("--targets", default="gfx9,gfx950")
    parser.add_argument("--mask", default="simplified")
    args = parser.parse_args()
    if args.direction != "fwd" or args.receipt != 104 or args.optdim != "192":
        parser.error("this probe is limited to the biased Kimi coarse receipt")
    sys.path.insert(0, str(args.base_generator.parent))
    spec = importlib.util.spec_from_file_location("lod_kimi_ck_codegen", args.base_generator)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    forward = next(op for op in generator.ops if op.__name__.endswith("fmha_fwd"))
    factory = forward.KernelComponentFactoryGfx9
    original = factory.get_hdim_tile_size_dict
    original_rules = factory.get_rules

    def modified(cls, dtype):
        tiles = original(dtype)
        if dtype == "bf16" and tiles is not None and (192, 128) in tiles:
            tiles = dict(tiles)
            warps = 2 if args.query_tile == 64 else 4
            tiles[(192, 128)] = [replace(tile, F_bm0=args.query_tile, F_rm0=warps,
                                        F_rm1=warps, F_bk0=args.key_step)
                                  for tile in tiles[(192, 128)]]
        return tiles

    factory.get_hdim_tile_size_dict = classmethod(modified)
    # The stock generator conservatively admits M=128 only for non-D128 QR
    # cases. Leave every other compatibility rule intact and let the template
    # compiler plus direct dense-reference tests validate this isolated M64
    # probe. This override cannot affect installed AITER or ordinary builds.
    factory.get_rules = classmethod(lambda cls: [rule for rule in original_rules()
                                                if rule.__name__ != "check_hdim_tile"])
    generator.write_blobs(args.targets.split(","), args.output_dir, ["fwd"],
                          [args.filter], [192], 104, args.mask)


if __name__ == "__main__":
    main()
