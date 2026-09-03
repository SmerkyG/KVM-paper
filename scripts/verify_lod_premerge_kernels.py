#!/usr/bin/env python3
"""Focused parity checks for fused adjacent-token LOD premerge kernels."""

from __future__ import annotations

import json

from scripts.verify_qwen35_lod_state_kernels import (
    verify_adjacent_premerge_kernels,
)


def main() -> None:
    results = verify_adjacent_premerge_kernels()
    print(json.dumps(results, indent=2, sort_keys=True))
    for factor, result in results.items():
        if (
            result["key_max_abs"] != 0.0
            or result["value_max_abs"] != 0.0
            or not result["counts_exact"]
            or not result["owners_exact"]
        ):
            raise AssertionError(
                f"fused adjacent premerge factor {factor} differs from reference"
            )


if __name__ == "__main__":
    main()
