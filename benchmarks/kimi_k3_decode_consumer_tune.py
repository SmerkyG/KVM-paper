"""Graph-time the complete compact attention consumer across split counts.

Synthetic cache geometry, not full-model serving. Reuse the existing biased
attention reference probe, including per-head-tile coarse replacement masks.
"""

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from unittest.mock import patch

from benchmarks import kimi_gluon_lod_decode_probe as probe
from benchmarks.kimi_k3_decode_route_tune import graph_us


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--large-only", action="store_true",
                        help="Check the selected split geometry on longer compact sequences")
    parser.add_argument("--single-row-only", action="store_true",
                        help="Tune physical B1 used by ordinary B1 and request-owned decode")
    args = parser.parse_args()
    if args.large_only and args.single_row_only:
        parser.error("choose one consumer geometry sweep")
    result = dict(scope="compact consumer only; synthetic KV, excludes routing/communication", rows=[])
    geometries = ((1, 512, 16), (1, 2048, 256), (1, 4096, 256),
                  (1, 8192, 1024)) if args.single_row_only else (
        ((8, 8192, 1024), (8, 16384, 4096)) if args.large_only else (
            (1, 512, 16), (8, 512, 16), (8, 512, 128), (8, 2048, 256)))
    for batch, coarse, pages in geometries:
        for splits in ((16, 64) if args.large_only else (8, 16, 32, 64)):
            argv = ["probe", "--batch-size", str(batch), "--heads", "96", "--coarse",
                    str(coarse), "--local", "64", "--exact-pages", str(pages),
                    "--splits", str(splits), "--head-tiled-metadata"]
            capture = io.StringIO()
            with patch("sys.argv", argv), patch.object(probe, "_time_ms",
                    lambda fn, warmup, iterations: graph_us(fn)[0]/1000), redirect_stdout(capture):
                probe.main()
            row = json.loads(capture.getvalue().splitlines()[-1])
            if row["max_abs_error"] > .003 or row["max_lse_error"] > 5e-4:
                raise AssertionError(row)
            result["rows"].append(row)
            print(json.dumps(row), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2)+"\n")
    result["status"] = "complete"
    args.output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
