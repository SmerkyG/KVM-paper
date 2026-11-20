"""Run two K2 routing policies sequentially on one allocation."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs=2, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for name in args.variants:
        print(f"Timing {name}", flush=True)
        command = [sys.executable, "-P", "-m", "benchmarks.prolong",
                   "--checkpoint", "IFM/K2-Horizon-32B-FP8", "--measure", "speed",
                   "--mode", "two-tier", "--lengths", "65536", "--batch-size", "1",
                   "--decode-tokens", "1025", "--repeats", str(args.repeats),
                   "--output", str(args.output / (name + ".json"))]
        subprocess.run(command, env={**os.environ, **settings[name]}, check=True)


if __name__ == "__main__":
    main()
