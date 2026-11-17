"""Run the standard training/evaluation pipeline, then evaluate RULER 8K."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from standard_train_eval_pipeline import worker


def run(train_output, experiment, eval_output):
    training = json.loads((train_output / "manifest.json").read_text())
    checks = training["experiments"][experiment].get("gpu_preflight", [])
    if checks:
        subprocess.run([
            "uv", "run", "--no-sync", "python", "-m", "unittest", "-v", *checks,
        ], cwd=training["source"], env=os.environ | {"WANDB_MODE": "disabled"}, check=True)
    try:
        worker(train_output, experiment, eval_output)
    finally:
        status = train_output / "pipeline_status.json"
        if status.is_file() and json.loads(status.read_text()).get("final_validation_loss") is not None:
            manifest = json.loads((eval_output / "manifest.json").read_text())
            ruler = manifest["ruler_8k"]
            subprocess.run([
                "uv", "run", "--project", manifest["source"], "--no-sync", "python",
                ruler["runner"], "worker", ruler["model"],
            ], env=os.environ | {
                "UV_PROJECT_ENVIRONMENT": manifest["environment"], "WANDB_MODE": "disabled",
            }, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--eval-output", type=Path, required=True)
    args = parser.parse_args()
    run(args.train_output, args.experiment, args.eval_output)
