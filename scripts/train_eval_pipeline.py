"""Train one frozen Wave 1 experiment, then evaluate its final checkpoint.

Submit through cluster-run with eight GPUs. Training uses the training uv
environment and online W&B; evaluations use the isolated evaluation environment
and local result files. Both source snapshots must be prepared beforehand.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import subprocess


def worker(train_output, experiment, eval_output):
    train = json.loads((train_output / "manifest.json").read_text())
    evaluation = json.loads((eval_output / "manifest.json").read_text())
    checkpoint = train_output / "runs" / experiment
    if list(evaluation["models"].values()) != [str(checkpoint)]:
        raise ValueError("Evaluation must target this experiment's final checkpoint")
    status_path = train_output / "pipeline_status.json"
    status = {"state": "training", "experiment": experiment,
              "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "checkpoint": str(checkpoint), "eval_output": str(eval_output)}

    def save():
        status_path.write_text(json.dumps(status, indent=2) + "\n")

    try:
        save()
        env = os.environ | {"UV_PROJECT_ENVIRONMENT": train["uv_environment"],
                            "WANDB_MODE": "online"}
        subprocess.run([
            "uv", "run", "--project", train["source"], "--no-sync", "python",
            str(Path(train["source"]) / "scripts/wave1.py"), "worker",
            "--output", str(train_output), "--experiment", experiment,
        ], env=env, check=True)
        steps = train["experiments"][experiment]["resolved_config"]["train"]["num_iterations"]
        losses = re.findall(rf"step:{steps}/{steps} val_loss:([0-9.eE+-]+)",
                            (checkpoint / "train.log").read_text())
        if len(losses) != 1 or not math.isfinite(float(losses[0])):
            raise ValueError("Training did not finish with one finite final validation loss")
        for filename in ("model.safetensors", "config.json", "tokenizer.json"):
            if not (checkpoint / filename).is_file():
                raise FileNotFoundError(checkpoint / filename)
        evaluation.pop("checkpoint_pending", None)
        (eval_output / "manifest.json").write_text(json.dumps(evaluation, indent=2) + "\n")
        status.update(state="evaluating", final_validation_loss=float(losses[0]))
        save()
        env = os.environ | {"UV_PROJECT_ENVIRONMENT": evaluation["environment"],
                            "WANDB_MODE": "disabled"}
        subprocess.run([
            "uv", "run", "--project", evaluation["source"], "--no-sync", "python",
            str(Path(evaluation["source"]) / "scripts/eval_wave1.py"), "worker",
            "--output", str(eval_output),
        ], env=env, check=True)
        status["state"] = "complete"
    except BaseException as exc:
        status.update(state="failed", error=repr(exc))
        raise
    finally:
        status["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-output", type=lambda x: Path(x).resolve(), required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--eval-output", type=lambda x: Path(x).resolve(), required=True)
    args = parser.parse_args()
    worker(args.train_output, args.experiment, args.eval_output)
