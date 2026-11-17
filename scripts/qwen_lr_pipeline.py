"""Finish one LR trial, select by validation loss, then evaluate the winner.

Uses one cluster-run allocation for both phases, with separate uv environments.
Only training logs to W&B. The evaluation source is frozen before submission;
its checkpoint path is populated after validation-only model selection.
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
import time


def select_candidate(outputs):
    candidates = {}
    for output in outputs:
        manifest = json.loads((output / "manifest.json").read_text())
        for name, experiment in manifest["experiments"].items():
            config = experiment["resolved_config"]
            if config["model"]["model_class_path"] != "model.qwen3_backbone.Qwen3BackboneForCausalLM":
                continue
            checkpoint = output / "runs" / name
            text = (checkpoint / "train.log").read_text()
            losses = re.findall(r"step:6000/6000 val_loss:([0-9.eE+-]+)", text)
            if len(losses) != 1:
                raise ValueError(f"Expected one final validation loss: {checkpoint}")
            loss = float(losses[0])
            if not math.isfinite(loss) or not (checkpoint / "model.safetensors").is_file():
                raise ValueError(f"Incomplete or nonfinite checkpoint: {checkpoint}")
            candidates[str(checkpoint)] = {
                "experiment": name, "checkpoint": str(checkpoint),
                "peak_lr": config["train"]["optimizer_defaults"]["adam"]["args"]["lr"],
                "final_validation_loss": loss,
            }
    if not candidates:
        raise ValueError("No completed Qwen candidates")
    rows = sorted(candidates.values(), key=lambda row: row["peak_lr"])
    return {"criterion": "minimum final DCLM validation loss at step 6000",
            "selected_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "candidates": rows,
            "winner": min(rows, key=lambda row: row["final_validation_loss"])}


def wait_for_shared_evaluation(output, selection_only=False, timeout=7200):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        manifest = json.loads((output / "manifest.json").read_text())
        pending = manifest.get("selection_pending")
        if selection_only and pending is None:
            return manifest
        if pending:
            controller = Path(pending["controller_output"]) / "pipeline_status.json"
            if controller.exists() and json.loads(controller.read_text()).get("state") == "failed":
                raise RuntimeError(f"Shared selection failed: {controller}")
        status_path = output / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get("state") == "complete":
                return status
            if status.get("state") == "failed":
                raise RuntimeError(f"Shared evaluation failed: {status_path}")
        time.sleep(10)
    raise TimeoutError(f"Timed out waiting for shared selection/evaluation: {output}")


def worker(train_output, experiment, eval_output, candidate_outputs, reuse_eval_output=None):
    train = json.loads((train_output / "manifest.json").read_text())
    evaluation = json.loads((eval_output / "manifest.json").read_text())
    status_path = train_output / "pipeline_status.json"
    status = {"state": "training", "experiment": experiment,
              "eval_output": str(eval_output)}

    def save():
        status_path.write_text(json.dumps(status, indent=2) + "\n")

    try:
        save()
        env = os.environ | {"UV_PROJECT_ENVIRONMENT": train["uv_environment"], "WANDB_MODE": "online"}
        subprocess.run([
            "uv", "run", "--project", train["source"], "--no-sync", "python",
            str(Path(train["source"]) / "scripts/wave1.py"), "worker",
            "--output", str(train_output), "--experiment", experiment,
        ], env=env, check=True)
        shared = (wait_for_shared_evaluation(reuse_eval_output, selection_only=True)
                  if reuse_eval_output is not None else None)
        selection = select_candidate([*candidate_outputs, train_output])
        (train_output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        print("Validation-only selection:", json.dumps(selection), flush=True)
        evaluation["models"] = {"Qwen-selected": selection["winner"]["checkpoint"]}
        evaluation["selection"] = selection
        evaluation.pop("selection_pending", None)
        (eval_output / "manifest.json").write_text(json.dumps(evaluation, indent=2) + "\n")
        if shared is not None and list(shared["models"].values()) == [selection["winner"]["checkpoint"]]:
            shared_status_path = reuse_eval_output / "status.json"
            shared_status = json.loads(shared_status_path.read_text()) if shared_status_path.exists() else {}
            if shared_status.get("state") != "failed":
                evaluation["reused_evaluation_output"] = str(reuse_eval_output)
                (eval_output / "manifest.json").write_text(json.dumps(evaluation, indent=2) + "\n")
                (eval_output / "status.json").symlink_to(shared_status_path)
                status.update(state="complete" if shared_status.get("state") == "complete" else "evaluation_shared",
                              selection=selection, reused_evaluation_output=str(reuse_eval_output))
                print(f"Reusing evaluation from {reuse_eval_output}; releasing this GPU allocation.", flush=True)
                return
        status.update(state="evaluating", selection=selection)
        save()
        env = os.environ | {"UV_PROJECT_ENVIRONMENT": evaluation["environment"], "WANDB_MODE": "disabled"}
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
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-output", type=lambda x: Path(x).resolve(), required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--eval-output", type=lambda x: Path(x).resolve(), required=True)
    parser.add_argument("--candidate-output", type=lambda x: Path(x).resolve(), action="append", required=True)
    parser.add_argument("--reuse-eval-output", type=lambda x: Path(x).resolve())
    args = parser.parse_args()
    worker(args.train_output, args.experiment, args.eval_output, args.candidate_output, args.reuse_eval_output)
