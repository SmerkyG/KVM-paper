"""Repair the two running Qwen exports before their local evals load them."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
HELPERS = b"\nfrom model.rwkv7_backbone import RotaryEmbedding, apply_rotary_embeddings\n"


def check_run(name):
    directory = ROOT / name
    export = directory / "runs" / name
    wrapper = export / "rwkv7_backbone.py"
    status_file = directory / "pipeline_status.json"
    if not status_file.is_file():
        return False
    try:
        status = json.loads(status_file.read_text())
    except json.JSONDecodeError:
        return False
    if wrapper.is_file() and (export / "model.safetensors").is_file():
        code = wrapper.read_bytes()
        if b"RWKV7BackboneConfig = MixerConfig" in code and b"RotaryEmbedding" not in code:
            temporary = wrapper.with_suffix(".py.tmp")
            temporary.write_bytes(code + HELPERS)
            temporary.replace(wrapper)
            print(f"{name} checkpoint wrapper repaired", flush=True)

    eval_status = directory / "eval/status.json"
    if eval_status.is_file():
        try:
            stages = json.loads(eval_status.read_text())["stages"]
        except json.JSONDecodeError:
            return False
        if any(stage["suite"] == "smoke" and stage["state"] == "complete" for stage in stages):
            print(f"{name} evaluation loaded the exported checkpoint", flush=True)
            return True
        if status["state"] == "failed":
            failed = [stage for stage in stages if stage["state"] == "failed"]
            if failed and all(
                stage["suite"] == "smoke"
                and "cannot import name 'RotaryEmbedding'" in Path(stage["log"]).read_text()
                for stage in failed
            ):
                subprocess.run([
                    sys.executable, str(REPO / "scripts/retry_eval.py"),
                    "--previous", str(directory / "eval"),
                    "--output", str(directory / "eval-export-retry"),
                ], cwd=REPO, check=True)
                return True
    if status["state"] in ("complete", "failed"):
        print(f"{name} pipeline {status['state']}", flush=True)
        return True
    return False


if __name__ == "__main__":
    (ROOT / "finish_qwen_exports.pid").write_text(f"{os.getpid()}\n")
    print("Watching Q0 and Q1 checkpoint exports", flush=True)
    remaining = {"Q0", "Q1"}
    deadline = time.monotonic() + 16 * 3600
    while remaining and time.monotonic() < deadline:
        remaining = {name for name in remaining if not check_run(name)}
        if remaining:
            time.sleep(2)
    if remaining:
        raise TimeoutError(f"Still waiting on {', '.join(sorted(remaining))}")
