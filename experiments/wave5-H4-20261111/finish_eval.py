"""Resume H4's evaluation once after a recognized NIAH or model-cache failure."""
import json
from pathlib import Path
import subprocess
import sys
import time


root = Path(__file__).resolve().parent
run = root / "H4"
evaluation = run / "eval"
repo = root.parents[1]
deadline = time.monotonic() + 6 * 3600
while time.monotonic() < deadline:
    try:
        status = json.loads((run / "pipeline_status.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        time.sleep(30)
        continue
    if status["state"] == "complete":
        print("H4 completed its original evaluation", flush=True)
        break
    if status["state"] != "failed":
        time.sleep(30)
        continue
    result = evaluation / "status.json"
    if not result.is_file():
        raise RuntimeError("H4 failed before evaluation. Inspect its training log.")
    stages = json.loads(result.read_text())["stages"]
    failed = [stage for stage in stages if stage["state"] == "failed"]
    recognized = {
        "niah-4k-8k-16k": "Memory access fault",
        "smoke": "has no attribute 'MixerConfig'",
    }
    if not failed or any(
        stage["suite"] not in recognized
        or recognized[stage["suite"]] not in Path(stage["log"]).read_text()
        for stage in failed
    ):
        raise RuntimeError("H4 encountered an evaluation failure outside the diagnosed cases.")
    subprocess.run([
        sys.executable, str(repo / "scripts/retry_eval.py"),
        "--previous", str(evaluation), "--output", str(root / "eval-retry8"),
        "--overlay", "scripts/eval_wave1.py", "--overlay", "wrap_lmeval.py",
        "--batch-size", "niah-4k-8k-16k", "8",
    ], cwd=repo, check=True)
    break
else:
    raise TimeoutError("H4 did not finish within six hours")
