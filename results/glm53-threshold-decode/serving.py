"""Run matched 64K two-tier FP8-weight/BF16-KV full-model decode arms."""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--transfer-b8-recheck", action="store_true")
parser.add_argument("--transfer-comparison", action="store_true")
parser.add_argument("--port-comparison", action="store_true")
parser.add_argument("--trace-cache", type=Path, required=True)
parser.add_argument(
    "--panels",
    nargs="+",
    default=["b1-top8", "b1-c16", "b8-c16", "b8-top8", "b8-native"],
)
args = parser.parse_args()
dest = Path(__file__).resolve().parent
root = dest.parents[1]
if args.port_comparison:
    dest = dest / "gqa-ports"
    args.panels = ["b1-c16", "b1-ported", "b8-ported", "b8-c16"]
if args.transfer_comparison:
    dest = dest / "gqa-transfer"
    args.panels = ["b1-ported", "b1-transfer", "b8-transfer", "b8-ported"]
if args.transfer_b8_recheck:
    dest = dest / "gqa-transfer" / "b8-audit"
    dest.mkdir(parents=True, exist_ok=True)
    args.panels = ["b8-ported", "b8-transfer"]
weights = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())[
    "weight_map"
]
for file in set(weights.values()):
    if not (args.checkpoint / file).is_file():
        raise FileNotFoundError(args.checkpoint / file)


def run_panel(panel, cache_dir):
    batch, mode = panel.split("-")
    assert batch in ("b1", "b8") and mode in ("native", "top8", "c16", "c32", "ported", "transfer")
    env = {
        **os.environ,
        "LOD_REPO_ROOT": str(root),
        "VLLM_WEIGHT_CACHE_DIR": cache_dir,
        "PROLONG_SPEED_TOKEN_CACHE": str(args.trace_cache),
        "TORCH_NCCL_BLOCKING_WAIT": "1",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "0",
        "TORCH_NCCL_ENABLE_MONITORING": "0",
        "LOD_GLM_PROJECTED_LEAVES": "1",
        "LOD_GLM_KDA_PREFILL": "1",
        "OMP_NUM_THREADS": "2",
    }
    env.pop("LOD_GLM_DECODE_THRESHOLD", None)
    env.pop("LOD_GLM_THRESHOLD_DEFERRED_MERGE", None)
    env.pop("LOD_GLM_THRESHOLD_GQA_PORTS", None)
    if mode == "transfer":
        env["LOD_GLM_THRESHOLD_GQA_PORTS"] = "1"
        env["LOD_GLM_DECODE_THRESHOLD"] = "16"
    elif mode == "ported":
        env["LOD_GLM_THRESHOLD_DEFERRED_MERGE"] = "1"
        env["LOD_GLM_DECODE_THRESHOLD"] = "16"
    elif mode == "native":
        env.pop("LOD_GLM_PROJECTED_LEAVES")
        env.pop("LOD_GLM_KDA_PREFILL")
    elif mode != "top8":
        env["LOD_GLM_DECODE_THRESHOLD"] = mode[1:]
    out = dest / f"serving-{panel}.json"
    if (
        panel != "b1-native"
        and out.exists()
        and json.loads(out.read_text()).get("status") == "complete"
    ):
        return
    cmd = [
        "bash",
        "benchmarks/run_kimi_k3_v10_direct.sh",
        "-m",
        "benchmarks.glm53_flash_full",
        "--checkpoint",
        str(args.checkpoint),
        "--mode",
        "full" if mode == "native" else "two-tier",
        "--tp",
        "4",
        "--batch-size",
        batch[1:],
        "--length",
        "65536",
        "--decode-tokens",
        "1026",
        "--measure",
        "speed",
        "--speed-repeats",
        "2",
        "--kv-cache-gib",
        "16",
        "--weight-cache-id",
        "glm-threshold",
        "--output",
        str(out),
    ]
    print(panel, cmd, flush=True)
    subprocess.run(cmd, cwd=root, env=env, check=True)


# Load the complete trained model once, including the native GLM indexer.
# Each LoD client imports the weights it uses from this shared FP8 checkpoint.
with tempfile.TemporaryDirectory(prefix="glm-speed-", dir="/tmp") as cache_dir:
    broker_env = {
        **os.environ,
        "LOD_REPO_ROOT": str(root),
        "VLLM_PLUGINS": "lod_attention",
        "VLLM_LOD_ENABLED": "0",
        "VLLM_WEIGHT_CACHE_DIR": cache_dir,
        "OMP_NUM_THREADS": "2",
    }
    cmd = [
        "bash",
        "benchmarks/run_kimi_k3_v10_direct.sh",
        "-m",
        "vllm_lod_plugin.weight_cache_daemon",
        "serve",
        "--cache-id",
        "glm-threshold",
        "--max-cache-gb-per-gpu",
        "115",
        "--load-timeout",
        "1800",
    ]
    broker = subprocess.Popen(cmd, cwd=root, env=broker_env)
    try:
        ready = Path(cache_dir) / "glm-threshold/control.ready.json"
        deadline = time.monotonic() + 180
        while not ready.is_file():
            if broker.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("The GLM weight-cache broker failed to start")
            time.sleep(0.5)
        # Initialize the broker with the native module tree, including its learned indexer.
        run_panel("b1-native", cache_dir)
        for panel in args.panels:
            run_panel(panel, cache_dir)
    finally:
        broker.terminate()
        broker.wait(timeout=180)
