"""Exercise the real vLLM HTTP server locally; submit through cluster-run."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen

from benchmarks._identity import benchmark_identity
from benchmarks._vllm import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("two-tier", "three-tier-bf16", "three-tier-int4"),
        default="three-tier-int4",
    )
    args = parser.parse_args()
    port = os.environ["CLUSTER_RUN_PORT_0"]
    base = f"http://127.0.0.1:{port}"
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.checkpoint,
        "--served-model-name",
        "lod-llama",
        "--host",
        "127.0.0.1",
        "--port",
        port,
        "--attention-config",
        '{"backend":"CUSTOM"}',
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "bfloat16",
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "16385",
        "--long-prefill-token-threshold",
        "16384",
        "--scheduler-cls",
        "vllm_lod_plugin.scheduler.LODChunkAlignedScheduler",
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization",
        "0.8",
    ]
    env = os.environ | dict(
        VLLM_PLUGINS="lod_attention",
        VLLM_LOD_MODE=args.mode,
        VLLM_LOD_POOL_SIZE="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
    )
    result = dict(
        benchmark="h100-vllm-http-smoke",
        mode=args.mode,
        benchmark_identity=benchmark_identity(),
        command=command,
        stage="starting",
        responses=[],
    )
    write_json(args.output, result)
    process = subprocess.Popen(command, env=env)
    started = time.monotonic()
    try:
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"vLLM server exited with {process.returncode}")
            try:
                with urlopen(base + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() - started > 300:
                raise TimeoutError("vLLM server did not become healthy")
            time.sleep(0.5)
        result["startup_seconds"] = time.monotonic() - started
        filler = "The archive contains routine records about weather, roads, and buildings.\n"
        cases = [
            ("What is the capital of France? Answer with one word.", "paris"),
            ("What is 17 plus 25? Answer with just the number.", "42"),
            (
                "Find the secret access code.\n"
                + filler * 750
                + "\nThe secret access code is 735926.\n"
                + filler * 750
                + "\nReport only the secret access code.",
                "735926",
            ),
        ]
        for prompt, expected in cases:
            payload = dict(
                model="lod-llama",
                messages=[dict(role="user", content=prompt)],
                temperature=0,
                max_tokens=32,
                seed=0,
            )
            request = Request(
                base + "/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=180) as response:
                body = json.load(response)
            text = body["choices"][0]["message"]["content"]
            row = dict(
                expected=expected,
                text=text,
                usage=body["usage"],
                correct=expected in text.lower(),
            )
            result["responses"].append(row)
            write_json(args.output, result)
            print("HTTP_RESPONSE " + json.dumps(row), flush=True)
            if not row["correct"]:
                raise AssertionError(f"Unexpected answer: {text!r}")
        if result["responses"][-1]["usage"]["prompt_tokens"] <= 16384:
            raise AssertionError("The HTTP test did not exercise remote LoD prefill")
        result["stage"] = "complete"
        write_json(args.output, result)
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    main()
