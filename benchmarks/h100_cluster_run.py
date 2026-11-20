#!/usr/bin/env python3
"""Logged cluster-run adapter for dedicated SSH H100 hosts.

This is independent of the paper cluster's fleet scheduler. It locks each
selected GPU and snapshots source for every experiment before submission.

Submit: cluster-run --name NAME --detach --gpu GPU --num-gpus 1 -- COMMAND [ARGS...]
Inspect: cluster-run status
Read log: cluster-run logs RUN_ID
Stop: cluster-run cancel RUN_ID
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = Path.home() / ".local/state/lod-cluster-run"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def worker(directory):
    metadata_path = directory / "run.json"
    metadata = read(metadata_path)
    gpu = metadata["gpu"]
    with (STATE / f"gpu-{gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with socket.socket() as port_socket:
            port_socket.bind(("127.0.0.1", 0))
            port = port_socket.getsockname()[1]
        metadata.update(
            state="running", started_epoch=time.time(), pid=os.getpid(), port=port
        )
        metadata["hardware"] = subprocess.check_output(
            [
                "nvidia-smi", "-i", gpu,
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        write(metadata_path, metadata)
        source = directory / "source"
        env = os.environ | dict(
            CUDA_VISIBLE_DEVICES=gpu,
            CLUSTER_RUN_NODE=socket.gethostname(),
            CLUSTER_RUN_GPUS=gpu,
            MASTER_PORT=str(port),
            CLUSTER_RUN_PORT_0=str(port),
            PYTHONUNBUFFERED="1",
            PYTHONPATH=f"{source}:{source}/integrations/vllm_lod",
            PATH=str(Path.home() / ".local/bin") + ":" + os.environ["PATH"],
        )
        process = subprocess.Popen(
            metadata["command"], cwd=source, env=env, start_new_session=True
        )

        def terminate(signum, _frame):
            if process.poll() is None:
                os.killpg(process.pid, signum)

        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        metadata["command_pid"] = process.pid
        write(metadata_path, metadata)
        code = process.wait()
        metadata.update(state="finished", exit_code=code, finished_epoch=time.time())
        write(metadata_path, metadata)
        return code


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        raise SystemExit(worker(Path(sys.argv[2])))
    if len(sys.argv) > 1 and sys.argv[1] in ("status", "logs", "cancel"):
        action = sys.argv[1]
        if action == "status":
            for path in sorted(STATE.glob("*/run.json")):
                row = read(path)
                print(row["id"], row["state"], row.get("exit_code", ""), row["command"])
        else:
            directory = STATE / sys.argv[2]
            if action == "logs":
                print((directory / "output.log").read_text(), end="")
            else:
                row = read(directory / "run.json")
                if row["state"] == "finished":
                    return
                pid = int((directory / "worker.pid").read_text())
                os.kill(pid, signal.SIGTERM)
                if row["state"] == "queued":
                    row.update(
                        state="finished",
                        exit_code=-signal.SIGTERM,
                        finished_epoch=time.time(),
                    )
                    write(directory / "run.json", row)
        return
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--num-gpus", type=int, choices=(1,), default=1)
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU index on this host")
    parser.add_argument("--source-dir", type=Path, default=ROOT, help="Snapshot this source checkout")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide -- COMMAND [ARGS...]")
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+", args.name):
        parser.error("name must contain letters, digits, dots, underscores, or dashes")
    if args.gpu < 0 or subprocess.run(
        ["nvidia-smi", "-i", str(args.gpu), "--query-gpu=index", "--format=csv,noheader"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    ).returncode:
        parser.error(f"GPU index is unavailable: {args.gpu}")
    root = args.source_dir.resolve()
    run_id = f"{time.time_ns()}-{args.name}"
    directory = STATE / run_id
    source = directory / "source"
    source.mkdir(parents=True)
    for name in ("lod_attention", "integrations", "benchmarks", "tests"):
        shutil.copytree(
            root / name,
            source / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "figures"),
        )
    for name in ("pyproject.toml", "uv.lock", "README.md"):
        shutil.copy2(root / name, source / name)
    files = {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    }
    metadata = dict(
        id=run_id,
        state="queued",
        submitted_epoch=time.time(),
        command=command,
        original_workdir=str(root),
        source=str(source),
        source_files=files,
        gpu=str(args.gpu),
        node=socket.gethostname(),
    )
    write(directory / "run.json", metadata)
    with (directory / "output.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "_worker", str(directory)],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    (directory / "worker.pid").write_text(str(process.pid))
    print("Submitted " + run_id, flush=True)
    print("Metadata: " + str(directory / "run.json"), flush=True)
    print("Log: " + str(directory / "output.log"), flush=True)
    if not args.detach:
        try:
            raise SystemExit(process.wait())
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            raise SystemExit(process.wait())


if __name__ == "__main__":
    main()
