"""Resume a frozen evaluation and preserve its successful suites."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from eval_wave1 import reuse_completed_stages
from wave1 import REPO, verify_source
from wave3 import read_verified


def main(previous, output, overlays, batch_sizes, node):
    manifest = json.loads((previous / "manifest.json").read_text())
    verify_source(manifest)
    output.mkdir(parents=True)
    if overlays:
        source = output / "source"
        for name in manifest["source_files"]:
            original = read_verified(manifest, name)
            content = (REPO / name).read_bytes() if name in overlays else original
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            manifest["source_files"][name] = hashlib.sha256(content).hexdigest()
        manifest["source"] = str(source)
    manifest["reused_stages"] = reuse_completed_stages(previous, output)
    if batch_sizes:
        manifest["batch_size_overrides"] = {suite: int(size) for suite, size in batch_sizes}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    source = Path(manifest["source"])
    command = ["cluster-run", "--detach", "--num-gpus", "8", "--name", output.name]
    if node:
        command.extend(["--nodes", node])
    command.extend([
        "--", "env", f"UV_PROJECT_ENVIRONMENT={manifest['environment']}", "WANDB_MODE=disabled",
        "uv", "run", "--no-sync", "--project", str(source), "python",
        str(source / "scripts/eval_wave1.py"), "worker", "--output", str(output),
    ])
    submitted = subprocess.check_output(command, text=True)
    (output / "submission.json").write_text(
        json.dumps({"command": command, "response": submitted}, indent=2) + "\n"
    )
    print(submitted, end="", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--batch-size", nargs=2, action="append", metavar=("SUITE", "SIZE"))
    parser.add_argument("--node")
    args = parser.parse_args()
    main(args.previous.resolve(), args.output.resolve(), args.overlay, args.batch_size, args.node)
