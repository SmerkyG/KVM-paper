"""Evaluate the released 120M KVM-sqrt checkpoint under K0's NIAH protocol."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

output = Path(__file__).resolve().parent
reference = output.parent / "kvm-sqrt-original-eval-20261110"
manifest = json.loads((reference / "manifest.json").read_text())
status = json.loads((reference / "status.json").read_text())
stage = next(stage for stage in status["stages"] if stage["suite"] == "niah-4k-8k-16k")
released = Path.home() / ".cache/huggingface/hub/models--featherless-ai--kvmpaper_kvmsqrt16_120M/snapshots/d771c87cc263ec36fe609182c2e9c262f8e68d29"
assert (released / "model.safetensors").is_file()
source = Path(manifest["source"])
result = output / "niah"
result.mkdir()
command = list(stage["command"])
model_args = command.index("--model_args") + 1
original_args = command[model_args].split(",")
original_args[0] = f"pretrained={released}"
command[model_args] = ",".join(original_args)
command[command.index("--main_process_port") + 1] = os.environ["MASTER_PORT"]
command[command.index("--output_path") + 1] = str(result)
cache = Path(tempfile.mkdtemp(prefix="kvm-published-niah-"))
env = os.environ | {
    "HF_WRAPPER_BASE_PATH": str(source), "PYTHONPATH": str(source),
    "NLTK_DATA": manifest["nltk_data"], "HF_MODULES_CACHE": str(cache / "modules"),
    "TRITON_CACHE_DIR": str(cache / "triton"), "TORCHINDUCTOR_DIR": str(cache / "inductor"),
    "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
    "WANDB_MODE": "disabled", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "TORCHINDUCTOR_COMPILE_THREADS": "2",
}
record = {"command": command, "source": str(source), "checkpoint": str(released),
          "reference_result": stage["result_file"], "state": "running"}
record_path = output / "status.json"
record_path.write_text(json.dumps(record, indent=2) + "\n")
try:
    # Stage the local wrapper once to avoid concurrent HF file copies by the ranks.
    subprocess.run([
        sys.executable, "-c",
        "from transformers import AutoConfig; import sys; AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True, local_files_only=True)",
        str(released),
    ], cwd=source, env=env, check=True)
    subprocess.run(command, cwd=source, env=env, check=True)
    raw, = result.rglob("results_*.json")
    record.update(state="complete", result_file=str(raw))
except BaseException as exc:
    record.update(state="failed", error=repr(exc))
    raise
finally:
    record_path.write_text(json.dumps(record, indent=2) + "\n")
