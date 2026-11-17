"""Full-shape Q1 compiled forward/backward regression; run through cluster-run.

Requires torchrun with eight ranks and a frozen Q1 manifest. This deliberately
uses the real 12-layer geometry and per-device 8x8192 training shape. It does
not update weights, write a model checkpoint, or log to W&B.
"""

from __future__ import annotations

import argparse
import gc
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-output", type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument("--result", type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    manifest = json.loads((args.train_output / "manifest.json").read_text())
    source = Path(manifest["source"])
    for name, expected in manifest["source_files"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source snapshot changed: {name}")
    sys.path.insert(0, str(source))
    os.chdir(source)
    cache = args.result.parent / "trace-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
    os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")

    import torch
    import torch.distributed as dist
    from torch._inductor import config as inductor_config
    faulthandler.enable()
    inductor_config.triton.debug_sync_kernel = True
    from torch.nn.parallel import DistributedDataParallel as DDP
    from model.rwkv7_backbone import MixerConfig
    from model.qwen_kvm_backbone import QwenKVMBackboneForCausalLM
    from utils.defer import apply_deferred

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    if dist.get_world_size() != 1:
        raise ValueError("Submit this regression as one eight-GPU torchrun job")
    config = manifest["experiments"]["Q1"]["resolved_config"]
    if config["model"]["num_hidden_layers"] != 12:
        raise ValueError("The regression must use the full Q1 model")
    torch.manual_seed(config["train"]["init_seed"])
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with torch.device(f"cuda:{rank}"):
        raw = QwenKVMBackboneForCausalLM(MixerConfig(**config["model"]))
    torch.set_default_dtype(old_dtype)
    apply_deferred(raw)
    model = DDP(raw, device_ids=[rank])
    shape = (config["train"]["train_data"]["device_batch_size"],
             config["train"]["train_data"]["sequence_length"])
    if shape != (8, 8192):
        raise ValueError(f"Unexpected training input shape: {shape}")
    rows = []
    for step in range(args.steps):
        tokens = torch.randint(0, config["model"]["vocab_size"], shape, device=f"cuda:{rank}")
        labels = tokens.roll(-1, dims=1).contiguous()
        started = time.monotonic()
        print(json.dumps({"rank": rank, "step": step, "phase": "forward_start"}), flush=True)
        result = model(input_ids=tokens, labels=labels, return_logits=False)
        torch.cuda.synchronize()
        loss = result.loss.item()
        if not torch.isfinite(result.loss):
            raise FloatingPointError(f"Nonfinite loss: {loss}")
        print(json.dumps({"rank": rank, "step": step, "phase": "backward_start", "loss": loss}), flush=True)
        result.loss.backward()
        torch.cuda.synchronize()
        bad = [name for name, value in model.named_parameters()
               if value.grad is not None and not torch.isfinite(value.grad).all()]
        if bad:
            raise FloatingPointError(f"Nonfinite gradients: {bad}")
        row = {"rank": rank, "step": step, "phase": "backward_complete", "loss": loss,
               "seconds": time.monotonic()-started,
               "peak_memory_mib": torch.cuda.max_memory_allocated()/1024**2}
        rows.append(row)
        print(json.dumps(row), flush=True)
        model.zero_grad(set_to_none=True)
        del result, tokens, labels
        gc.collect()
    gathered = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(rows, gathered, dst=0)
    if rank == 0:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps({"state": "passed", "source_sha256": manifest["source_sha256"],
            "layers": 12, "per_device_shape": list(shape), "steps": args.steps, "diagnostic": "synchronized decoder backward",
            "world_size": 1, "optimizer_steps": 0, "results": gathered}, indent=2)+"\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
