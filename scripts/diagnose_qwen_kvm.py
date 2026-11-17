"""Reproduce Q1 startup on eight GPUs without training or changing its mixer.

Each rank checks a single Qwen+KVM layer at the actual training batch/context,
first eagerly and then with the existing deferred torch.compile decoration.
No optimizer steps, checkpoints, or W&B runs are created.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--layers", type=int, default=1)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cache = Path(tempfile.mkdtemp(prefix="kvm-q1-diagnostic-"))
    os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from model.rwkv7_backbone import MixerConfig
    from model.qwen_kvm_backbone import QwenKVMBackboneForCausalLM
    from scripts.wave1 import read_config
    from utils.defer import apply_deferred

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    config = read_config("Q1").model.model_dump()
    config["num_hidden_layers"] = args.layers

    def report(**values):
        print(json.dumps({"rank": local_rank, **values}), flush=True)

    report(source=str(args.source), layers=args.layers, cache=str(cache))
    for compiled in (False, True):
        torch.manual_seed(0)
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        with torch.device(f"cuda:{local_rank}"):
            model = QwenKVMBackboneForCausalLM(MixerConfig(**config))
        torch.set_default_dtype(old_dtype)
        if compiled:
            apply_deferred(model)
        model = DDP(model, device_ids=[local_rank])
        tokens = torch.randint(0, config["vocab_size"], (8, 8192), device=f"cuda:{local_rank}")
        torch.cuda.reset_peak_memory_stats()
        report(compiled=compiled, phase="forward_start")
        started = time.monotonic()
        result = model(input_ids=tokens, labels=tokens, return_logits=False)
        torch.cuda.synchronize()
        report(compiled=compiled, phase="forward_complete", loss=result.loss.item())
        result.loss.backward()
        torch.cuda.synchronize()
        report(compiled=compiled, phase="backward_complete",
               seconds=time.monotonic()-started,
               memory_mib=torch.cuda.max_memory_allocated()/1024**2)
        dist.barrier()
        del result, tokens, model
        gc.collect()
        torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
