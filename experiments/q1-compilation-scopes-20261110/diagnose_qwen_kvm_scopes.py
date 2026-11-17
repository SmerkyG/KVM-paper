"""Isolate Q1 compilation failures without optimizer steps or source changes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def split_layer(layer, torch):
    """Diagnostic graph boundaries matching separate attention/FFN compilation."""
    def attention(hidden_states, position_embeddings):
        value, _ = layer.self_attn(layer.input_layernorm(hidden_states), position_embeddings)
        return hidden_states + value

    def mlp(hidden_states):
        return hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))

    compiled_attention = torch.compile(attention)
    compiled_mlp = torch.compile(mlp)

    def forward(hidden_states, position_embeddings=None, **kwargs):
        return compiled_mlp(compiled_attention(hidden_states, position_embeddings))

    layer.forward = forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--scope", choices=("decoder", "loss", "split"), required=True)
    args = parser.parse_args()
    manifest = json.loads((args.train_output.resolve() / "manifest.json").read_text())
    sys.path.insert(0, manifest["source"])
    os.chdir(manifest["source"])
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
    cache = Path(tempfile.mkdtemp(prefix=f"kvm-q1-{args.scope}-"))
    os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from model.rwkv7_backbone import MixerConfig
    from model.qwen_kvm_backbone import QwenKVMBackboneForCausalLM

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    config = manifest["experiments"]["Q1"]["resolved_config"]["model"]
    config = dict(config, num_hidden_layers=1)
    torch.manual_seed(0)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with torch.device(f"cuda:{rank}"):
        model = QwenKVMBackboneForCausalLM(MixerConfig(**config))
    torch.set_default_dtype(old_dtype)
    if args.scope == "loss":
        model._compute_loss_and_logits = torch.compile(model._compute_loss_and_logits)
    elif args.scope == "decoder":
        for layer in model.transformer.layers:
            layer.forward = torch.compile(layer.forward)
    else:
        for layer in model.transformer.layers:
            split_layer(layer, torch)
    model = DDP(model, device_ids=[rank])
    tokens = torch.randint(0, config["vocab_size"], (8, 8192), device=f"cuda:{rank}")

    def report(**values):
        print(json.dumps({"rank": rank, "scope": args.scope, **values}), flush=True)

    report(phase="forward_start", cache=str(cache), source_sha256=manifest["source_sha256"])
    started = time.monotonic()
    result = model(input_ids=tokens, labels=tokens, return_logits=False)
    torch.cuda.synchronize()
    report(phase="forward_complete", loss=result.loss.item())
    result.loss.backward()
    torch.cuda.synchronize()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    report(phase="backward_complete", seconds=time.monotonic()-started,
           memory_mib=torch.cuda.max_memory_allocated()/1024**2)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
