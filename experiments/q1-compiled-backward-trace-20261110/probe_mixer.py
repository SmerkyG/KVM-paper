"""Compare compiled state read and update on GQA and full-head controls."""

import json
import os
from pathlib import Path
import sys
import tempfile
import time

manifest = json.loads(Path(sys.argv[1]).read_text())
sys.path.insert(0, manifest["source"])
os.chdir(manifest["source"])
cache = Path(tempfile.mkdtemp(prefix="q1-mixer-ops-"))
os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")

import torch
from model.kvm_mixer import SequenceMixer
from model.rwkv7_backbone import MixerConfig

rank = int(os.environ["LOCAL_RANK"])
device = f"cuda:{rank}"
torch.cuda.set_device(rank)
torch.manual_seed(0)
heads = (3, 6)[rank]
settings = manifest["experiments"]["Q1"]["resolved_config"]["model"]
settings = dict(settings, num_key_value_heads=heads)
with torch.device(device):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    mixer = SequenceMixer(MixerConfig(**settings), 0)
    torch.set_default_dtype(previous)


def rand(shape, dtype=torch.bfloat16):
    return torch.randn(shape, dtype=dtype, device=device, requires_grad=True)


def check(label, function, inputs):
    for compiled in (False, True):
        print(json.dumps(dict(heads=heads, op=label, compiled=compiled, phase="start")), flush=True)
        started = time.monotonic()
        current = torch.compile(function, fullgraph=True) if compiled else function
        value = current(*inputs)
        values = value if isinstance(value, tuple) else (value,)
        loss = sum(x.float().square().mean() for x in values)
        torch.cuda.synchronize()
        print(json.dumps(dict(heads=heads, op=label, compiled=compiled, phase="forward", loss=loss.item())), flush=True)
        loss.backward()
        torch.cuda.synchronize()
        assert torch.isfinite(loss)
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in inputs)
        print(json.dumps(dict(heads=heads, op=label, compiled=compiled, phase="backward",
                              seconds=time.monotonic()-started)), flush=True)
        for x in inputs:
            x.grad = None
        mixer.zero_grad(set_to_none=True)


state_k = rand((8, heads, 1425, 128))
state_v = rand((8, heads, 1425, 128))
radius = rand((8, heads, 1425, 1), torch.float32)
check("state_attention", mixer._attend_with_state_and_bswa,
      (rand((8, 6, 256, 128)), rand((8, heads, 512, 128)), rand((8, heads, 512, 128)),
       state_k, state_v, radius))


def merge(s_k, s_v, s_r, new_k, new_v, gates):
    return mixer._update_state_from_overflow_tokens(s_k, s_v, s_r, new_k, new_v, gates, 8192, 7680)


check("state_update", merge,
      (state_k, state_v, radius, rand((8, heads, 256, 128)), rand((8, heads, 256, 128)),
       rand((8, heads, 256, 1))))
