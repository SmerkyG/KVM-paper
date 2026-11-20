"""Check the QUEST benchmark's fused Llama 3.1 RoPE against Transformers."""
import argparse

import torch
from flashinfer.rope import apply_llama31_rope_inplace
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
args = parser.parse_args()
torch.manual_seed(1234)
config = AutoConfig.from_pretrained(args.checkpoint)
rotary = LlamaRotaryEmbedding(config=config, device="cuda")
for start, length in [(0, 16), (8191, 2), (32768, 1), (65536, 1), (66560, 1)]:
    q = torch.randn(length, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(length, 8, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(start, start + length, device="cuda").unsqueeze(0)
    cos, sin = rotary(k.transpose(0, 1).unsqueeze(0), positions)
    qr, kr = apply_rotary_pos_emb(q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), cos, sin)
    indptr = torch.tensor([0, length], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([start], device="cuda", dtype=torch.int32)
    apply_llama31_rope_inplace(q, k, indptr, offsets)
    torch.testing.assert_close(q, qr.squeeze(0).transpose(0, 1), rtol=0.02, atol=0.04)
    torch.testing.assert_close(k, kr.squeeze(0).transpose(0, 1), rtol=0.02, atol=0.04)
    print(f"Llama 3.1 RoPE agrees at start={start}, length={length}", flush=True)
