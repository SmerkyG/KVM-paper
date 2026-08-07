import os

import torch
from torch.nn.attention.flex_attention import flex_attention


_COMPILE_MODE = os.environ.get(
    "KVM_FLEX_ATTENTION_COMPILE_MODE", "max-autotune-no-cudagraphs"
)


@torch.compile(mode=_COMPILE_MODE, fullgraph=True)
def compiled_flex_attention(*args, **kwargs):
    return flex_attention(*args, **kwargs)


@torch.compiler.disable
def separately_compiled_flex_attention(*args, **kwargs):
    return compiled_flex_attention(*args, **kwargs)


def causal_mask_mod(b, h, q_idx, kv_idx):
    return kv_idx <= q_idx
