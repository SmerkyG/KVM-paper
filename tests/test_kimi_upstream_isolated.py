"""CPU reference checks; device/graph tests live in the three probes."""

import torch
from benchmarks.kimi_k3_moe_front_probe import reference_front


def test_moe_reference_branch_layout_and_rounding():
    x = torch.tensor([[1.0, 2.0], [-3.0, 1.0]], dtype=torch.bfloat16)
    w = torch.arange(14, dtype=torch.float32).reshape(7, 2).bfloat16()/32
    shared, router, latent = reference_front(x, w, 2, 1)
    p = x.float() @ w.float().T
    gate, up = p[:, :2].bfloat16().float(), p[:, 2:4].bfloat16().float()
    torch.testing.assert_close(shared, (4*torch.tanh(gate/4)*gate.sigmoid()*25*torch.tanh(up/25)).bfloat16())
    torch.testing.assert_close(router, p[:, 4:5])
    torch.testing.assert_close(latent, p[:, 5:].bfloat16())
    assert shared.dtype == latent.dtype == torch.bfloat16 and router.dtype == torch.float32


def test_packing_does_not_change_linear_branches():
    torch.manual_seed(1234)
    x = torch.randn(8, 32).bfloat16()
    w = torch.randn(7, 32).bfloat16()
    a = reference_front(x, w, 2, 1)
    p = torch.cat([x.float()@chunk.float().T for chunk in w.split([4, 1, 2])], dim=-1)
    gate, up = p[:, :2].bfloat16().float(), p[:, 2:4].bfloat16().float()
    b = ((4*torch.tanh(gate/4)*gate.sigmoid()*25*torch.tanh(up/25)).bfloat16(), p[:, 4:5], p[:, 5:].bfloat16())
    for left, right in zip(a, b, strict=True):
        torch.testing.assert_close(left, right)
