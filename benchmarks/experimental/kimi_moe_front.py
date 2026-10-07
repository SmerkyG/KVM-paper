"""Isolated gfx942 version of AITER #5321's merged K3 MoE front.

BF16 weights/inputs, FP32 merged accumulation, and explicit BF16 rounding
before SiTU preserve the native branch contracts. Routed experts and output
transforms are not part of this experiment. No production patch is applied.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _split_situ(p, shared, router, latent, ROW: tl.constexpr,
                SHARED: tl.constexpr, ROUTER: tl.constexpr, LATENT: tl.constexpr,
                BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(p + row * ROW + col, mask=col < SHARED, other=0).to(tl.bfloat16).to(tl.float32)
    up = tl.load(p + row * ROW + SHARED + col, mask=col < SHARED, other=0).to(tl.bfloat16).to(tl.float32)
    # Do not use 2*sigmoid(2*x)-1: it loses relative precision near zero,
    # unlike the native SiTU tanhf, and failed the small-model oracle.
    result = (4.0 * libdevice.tanh(gate / 4.0) * tl.sigmoid(gate)) * (25.0 * libdevice.tanh(up / 25.0))
    tl.store(shared + row * SHARED + col, result, mask=col < SHARED)
    r = tl.load(p + row * ROW + 2 * SHARED + col, mask=col < ROUTER, other=0)
    tl.store(router + row * ROUTER + col, r, mask=col < ROUTER)
    v = tl.load(p + row * ROW + 2 * SHARED + ROUTER + col, mask=col < LATENT, other=0)
    tl.store(latent + row * LATENT + col, v, mask=col < LATENT)


def merged_front(x, packed_weight, projection, shared, router, latent):
    if x.dtype != torch.bfloat16 or packed_weight.dtype != torch.bfloat16:
        raise ValueError("this experiment does not downcast FP32 router weights")
    torch.mm(x, packed_weight.T, out=projection, out_dtype=torch.float32)
    _split_situ[(x.shape[0], triton.cdiv(max(shared.shape[1], router.shape[1], latent.shape[1]), 256))](
        projection, shared, router, latent, ROW=packed_weight.shape[0],
        SHARED=shared.shape[1], ROUTER=router.shape[1], LATENT=latent.shape[1], BLOCK=256,
        num_warps=4)
    return shared, router, latent


@triton.jit
def _split_routed(p, router, latent, ROW: tl.constexpr, ROUTER: tl.constexpr,
                  LATENT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)*BLOCK + tl.arange(0, BLOCK)
    r = tl.load(p + row*ROW + col, mask=col < ROUTER, other=0)
    v = tl.load(p + row*ROW + ROUTER + col, mask=col < LATENT, other=0)
    tl.store(router + row*ROUTER + col, r, mask=col < ROUTER)
    tl.store(latent + row*LATENT + col, v, mask=col < LATENT)


def merged_routed_front(x, packed_weight, projection, router, latent):
    if x.dtype != torch.bfloat16 or packed_weight.dtype != torch.bfloat16:
        raise ValueError("this experiment does not downcast FP32 router weights")
    torch.mm(x, packed_weight.T, out=projection, out_dtype=torch.float32)
    _split_routed[(x.shape[0], triton.cdiv(max(router.shape[1], latent.shape[1]), 256))](
        projection, router, latent, ROW=packed_weight.size(0), ROUTER=router.size(1),
        LATENT=latent.size(1), BLOCK=256, num_warps=4)
    return router, latent


class DecodeFront:
    """Decode-sized adapter; native MoE routing, streams and tail stay intact.

    Prepare once after loading weights, before graph capture. The shared
    expert's native event ordering makes this main-stream output visible on
    its auxiliary stream; the native runner waits before reusing the buffer.
    Each layer owns its scratch, and concurrent/DBO forwards are rejected.
    """

    def __init__(self, layer, max_tokens=8, kind="all"):
        from types import MethodType

        if max_tokens < 1 or layer.experts.enable_dbo:
            raise ValueError("requires positive capacity and no DBO")
        if kind not in ("all", "routed"):
            raise ValueError("front kind must be all or routed")
        shared = layer.shared_experts
        weights = (shared.gate_up_proj.weight, layer.gate.weight,
                   layer.routed_expert_down_proj.weight)
        if any(w.dtype != torch.bfloat16 or w.ndim != 2 for w in weights):
            raise ValueError("merged front requires plain BF16 weights; no router downcast")
        if any(getattr(m, "bias", None) is not None for m in (
                shared.gate_up_proj, layer.gate, layer.routed_expert_down_proj)):
            raise ValueError("merged front requires bias-free input projections")
        if (shared.act_fn.beta, shared.act_fn.linear_beta) != (4.0, 25.0):
            raise ValueError("unexpected K3 SiTU constants")
        if not all(w.shape[1] == weights[0].shape[1] for w in weights):
            raise ValueError("incompatible input projection dimensions")
        if layer.experts.shared_experts._layer is not shared:
            raise ValueError("native runner must retain this shared-expert module")
        self.layer, self.max_tokens, self.enabled, self.kind = layer, max_tokens, False, kind
        self.packed = torch.cat(weights if kind == "all" else weights[1:], dim=0).contiguous()
        device = self.packed.device
        self.projection = torch.empty(max_tokens, self.packed.size(0), device=device, dtype=torch.float32)
        self.shared = torch.empty(max_tokens, weights[0].size(0) // 2, device=device, dtype=torch.bfloat16)
        self.router = torch.empty(max_tokens, weights[1].size(0), device=device, dtype=torch.float32)
        self.latent = torch.empty(max_tokens, weights[2].size(0), device=device, dtype=torch.bfloat16)
        self.native_forward, self.native_shared = layer.forward, shared.forward

        def forward(module, hidden_states):
            n = hidden_states.size(0)
            if not self.enabled or n > self.max_tokens:
                return self.native_forward(hidden_states)
            _, logits, latent = self.project(hidden_states)
            # Passing original input separately skips the routed down-proj,
            # without changing output dimensions or the native sharded tail.
            return module.experts(hidden_states=latent, router_logits=logits,
                                  shared_experts_input=hidden_states)

        def shared_forward(module, hidden_states):
            n = hidden_states.size(0)
            if not self.enabled or n > self.max_tokens or self.kind == "routed":
                return self.native_shared(hidden_states)
            return module.down_proj(self.shared[:n])[0]

        layer.forward = MethodType(forward, layer)
        shared.forward = MethodType(shared_forward, shared)

    def project(self, x):
        n = x.size(0)
        if n > self.max_tokens:
            raise ValueError("decode input exceeds prepared scratch capacity")
        if self.kind == "routed":
            router, latent = merged_routed_front(x, self.packed, self.projection[:n],
                                               self.router[:n], self.latent[:n])
            return None, router, latent
        return merged_front(x, self.packed, self.projection[:n], self.shared[:n],
                            self.router[:n], self.latent[:n])

    @property
    def extra_bytes(self):
        return sum(t.numel() * t.element_size() for t in (
            self.packed, self.projection, self.shared, self.router, self.latent))


class EarlySharedFront:
    """Start native shared-expert work before the two routed projections.

    Reuses the native auxiliary stream, input/output events, output buffer,
    and wait. The runner's later enqueue is suppressed only for this same
    synchronous invocation. No math, weights, or device storage changes.
    A backend that handles overlap internally is left alone. DBO/reentrant
    calls are unsupported, as in DecodeFront.
    """

    kind = "early"
    extra_bytes = 0

    def __init__(self, layer, max_tokens=8):
        from types import MethodType
        if max_tokens < 1 or layer.experts.enable_dbo:
            raise ValueError("requires positive capacity and no DBO")
        self.layer, self.max_tokens, self.enabled = layer, max_tokens, False
        self.native_forward = layer.forward
        shared = layer.experts.shared_experts
        self.native_enqueue = shared.maybe_forward_async
        self.pending, self.enqueued, self.capture_enqueues = False, False, 0
        self.record_capture = False

        def enqueue(module, hidden):
            if self.pending:
                return self.enqueued
            return self.native_enqueue(hidden)

        def forward(module, hidden):
            if not self.enabled or hidden.size(0) > self.max_tokens:
                return self.native_forward(hidden)
            if self.pending:
                raise RuntimeError("early shared-expert front is not reentrant")
            self.enqueued = self.native_enqueue(hidden)
            if self.record_capture and self.enqueued:
                self.capture_enqueues += 1
            self.pending = True
            try:
                return self.native_forward(hidden)
            finally:
                self.pending = False

        layer.forward = MethodType(forward, layer)
        shared.maybe_forward_async = MethodType(enqueue, shared)

    def project(self, hidden):
        # Independent branch check API; the actual forward keeps these calls
        # in the native model/runner, with identical dtypes and rounding.
        shared = self.layer.shared_experts
        return (shared.act_fn(shared.gate_up_proj(hidden)[0]),
                self.layer.gate(hidden)[0],
                self.layer.routed_expert_down_proj(hidden)[0])
