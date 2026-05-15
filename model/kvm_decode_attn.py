from __future__ import annotations

import os
from typing import Any, cast

import torch

import triton  # pyright: ignore[reportMissingImports]
import triton.language as tl  # pyright: ignore[reportMissingImports]


def calc_initial_state(k_norope_normalized_gated, v_gated, chunk_len):
    s_k = k_norope_normalized_gated[:, :, :chunk_len]
    s_v = v_gated[:, :, :chunk_len]
    return s_k, s_v


# ---------------------------------------------------------------------------
# Helper: next power-of-two, clamped to [min_val, max_val]
# ---------------------------------------------------------------------------
def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 1 else 1


def _state_bank_smem_estimate_bytes(
    *, block_s: int, block_k: int, block_v: int, block_o: int
) -> int:
    # Conservative estimate of kernel-shared-memory demand for tile-local fp32 buffers.
    # Includes temporary key/value/state tile storage and score-reduction scalars.
    return 4 * (block_s * (2 * block_k + block_v) + block_s + 2 * block_o)


def _state_bank_smem_budget_bytes() -> int:
    return int(os.environ.get("KVM_STATE_BANK_SMEM_BUDGET_BYTES", "65536"))


def _state_bank_cfg_fits_smem(
    *, block_o: int, block_s: int, block_k: int, block_v: int
) -> bool:
    return (
        _state_bank_smem_estimate_bytes(
            block_o=block_o,
            block_s=block_s,
            block_k=block_k,
            block_v=block_v,
        )
        <= _state_bank_smem_budget_bytes()
    )


_STATE_BANK_AUTOTUNE_CACHE: dict[
    tuple[int, int, int, int, int, int, int, int, torch.dtype],
    tuple[int, int, int, int],
] = {}
_STATE_BANK_AUTOTUNE_PRINTED: set[
    tuple[int, int, int, int, int, int, int, int, torch.dtype]
] = set()


def _state_bank_launch_cfg(
    chunk_len: int, *, k_dim: int, v_dim: int
) -> tuple[int, int, int, int]:
    env_cfg = os.environ.get("KVM_STATE_BANK_LAUNCH_CFG")
    if env_cfg:
        parts = env_cfg.split(",")
        if len(parts) == 4:
            try:
                block_o, block_s, num_warps, num_stages = (
                    int(parts[0]),
                    int(parts[1]),
                    int(parts[2]),
                    int(parts[3]),
                )
                if min(block_o, block_s, num_warps, num_stages) >= 1:
                    block_o = min(block_o, chunk_len)
                    block_s = min(block_s, chunk_len)
                    block_k = _next_pow2(k_dim)
                    block_v = _next_pow2(v_dim)
                    if _state_bank_cfg_fits_smem(
                        block_o=block_o,
                        block_s=block_s,
                        block_k=block_k,
                        block_v=block_v,
                    ):
                        return (
                            block_o,
                            block_s,
                            num_warps,
                            num_stages,
                        )
                    print(
                        f"[kvm_state_bank_cfg] overriding env config {env_cfg} "
                        f"because smem estimate {_state_bank_smem_estimate_bytes(block_o=block_o, block_s=block_s, block_k=block_k, block_v=block_v)} "
                        f"> {_state_bank_smem_budget_bytes()}"
                    )
            except ValueError:
                pass

    if chunk_len >= 256:
        return 32, 32, 4, 2
    if chunk_len >= 64:
        return 16, 32, 8, 3
    return 16, 16, 4, 2


def _state_bank_autotune_candidates(
    chunk_len: int, k_dim: int, v_dim: int
) -> list[tuple[int, int, int, int]]:
    if chunk_len >= 256:
        candidates = [
            (256, 32, 4, 2),
        ]
    else:
        candidates = [
            (16, 16, 4, 2),
            (16, 8, 4, 2),
            (8, 16, 4, 2),
        ]
    return candidates


def _state_bank_launch_args(
    *,
    B: int,
    H: int,
    chunk_len: int,
    overflow_base: int,
    n_steps: int,
    sink_len: int,
    ln_eps: float,
    block_o: int,
    block_s: int,
    num_warps: int,
    num_stages: int,
    k_norope_normalized_gated: torch.Tensor,
    v_gated: torch.Tensor,
    ln_dk_weight: torch.Tensor,
    ln_dk_bias: torch.Tensor,
    state_head_temp: torch.Tensor,
    state_k_bank: torch.Tensor,
    state_v_bank: torch.Tensor,
) -> None:
    grid = (B * H,)
    K = int(k_norope_normalized_gated.size(-1))
    V = int(v_gated.size(-1))
    _kvm_state_bank_kernel[grid](
        k_norope_normalized_gated,
        k_norope_normalized_gated.stride(0),
        k_norope_normalized_gated.stride(1),
        k_norope_normalized_gated.stride(2),
        k_norope_normalized_gated.stride(3),
        v_gated,
        v_gated.stride(0),
        v_gated.stride(1),
        v_gated.stride(2),
        v_gated.stride(3),
        ln_dk_weight,
        ln_dk_bias,
        state_head_temp,
        state_k_bank,
        state_k_bank.stride(0),
        state_k_bank.stride(1),
        state_k_bank.stride(2),
        state_k_bank.stride(3),
        state_v_bank,
        state_v_bank.stride(0),
        state_v_bank.stride(1),
        state_v_bank.stride(2),
        state_v_bank.stride(3),
        H,
        overflow_base,
        CHUNK_LEN=chunk_len,
        NUM_O_BLOCKS=(chunk_len + block_o - 1) // block_o,
        BLOCK_O=block_o,
        BLOCK_S=block_s,
        N_STEPS=n_steps,
        SINK_LEN=sink_len,
        K=K,
        VDIM=V,
        BLOCK_K=_next_pow2(K),
        BLOCK_V=_next_pow2(V),
        ln_eps=ln_eps,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _state_bank_autotune_cfg(
    *,
    B: int,
    H: int,
    chunk_len: int,
    overflow_base: int,
    n_steps: int,
    device: torch.device,
    k_norope_normalized_gated: torch.Tensor,
    v_gated: torch.Tensor,
    ln_dk_weight: torch.Tensor,
    ln_dk_bias: torch.Tensor,
    state_head_temp: torch.Tensor,
    out_dtype: torch.dtype,
    ln_eps: float,
    sink_len: int,
) -> tuple[int, int, int, int]:
    assert chunk_len > 0
    K = k_norope_normalized_gated.size(-1)
    V = int(v_gated.size(-1))
    key = (
        -1,  # int(device.index) if device.index is not None else -1,
        B,
        H,
        chunk_len,
        int(K),
        int(V),
        n_steps,
        sink_len,
        out_dtype,
    )
    if key in _STATE_BANK_AUTOTUNE_CACHE:
        return _STATE_BANK_AUTOTUNE_CACHE[key]

    s_k, s_v = calc_initial_state(k_norope_normalized_gated, v_gated, chunk_len)
    tail_len = chunk_len * (n_steps if n_steps > 0 else 1)
    assert tail_len > 0
    state_k_bank = torch.empty(B, H, tail_len, K, device=device, dtype=out_dtype)
    state_v_bank = torch.empty(B, H, tail_len, V, device=device, dtype=out_dtype)
    state_k_bank[:, :, :chunk_len].copy_(s_k)
    state_v_bank[:, :, :chunk_len].copy_(s_v)

    best_cfg = _state_bank_launch_cfg(chunk_len, k_dim=int(K), v_dim=int(V))
    best_ms = float("inf")

    candidates = _state_bank_autotune_candidates(chunk_len, int(K), int(V))
    if not candidates:
        print(
            "[kvm_state_bank_autotune] no candidate fits SMEM budget; "
            "using default small config"
        )
        candidates = [(16, 16, 4, 2)]
    for block_o, block_s, num_warps, num_stages in candidates:
        print(
            f"Autotune is testing candidate {block_o} {block_s} {num_warps} {num_stages}..."
        )
        block_o = min(block_o, chunk_len)
        block_s = min(block_s, chunk_len)
        state_k_bank[:, :, :chunk_len].copy_(s_k)
        state_v_bank[:, :, :chunk_len].copy_(s_v)
        print("Pre-test beginning")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _state_bank_launch_args(
            B=B,
            H=H,
            chunk_len=chunk_len,
            overflow_base=overflow_base,
            n_steps=n_steps,
            sink_len=sink_len,
            ln_eps=ln_eps,
            block_o=block_o,
            block_s=block_s,
            num_warps=num_warps,
            num_stages=num_stages,
            k_norope_normalized_gated=k_norope_normalized_gated,
            v_gated=v_gated,
            ln_dk_weight=ln_dk_weight,
            ln_dk_bias=ln_dk_bias,
            state_head_temp=state_head_temp,
            state_k_bank=state_k_bank,
            state_v_bank=state_v_bank,
        )
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = float(start.elapsed_time(end))
        print(f"{elapsed_ms:.1f}ms")

        print("Full test beginning")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        reps = 2
        start.record()
        for _ in range(reps):
            _state_bank_launch_args(
                B=B,
                H=H,
                chunk_len=chunk_len,
                overflow_base=overflow_base,
                n_steps=n_steps,
                sink_len=sink_len,
                ln_eps=ln_eps,
                block_o=block_o,
                block_s=block_s,
                num_warps=num_warps,
                num_stages=num_stages,
                k_norope_normalized_gated=k_norope_normalized_gated,
                v_gated=v_gated,
                ln_dk_weight=ln_dk_weight,
                ln_dk_bias=ln_dk_bias,
                state_head_temp=state_head_temp,
                state_k_bank=state_k_bank,
                state_v_bank=state_v_bank,
            )
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = float(start.elapsed_time(end) / reps)
        print(f"{elapsed_ms:.1f}ms")
        if elapsed_ms < best_ms:
            best_ms = elapsed_ms
            best_cfg = (block_o, block_s, num_warps, num_stages)

    key_name = (
        f"B={B} H={H} chunk_len={chunk_len} n_steps={n_steps} K={int(K)} V={V} "
        f"sink_len={sink_len} dtype={out_dtype} dev={int(device.index) if device.index is not None else -1}"
    )
    if key not in _STATE_BANK_AUTOTUNE_PRINTED:
        print(
            f"[kvm_state_bank_autotune] {key_name} best_cfg={best_cfg} ms={best_ms:.3f}"
        )
        _STATE_BANK_AUTOTUNE_PRINTED.add(key)

    _STATE_BANK_AUTOTUNE_CACHE[key] = best_cfg
    return best_cfg


# ---------------------------------------------------------------------------
# Python-level dispatch
# ---------------------------------------------------------------------------


@triton.jit
def _kvm_state_bank_kernel(
    KN_ptr,
    stride_knb,
    stride_knh,
    stride_knt,
    stride_knk,
    V_ptr,
    stride_vb,
    stride_vh,
    stride_vt,
    stride_vv,
    # G_ptr, stride_gb, stride_gh, stride_gt,
    LN_W_ptr,
    LN_B_ptr,
    SHT_ptr,
    # BANKK_ptr, stride_bkb, stride_bkh, stride_bkt, stride_bkk,
    STATEK_ptr,
    stride_skb,
    stride_skh,
    stride_sks,
    stride_skk,
    STATEV_ptr,
    stride_svb,
    stride_svh,
    stride_svs,
    stride_svv,
    NH,
    OVERFLOW_BASE,
    CHUNK_LEN: tl.constexpr,
    N_STEPS: tl.constexpr,
    SINK_LEN: tl.constexpr,
    K: tl.constexpr,
    VDIM: tl.constexpr,
    NUM_O_BLOCKS: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    ln_eps: tl.constexpr,
):
    bh_pid = tl.program_id(0)
    b = bh_pid // NH
    h = bh_pid % NH

    k_offs = tl.arange(0, BLOCK_K)
    v_offs = tl.arange(0, BLOCK_V)
    s_offs_tile = tl.arange(0, BLOCK_S)
    o_offs = tl.arange(0, CHUNK_LEN)  # BLOCK_O)
    k_mask = k_offs < K
    v_mask = v_offs < VDIM

    ln_w = tl.load(LN_W_ptr + k_offs, mask=k_mask, other=1.0).to(tl.float32)
    ln_b = tl.load(LN_B_ptr + k_offs, mask=k_mask, other=0.0).to(tl.float32)
    # sht = tl.load(SHT_ptr + h).to(tl.float32)

    kn_base = KN_ptr + b * stride_knb + h * stride_knh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    # g_base = G_ptr + b * stride_gb + h * stride_gh
    statek_base = STATEK_ptr + b * stride_skb + h * stride_skh
    statev_base = STATEV_ptr + b * stride_svb + h * stride_svh
    # bankk_base = BANKK_ptr + b * stride_bkb + h * stride_bkh

    # Process each overflow chunk in sequence; one iteration updates one CHUNK_LEN window.
    for step in tl.range(0, N_STEPS):
        # bank_start = step * CHUNK_LEN
        overflow_start = OVERFLOW_BASE + step * CHUNK_LEN

        o_offs_block = o_offs  # + o_start
        o_mask = o_offs_block < CHUNK_LEN
        ok_ptrs = (
            kn_base
            + (overflow_start + o_offs_block)[:, None] * stride_knt
            + k_offs[None, :] * stride_knk
        )
        ok = tl.load(
            ok_ptrs, mask=o_mask[:, None] & k_mask[None, :], other=0.0
        )  # .to(tl.float32) # [CHUNK_LEN, K]

        ov_ptrs = (
            v_base
            + (overflow_start + o_offs_block)[:, None] * stride_vt
            + v_offs[None, :] * stride_vv
        )
        # #ov = tl.load(ov_ptrs, mask=valid_best[:, None] & v_mask[None, :], other=0.0)#.to(tl.float32)
        ov = tl.load(
            ov_ptrs, mask=o_mask[:, None] & v_mask[None, :], other=0.0
        )  # .to(tl.float32) # [CHUNK_LEN, K]

        # pre-apply gating - since it's per overflow token this does not impact relative scores across state tokens
        # og_ptrs = g_base + (overflow_start + o_offs_block) * stride_gt
        # og = tl.load(og_ptrs, mask=o_mask, other=0.0).to(tl.float32)

        # ok = (ok * og[:, None]).to(tl.bfloat16)
        # ov = (ov * og[:, None]).to(tl.bfloat16)

        best_state_score = tl.full((CHUNK_LEN,), float("-inf"), tl.float32)
        best_state_idx = tl.full((CHUNK_LEN,), -1, tl.int32)

        # Find the best match state score,index for each overflow token:
        #  Visit every state tile, scoring vs the overflow tokens
        #  Scan all state tiles and keep the running best score/index per output token.
        for s_start in tl.range(0, CHUNK_LEN, BLOCK_S):
            # Compute pre-normalized state keys for this chunk.
            s_offs = s_start + s_offs_tile
            s_mask = s_offs < CHUNK_LEN
            # bank_offs = bank_start + s_offs

            sk_ptrs = (
                statek_base
                + s_offs[:, None] * stride_sks
                + k_offs[None, :] * stride_skk
            )
            sk = tl.load(sk_ptrs, mask=s_mask[:, None] & k_mask[None, :], other=0.0).to(
                tl.float32
            )
            sk_mean = tl.sum(sk, axis=1) / K
            sk_c = sk - sk_mean[:, None]
            sk_var = tl.sum(sk_c * sk_c, axis=1) / K
            sk_inv_std = tl.rsqrt(sk_var + ln_eps)
            sk_norm = (
                sk_c * sk_inv_std[:, None] * ln_w[None, :] + ln_b[None, :]
            )  # .to(tl.bfloat16)
            sk_prepared = sk_norm  # * sht
            sk_prepared = sk_prepared

            valid_state = s_mask & (s_offs >= SINK_LEN)

            scores = tl.dot(
                ok.to(tl.bfloat16),
                tl.trans(sk_prepared.to(tl.bfloat16)),
                out_dtype=tl.float32,
            )  # [Overflow, State]
            valid_scores = o_mask[:, None] & valid_state[None, :]
            scores = tl.where(valid_scores, scores, float("-inf"))

            tile_best_state_score = tl.max(scores, axis=1)
            tile_best_rel = tl.argmax(scores, axis=1)
            tile_best_state_idx = (s_start + tile_best_rel).to(tl.int32)
            # tile_best_state_rel_idx = (tile_best_rel).to(tl.int32)

            # Compare this tile's best scores vs the running best scores and update where better.
            better = tile_best_state_score > best_state_score
            best_state_score = tl.where(better, tile_best_state_score, best_state_score)
            best_state_idx = tl.where(
                better, tile_best_state_idx, best_state_idx
            )  # best matching state column index for each overflow token

            best_state_idx = tl.where(o_mask, best_state_idx, -1)

        # Apply all merges for this chunk using the selected best indices.
        for s_start in tl.range(0, CHUNK_LEN, BLOCK_S):
            s_offs = s_start + s_offs_tile
            s_mask = s_offs < CHUNK_LEN

            # Keep this state tile resident and accumulate contributions from all matching overflow rows.
            sk_ptrs = (
                statek_base
                + s_offs[:, None] * stride_sks
                + k_offs[None, :] * stride_skk
            )
            sk_tile = tl.load(
                sk_ptrs, mask=s_mask[:, None] & k_mask[None, :], other=0.0
            ).to(tl.float32)

            sv_ptrs = (
                statev_base
                + s_offs[:, None] * stride_svs
                + v_offs[None, :] * stride_svv
            )
            sv_tile = tl.load(
                sv_ptrs, mask=s_mask[:, None] & v_mask[None, :], other=0.0
            ).to(tl.float32)

            valid_best = o_mask & (best_state_idx >= 0)

            # Find the matching state tile rows for the best indices of the overflow tokens and related scores
            # match = (s_offs[:, None] == best_state_idx[None, :]) & s_mask[:, None] & valid_best[None, :]
            match = tl.where(
                (s_offs[:, None] == best_state_idx[None, :])
                & s_mask[:, None]
                & valid_best[None, :],
                best_state_score,
                0,
            ).to(tl.bfloat16)
            # match_f = match.to(tl.float32)
            sk_tile += tl.dot(match, ok)
            sv_tile += tl.dot(match, ov)  # .to(tl.float32))

            tl.store(sk_ptrs, sk_tile, mask=s_mask[:, None] & k_mask[None, :])
            tl.store(sv_ptrs, sv_tile, mask=s_mask[:, None] & v_mask[None, :])


def kvm_generate_state_banks(
    k_norope_normalized_gated: torch.Tensor,
    v_gated: torch.Tensor,
    s_vlen: torch.Tensor,
    ln_dk_weight: torch.Tensor,
    ln_dk_bias: torch.Tensor,
    state_head_temp: torch.Tensor,
    bswa_len: int,
    chunk_len: int,
    sink_len: int,
    out_dtype: torch.dtype,
    ln_eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, total_t, K = k_norope_normalized_gated.shape
    V = int(v_gated.size(-1))
    tail_len = total_t - bswa_len
    assert tail_len >= 0, f"bswa_len ({bswa_len}) must not exceed total_t ({total_t})"
    assert tail_len % chunk_len == 0, (
        f"tail_len ({tail_len}) must be divisible by chunk_len ({chunk_len})"
    )
    n_steps = tail_len // chunk_len

    s_k, s_v = calc_initial_state(k_norope_normalized_gated, v_gated, chunk_len)

    state_k_bank = torch.empty(
        B, H, tail_len, K, device=v_gated.device, dtype=out_dtype
    )
    state_v_bank = torch.empty(
        B, H, tail_len, V, device=v_gated.device, dtype=out_dtype
    )
    state_k_bank[:, :, :chunk_len].copy_(s_k)
    state_v_bank[:, :, :chunk_len].copy_(s_v)

    if tail_len <= 0:
        return s_k, s_v, state_k_bank[:, :, :0], state_v_bank[:, :, :0]

    if os.environ.get("KVM_STATE_BANK_AUTOTUNE") == "1":
        block_o, block_s, num_warps, num_stages = _state_bank_autotune_cfg(
            B=B,
            H=H,
            chunk_len=chunk_len,
            overflow_base=bswa_len,
            n_steps=n_steps,
            device=v_gated.device,
            k_norope_normalized_gated=k_norope_normalized_gated,
            v_gated=v_gated,
            ln_dk_weight=ln_dk_weight,
            ln_dk_bias=ln_dk_bias,
            state_head_temp=state_head_temp,
            out_dtype=out_dtype,
            ln_eps=ln_eps,
            sink_len=sink_len,
        )
    else:
        block_o, block_s, num_warps, num_stages = _state_bank_launch_cfg(
            chunk_len,
            k_dim=int(K),
            v_dim=V,
        )
    _state_bank_launch_args(
        B=B,
        H=H,
        chunk_len=chunk_len,
        overflow_base=bswa_len,
        n_steps=n_steps,
        sink_len=sink_len,
        ln_eps=ln_eps,
        block_o=block_o,
        block_s=block_s,
        num_warps=num_warps,
        num_stages=num_stages,
        k_norope_normalized_gated=k_norope_normalized_gated,
        v_gated=v_gated,
        ln_dk_weight=ln_dk_weight,
        ln_dk_bias=ln_dk_bias,
        state_head_temp=state_head_temp,
        state_k_bank=state_k_bank,
        state_v_bank=state_v_bank,
    )
    return s_k, s_v, state_k_bank, state_v_bank
