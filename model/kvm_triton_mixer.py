from __future__ import annotations

import argparse
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch.autograd import Function

from .kvm_mixer import SequenceMixer as TorchKVMSequenceMixer


def _choose_dividing_block(size: int, candidates: tuple[int, ...]) -> int:
    for candidate in candidates:
        if candidate <= size and size % candidate == 0:
            return candidate
    raise ValueError(f"no supported block size divides {size}")


class _KvmTritonTrainingFunction(Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q_flat: torch.Tensor,
        bswa_k_flat: torch.Tensor,
        bswa_v_flat: torch.Tensor,
        overflow_k_flat: torch.Tensor,
        overflow_v_flat: torch.Tensor,
        overflow_select_k_flat: torch.Tensor,
        overflow_append_k_flat: torch.Tensor,
        overflow_append_v_flat: torch.Tensor,
        overflow_merge_k_flat: torch.Tensor,
        overflow_merge_v_flat: torch.Tensor,
        initial_k_flat: torch.Tensor,
        initial_v_flat: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        state_temperature: torch.Tensor,
        front_temperature: torch.Tensor,
        triton_args: argparse.Namespace,
        schedule: Any,
    ) -> torch.Tensor:
        from model.kernels.kvm_triton_training_kernels import (
            build_prefill_forward,
        )

        if not q_flat.is_cuda:
            raise RuntimeError("kvm_triton_mixer requires CUDA/ROCm tensors")

        forward = build_prefill_forward(
            triton_args,
            schedule,
            q_flat,
            overflow_k_flat,
            overflow_v_flat,
            bswa_k_flat,
            bswa_v_flat,
            ln_weight,
            ln_bias,
            state_temperature,
            front_temperature,
            initial_k_flat=initial_k_flat,
            initial_v_flat=initial_v_flat,
            overflow_select_k_flat=overflow_select_k_flat,
            overflow_append_k_flat=overflow_append_k_flat,
            overflow_append_v_flat=overflow_append_v_flat,
            overflow_merge_k_flat=overflow_merge_k_flat,
            overflow_merge_v_flat=overflow_merge_v_flat,
        )
        ctx.triton_args = triton_args
        ctx.schedule = schedule
        ctx.save_for_backward(
            q_flat,
            bswa_k_flat,
            bswa_v_flat,
            overflow_k_flat,
            overflow_v_flat,
            overflow_select_k_flat,
            overflow_append_k_flat,
            overflow_append_v_flat,
            overflow_merge_k_flat,
            overflow_merge_v_flat,
            initial_k_flat,
            initial_v_flat,
            ln_weight,
            ln_bias,
            state_temperature,
            front_temperature,
            forward["out"],
            forward["lse"],
            forward["state_k"],
            forward["state_v"],
            forward["state_k_attn"],
            forward["state_v_attn"],
            forward["state_vlen"],
            forward["append_pos_by_token"],
            forward["best_idx_by_token"],
            forward["undo_k_by_token"],
            forward["undo_v_by_token"],
        )
        return forward["out"]

    @staticmethod
    def backward(ctx, dout: torch.Tensor):  # type: ignore[override]
        from model.kernels.kvm_triton_training_kernels import (
            run_training_backward_reconstruct_live_state,
        )

        (
            q_flat,
            bswa_k_flat,
            bswa_v_flat,
            overflow_k_flat,
            overflow_v_flat,
            overflow_select_k_flat,
            overflow_append_k_flat,
            overflow_append_v_flat,
            overflow_merge_k_flat,
            overflow_merge_v_flat,
            initial_k_flat,
            initial_v_flat,
            ln_weight,
            ln_bias,
            state_temperature,
            front_temperature,
            out,
            lse,
            state_k,
            state_v,
            state_k_attn,
            state_v_attn,
            state_vlen,
            append_pos_by_token,
            best_idx_by_token,
            undo_k_by_token,
            undo_v_by_token,
        ) = ctx.saved_tensors
        saved_forward = {
            "out": out,
            "lse": lse,
            "state_k": state_k,
            "state_v": state_v,
            "state_k_attn": state_k_attn,
            "state_v_attn": state_v_attn,
            "state_vlen": state_vlen,
            "append_pos_by_token": append_pos_by_token,
            "best_idx_by_token": best_idx_by_token,
            "undo_k_by_token": undo_k_by_token,
            "undo_v_by_token": undo_v_by_token,
        }
        result = run_training_backward_reconstruct_live_state(
            ctx.triton_args,
            ctx.schedule,
            q_flat,
            bswa_k_flat,
            bswa_v_flat,
            overflow_k_flat,
            ln_weight,
            ln_bias,
            dout.contiguous(),
            state_temperature,
            front_temperature,
            initial_k_flat=initial_k_flat,
            initial_v_flat=initial_v_flat,
            overflow_select_k_flat=overflow_select_k_flat,
            overflow_append_k_flat=overflow_append_k_flat,
            overflow_append_v_flat=overflow_append_v_flat,
            overflow_merge_k_flat=overflow_merge_k_flat,
            overflow_merge_v_flat=overflow_merge_v_flat,
            saved_forward=saved_forward,
        )
        return (
            result["dq"].to(q_flat.dtype),
            result["d_bswa_k"].to(bswa_k_flat.dtype),
            result["d_bswa_v"].to(bswa_v_flat.dtype),
            result["d_overflow_k"].to(overflow_k_flat.dtype),
            torch.zeros_like(overflow_v_flat),
            torch.zeros_like(overflow_select_k_flat),
            result["d_append_k"].to(overflow_append_k_flat.dtype),
            result["d_append_v"].to(overflow_append_v_flat.dtype),
            result["d_merge_k"].to(overflow_merge_k_flat.dtype),
            result["d_merge_v"].to(overflow_merge_v_flat.dtype),
            result["d_initial_k"].to(initial_k_flat.dtype),
            result["d_initial_v"].to(initial_v_flat.dtype),
            result["d_ln_weight"].to(ln_weight.dtype),
            result["d_ln_bias"].to(ln_bias.dtype),
            result["d_state_temperature"].to(state_temperature.dtype),
            result["d_front_temperature"].to(front_temperature.dtype),
            None,
            None,
        )


class SequenceMixer(TorchKVMSequenceMixer):
    """KVM mixer backed by the current Triton training/prefill kernels.

    The integrated path uses the simplified Triton semantics consistently:
    sub-block append quotas, merge-before-append update ordering, and rounded
    fp16-delta state/normcache updates.

    Other intentional differences from classic `kvm_mixer.py`:
    - `kvm_use_vlens=1` is required;
    - prefill inputs are padded to the next chunk for Triton launch, but the
      schedule and returned outputs use the real ragged length;
    - training treats append/merge routing as fixed non-differentiable metadata.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        if not config.kvm_use_vlens:
            raise ValueError("kvm_triton_mixer currently requires kvm_use_vlens=1")
        if self.state_budget_mode == "kvm_saturation":
            raise ValueError(
                "kvm_triton_mixer currently supports fixed and power_law state schedules only"
            )

        self.triton_sub_block = _choose_dividing_block(
            self.chunk_len, (128, 64, 32, 16)
        )
        self.triton_attn_block = _choose_dividing_block(
            self.chunk_len, (64, 128, 32, 16)
        )
        self.triton_state_chunk = 16
        self.triton_group_chunks = 12
        self.triton_update_token_block = min(8, self.triton_sub_block)
        if self.triton_sub_block % self.triton_update_token_block:
            raise ValueError("internal Triton update token block must divide sub_block")

    def _triton_state_schedule_params(self) -> tuple[float, float, int]:
        if self.state_budget_mode == "fixed":
            return 0.0, 1.0, self.state_min_len
        if self.state_budget_mode == "power_law":
            return (
                float(self.state_growth_factor),
                float(self.state_growth_exponent),
                int(self.state_min_len),
            )
        raise ValueError(f"unsupported state_budget_mode={self.state_budget_mode!r}")

    def _make_triton_args(self, batch_size: int, q_len: int) -> argparse.Namespace:
        padded_q_len = int(math.ceil(q_len / self.chunk_len) * self.chunk_len)
        schedule_factor, schedule_exponent, state_min_len = (
            self._triton_state_schedule_params()
        )
        return argparse.Namespace(
            batch=batch_size,
            q_heads=self.num_attention_heads,
            kv_heads=self.num_key_value_heads,
            q_len=padded_q_len,
            logical_q_len=q_len,
            initial_state_len=min(q_len, self.chunk_len),
            # Let the Triton helper size temporary buffers from the actual
            # prefill schedule. The config max can be millions of slots and is
            # only a capacity cap, not the number of active state rows.
            max_state_len=0,
            state_chunk=self.triton_state_chunk,
            group_chunks=self.triton_group_chunks,
            update_token_block=self.triton_update_token_block,
            macro_block=self.chunk_len,
            bswa_chunks=self.n_bswa_chunks,
            sub_block=self.triton_sub_block,
            attn_block=self.triton_attn_block,
            schedule_factor=schedule_factor,
            schedule_exponent=schedule_exponent,
            state_min_len=state_min_len,
            state_round_down=self.state_round_down,
            dim=self.d_qk_head,
            value_dim=self.d_v_head,
            sink_len=self.sink_len,
            ln_eps=float(self.ln_s_k.eps),
            scan_num_warps=8,
            update_num_warps=8,
            attn_num_warps=4,
            waves_per_eu=1,
            scan_waves_per_eu=0,
            update_waves_per_eu=0,
            attn_waves_per_eu=1,
            q_head_loop_unroll_factor=4,
            skip_temperature_grad=False,
            skip_temperature_atomic=False,
            temperature_grad_backend="atomic",
            update_grad_backend="triton",
            attn_grad_backend="kv-owned",
            undo_mode="stash",
            cache_from_rounded_state=True,
            reconstruct_live_state_backward=True,
            fuse_restore_refresh=True,
            fuse_state_dkdv_raw=False,
            append_policy="subblock_quota",
            merge_order="merge_before_append",
        )

    def _make_schedule(self, triton_args: argparse.Namespace):
        from model.kernels.kvm_triton_training_kernels import make_schedule

        return make_schedule(triton_args)

    def _flatten_qkv_for_triton(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        merge_gate: torch.Tensor,
        padded_len: int,
    ) -> dict[str, torch.Tensor]:
        batch_size, q_heads, q_len, _ = q.shape
        kv_heads = int(k.size(1))
        if q_heads != self.num_attention_heads:
            raise AssertionError("KVM Triton query head count mismatch.")
        if kv_heads != self.num_key_value_heads:
            raise AssertionError("KVM Triton key/value head count mismatch.")
        pad_len = padded_len - q_len
        if pad_len < 0:
            raise ValueError("padded_len must be >= q_len")
        if pad_len:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            merge_gate = F.pad(merge_gate, (0, 0, 0, pad_len), value=1.0)

        prepared_k = self._prepare_state_update_k(k)
        key_gate = merge_gate if self.config.kvm_use_merge_gate_keys else 1.0
        value_gate = merge_gate if self.config.kvm_use_merge_gate_values else 1.0
        merge_k = (prepared_k * key_gate).to(prepared_k.dtype)
        merge_v = (v * value_gate).to(v.dtype)
        append_k = merge_k if self.config.kvm_apply_merge_gate_to_appends else prepared_k
        append_v = merge_v if self.config.kvm_apply_merge_gate_to_appends else v
        if self.config.kvm_apply_merge_gate_to_initial_state:
            initial_k = (prepared_k * merge_gate).to(prepared_k.dtype)
            initial_v = (v * merge_gate).to(v.dtype)
        else:
            initial_k = prepared_k
            initial_v = v

        def flatten_q(x: torch.Tensor, dim: int) -> torch.Tensor:
            return x.reshape(batch_size * q_heads, padded_len, dim).contiguous()

        def flatten_kv(x: torch.Tensor, dim: int) -> torch.Tensor:
            return x.reshape(batch_size * kv_heads, padded_len, dim).contiguous()

        return {
            "q": flatten_q(q, self.d_qk_head),
            "bswa_k": flatten_kv(k, self.d_qk_head),
            "bswa_v": flatten_kv(v, self.d_v_head),
            "select_k": flatten_kv(prepared_k, self.d_qk_head),
            "append_k": flatten_kv(append_k, self.d_qk_head),
            "append_v": flatten_kv(append_v, self.d_v_head),
            "merge_k": flatten_kv(merge_k, self.d_qk_head),
            "merge_v": flatten_kv(merge_v, self.d_v_head),
            "initial_k": flatten_kv(initial_k, self.d_qk_head),
            "initial_v": flatten_kv(initial_v, self.d_v_head),
        }

    def _head_temperatures(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.kvm_use_head_temps:
            return (
                self.state_head_temp.float(),
                self.front_head_temp.float(),
            )
        ones = torch.ones(self.num_attention_heads, device=device, dtype=torch.float32)
        return ones, ones

    def _triton_forward_prefill_raw(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        merge_gate: torch.Tensor,
        triton_args: argparse.Namespace,
        schedule,
    ) -> dict[str, torch.Tensor]:
        from model.kernels.kvm_triton_training_kernels import (
            build_prefill_forward,
        )

        flat = self._flatten_qkv_for_triton(q, k, v, merge_gate, triton_args.q_len)
        state_temperature, front_temperature = self._head_temperatures(q.device)
        return build_prefill_forward(
            triton_args,
            schedule,
            flat["q"],
            flat["merge_k"],
            flat["merge_v"],
            flat["bswa_k"],
            flat["bswa_v"],
            self.ln_s_k.weight.float(),
            self.ln_s_k.bias.float(),
            state_temperature,
            front_temperature,
            initial_k_flat=flat["initial_k"],
            initial_v_flat=flat["initial_v"],
            overflow_select_k_flat=flat["select_k"],
            overflow_append_k_flat=flat["append_k"],
            overflow_append_v_flat=flat["append_v"],
            overflow_merge_k_flat=flat["merge_k"],
            overflow_merge_v_flat=flat["merge_v"],
        )

    def _triton_prefill_out(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        merge_gate: torch.Tensor,
        triton_args: argparse.Namespace,
        schedule,
    ) -> torch.Tensor:
        batch_size, q_heads, _, _ = q.shape
        flat = self._flatten_qkv_for_triton(q, k, v, merge_gate, triton_args.q_len)
        state_temperature, front_temperature = self._head_temperatures(q.device)
        gates_enabled = (
            self.config.kvm_use_merge_gate_keys or self.config.kvm_use_merge_gate_values
        )
        if not gates_enabled:
            raise ValueError("kvm_triton_mixer training assumes merge gates exist")
        out_flat = _KvmTritonTrainingFunction.apply(
            flat["q"],
            flat["bswa_k"],
            flat["bswa_v"],
            flat["select_k"],
            flat["merge_v"],
            flat["select_k"],
            flat["append_k"],
            flat["append_v"],
            flat["merge_k"],
            flat["merge_v"],
            flat["initial_k"],
            flat["initial_v"],
            self.ln_s_k.weight.float(),
            self.ln_s_k.bias.float(),
            state_temperature,
            front_temperature,
            triton_args,
            schedule,
        )
        return out_flat.reshape(batch_size, q_heads, triton_args.q_len, self.d_v_head)

    def forward_prefill(
        self,
        q,
        k,
        v,
        merge_gate,
        v_first,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        del v_first, position_embeddings, kwargs
        if attention_mask is not None:
            raise ValueError("kvm_triton_mixer does not support attention_mask")

        batch_size, _, prefill_len, _ = q.size()
        triton_args = self._make_triton_args(batch_size, int(prefill_len))
        schedule = self._make_schedule(triton_args)

        needs_grad = torch.is_grad_enabled() and (
            q.requires_grad or k.requires_grad or v.requires_grad
        )
        if past_key_values is not None and needs_grad:
            raise ValueError(
                "kvm_triton_mixer does not support autograd prefill while updating "
                "past_key_values"
            )
        gates_enabled = (
            self.config.kvm_use_merge_gate_keys or self.config.kvm_use_merge_gate_values
        )
        if not gates_enabled:
            raise ValueError("kvm_triton_mixer prefill assumes merge gates exist")

        if needs_grad:
            out = self._triton_prefill_out(q, k, v, merge_gate, triton_args, schedule)
            out = out[:, :, :prefill_len, :]
            y = out.transpose(1, 2).contiguous().view(batch_size, prefill_len, -1)
            return self.c_proj(y)

        forward = self._triton_forward_prefill_raw(
            q, k, v, merge_gate, triton_args, schedule
        )
        out = forward["out"].reshape(
            batch_size, self.num_attention_heads, triton_args.q_len, self.d_v_head
        )[:, :, :prefill_len, :]
        y = out.transpose(1, 2).contiguous().view(batch_size, prefill_len, -1)
        y = self.c_proj(y)

        if past_key_values is not None:
            final_state_len = int(schedule.final_state_len)
            bswa_begin = self._bswa_begin_for_total_len(prefill_len)
            state_coverage_len = int(schedule.final_state_coverage_len)
            expected_state_coverage_len = max(triton_args.initial_state_len, bswa_begin)
            if state_coverage_len != expected_state_coverage_len:
                raise AssertionError(
                    "KVM Triton prefill state progression drifted from cache bookkeeping."
                )
            past_key_values.update(
                self.layer_idx,
                offset=prefill_len,
                states_dict={
                    self._CACHE_S_K: forward["state_k"][
                        :, :final_state_len
                    ].reshape(
                        batch_size,
                        self.num_key_value_heads,
                        final_state_len,
                        self.d_qk_head,
                    ),
                    self._CACHE_S_V: forward["state_v"][
                        :, :final_state_len
                    ].reshape(
                        batch_size,
                        self.num_key_value_heads,
                        final_state_len,
                        self.d_v_head,
                    ),
                    self._CACHE_S_VLEN: forward["state_vlen"][
                        :, :final_state_len
                    ].reshape(
                        batch_size, self.num_key_value_heads, final_state_len, 1
                    ),
                    self._CACHE_STATE_COVERAGE_LEN: state_coverage_len,
                    self._CACHE_BSWA_BEGIN: bswa_begin,
                    self._CACHE_BSWA_K: k[:, :, bswa_begin:, :],
                    self._CACHE_BSWA_V: v[:, :, bswa_begin:, :],
                    self._CACHE_BSWA_MERGE_GATE: torch.ones(
                        batch_size,
                        self.num_key_value_heads,
                        prefill_len - bswa_begin,
                        1,
                        device=k.device,
                        dtype=torch.float32,
                    )
                    if not (
                        self.config.kvm_use_merge_gate_keys
                        or self.config.kvm_use_merge_gate_values
                    )
                    else merge_gate[:, :, bswa_begin:, :],
                },
            )

        return y

    def _make_generation_update_schedule(
        self,
        current_state_len: int,
        state_after: int,
        n_append: int,
        overflow_len: int,
    ):
        from model.kernels.kvm_triton_training_kernels import MixerPrefillSchedule

        if overflow_len != self.chunk_len:
            raise AssertionError(
                "KVM Triton decode can only materialize one full overflow chunk per update."
            )
        return MixerPrefillSchedule(
            before_by_macro=torch.tensor([current_state_len], dtype=torch.int32),
            after_by_macro=torch.tensor([state_after], dtype=torch.int32),
            n_append_by_macro=torch.tensor([n_append], dtype=torch.int32),
            valid_update_by_macro=torch.tensor([1], dtype=torch.int32),
            attention_state_len_by_macro=torch.tensor(
                [current_state_len], dtype=torch.int32
            ),
            front_len=overflow_len,
            initial_state_len=current_state_len,
            final_state_len=state_after,
            final_state_coverage_len=0,
        )

    def _update_state_from_overflow_tokens(
        self,
        s_k: torch.Tensor,
        s_v: torch.Tensor,
        s_vlen: torch.Tensor,
        overflow_k: torch.Tensor,
        overflow_v: torch.Tensor,
        merge_gate: torch.Tensor,
        ctx_len: int,
        available_context: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from model.kernels.kvm_triton_training_kernels import (
            allocate_work_buffers,
            run_forward_state_update,
        )

        overflow_len = int(overflow_k.size(2))
        if overflow_len == 0:
            return s_k, s_v, s_vlen
        if overflow_len != self.chunk_len:
            raise AssertionError(
                "KVM Triton decode can only update from one full overflow chunk."
            )
        if not overflow_k.is_cuda:
            raise RuntimeError("kvm_triton_mixer generation updates require CUDA/ROCm")

        overflow_k_ungated = self._prepare_state_update_k(overflow_k)
        overflow_v_ungated = overflow_v
        if self.config.kvm_use_merge_gate_keys:
            overflow_k_gated = (overflow_k_ungated * merge_gate).to(
                overflow_k_ungated.dtype
            )
        else:
            overflow_k_gated = overflow_k_ungated
        if self.config.kvm_use_merge_gate_values:
            overflow_v_gated = (overflow_v_ungated * merge_gate).to(
                overflow_v_ungated.dtype
            )
        else:
            overflow_v_gated = overflow_v_ungated
        if self.config.kvm_apply_merge_gate_to_appends:
            overflow_k_append = overflow_k_gated
            overflow_v_append = overflow_v_gated
        else:
            overflow_k_append = overflow_k_ungated
            overflow_v_append = overflow_v_ungated

        batch_size, kv_heads, current_state_len, _ = s_k.shape
        if kv_heads != self.num_key_value_heads:
            raise AssertionError("KVM Triton decode cache head count mismatch.")
        current_state_len = int(s_k.size(2))
        desired_state_len = self._desired_state_len(
            ctx_len, available_context, current_state_len
        )
        n_append = min(max(desired_state_len - current_state_len, 0), overflow_len)
        state_after = current_state_len + n_append
        kv_rows = batch_size * kv_heads

        def pad_state(x: torch.Tensor, dim: int) -> torch.Tensor:
            flat = x.reshape(kv_rows, current_state_len, dim).contiguous()
            if state_after == current_state_len:
                return flat.clone()
            out = flat.new_zeros(kv_rows, state_after, dim)
            out[:, :current_state_len].copy_(flat)
            return out

        state_k = pad_state(s_k, self.d_qk_head)
        state_v = pad_state(s_v, self.d_v_head)
        state_k_attn = pad_state(self.ln_s_k(s_k), self.d_qk_head)
        s_v_attn = (F.normalize(s_v.float(), dim=-1) * s_vlen).to(s_v.dtype)
        state_v_attn = pad_state(s_v_attn, self.d_v_head)

        state_vlen_flat = s_vlen.reshape(kv_rows, current_state_len).contiguous()
        if state_after == current_state_len:
            state_vlen = state_vlen_flat.clone()
        else:
            state_vlen = torch.zeros(
                kv_rows,
                state_after,
                device=s_vlen.device,
                dtype=torch.float32,
            )
            state_vlen[:, :current_state_len].copy_(state_vlen_flat.float())

        if n_append < overflow_len and current_state_len <= self.sink_len:
            raise AssertionError(
                "KVM Triton decode update requires a non-sink state slot before merging."
            )

        triton_args = self._make_triton_args(batch_size, overflow_len)
        triton_args.max_state_len = state_after
        triton_args.initial_state_len = current_state_len
        schedule = self._make_generation_update_schedule(
            current_state_len=current_state_len,
            state_after=state_after,
            n_append=n_append,
            overflow_len=overflow_len,
        )
        buffers = allocate_work_buffers(triton_args, schedule, overflow_k.device)

        def flatten_update(x: torch.Tensor, dim: int) -> torch.Tensor:
            return x.reshape(kv_rows, overflow_len, dim).contiguous()

        overflow_select_k_flat = flatten_update(overflow_k_ungated, self.d_qk_head)
        overflow_append_k_flat = flatten_update(overflow_k_append, self.d_qk_head)
        overflow_append_v_flat = flatten_update(overflow_v_append, self.d_v_head)
        overflow_merge_k_flat = flatten_update(overflow_k_gated, self.d_qk_head)
        overflow_merge_v_flat = flatten_update(overflow_v_gated, self.d_v_head)

        append_pos_by_token = torch.full(
            (kv_rows, 1, overflow_len),
            -1,
            device=overflow_k.device,
            dtype=torch.int32,
        )
        best_idx_by_token = torch.full_like(append_pos_by_token, -1)
        undo_k_by_token = torch.empty(
            kv_rows,
            1,
            overflow_len,
            self.d_qk_head,
            device=overflow_k.device,
            dtype=state_k.dtype,
        )
        undo_v_by_token = torch.empty(
            kv_rows,
            1,
            overflow_len,
            self.d_v_head,
            device=overflow_k.device,
            dtype=state_v.dtype,
        )

        run_forward_state_update(
            triton_args,
            schedule,
            overflow_merge_k_flat,
            overflow_merge_v_flat,
            self.ln_s_k.weight.float(),
            self.ln_s_k.bias.float(),
            buffers,
            state_k,
            state_v,
            state_k_attn,
            state_v_attn,
            state_vlen,
            append_pos_by_token,
            best_idx_by_token,
            undo_k_by_token,
            undo_v_by_token,
            0,
            False,
            overflow_select_k_flat=overflow_select_k_flat,
            overflow_append_k_flat=overflow_append_k_flat,
            overflow_append_v_flat=overflow_append_v_flat,
            overflow_merge_k_flat=overflow_merge_k_flat,
            overflow_merge_v_flat=overflow_merge_v_flat,
        )

        return (
            state_k.reshape(batch_size, kv_heads, state_after, self.d_qk_head),
            state_v.reshape(batch_size, kv_heads, state_after, self.d_v_head),
            state_vlen.reshape(batch_size, kv_heads, state_after, 1),
        )
