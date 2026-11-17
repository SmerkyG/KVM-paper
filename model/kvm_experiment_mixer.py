"""Capacity-matched expansion and bipartite controls for the Wave 1 runs.

Controls inherit the reference mixer's original dense-mask attention path.
Bipartite matching replaces grouping only: sums, merge gates, sink protection,
and destination birth radii retain their KVM meanings. The first Wave 1 runs
used a Flash-compatible attention override, preserved in their frozen sources;
it is intentionally absent from subsequent runs.
"""

from __future__ import annotations

import os

import torch
from torch.nn import functional as F

from .kvm_mixer import SequenceMixer as ReferenceMixer, _all_idx, _gather_by_idx


class SequenceMixer(ReferenceMixer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        if config.kvm_append_policy not in {"least_redundant", "random", "uniform"}:
            raise ValueError(f"Unknown append policy: {config.kvm_append_policy}")
        if config.kvm_merge_strategy not in {"nearest", "bipartite"}:
            raise ValueError(f"Unknown merge strategy: {config.kvm_merge_strategy}")
        if config.kvm_bipartite_chunk_size < 2 or config.kvm_bipartite_chunk_size % 2:
            raise ValueError("Bipartite matching chunks must have positive even size")
        if config.kvm_merge_strategy == "bipartite" and not config.kvm_use_vlens:
            raise ValueError("The Wave 1 bipartite control uses KVM birth-radius readout")
        self.compile_attention = config.kvm_merge_strategy != "bipartite"
        self._append_generators = {}

    def reset_append_rng(self):
        """Reset independent train/evaluation routing streams for reproducibility."""
        self._append_generators.clear()

    @torch.compiler.disable
    def _split_append_merge_indices(self, k_block, n_append, k_ref):
        if self.config.kvm_append_policy == "least_redundant":
            return super()._split_append_merge_indices(k_block, n_append, k_ref)
        block_len = k_block.size(2)
        n_append = min(max(int(n_append), 0), block_len)
        all_idx = _all_idx(k_block, block_len)
        if n_append == 0 or n_append == block_len:
            return all_idx[..., :n_append], all_idx[..., n_append:]
        with torch.no_grad():
            if self.config.kvm_append_policy == "uniform":
                selected = ((torch.arange(n_append, device=k_block.device) + 0.5)
                            * block_len / n_append).long()
                selected = selected.view(1, 1, -1).expand(*all_idx.shape[:2], -1)
            else:
                stream = (str(k_block.device), self.training)
                if stream not in self._append_generators:
                    generator = torch.Generator(device=k_block.device)
                    generator.manual_seed(
                        self.config.kvm_routing_seed
                        + 1_000_003 * self.layer_idx
                        + 10_000_019 * int(os.environ.get("RANK", "0"))
                        + (0 if self.training else 100_000_007)
                    )
                    self._append_generators[stream] = generator
                scores = torch.rand(
                    all_idx.shape, device=k_block.device,
                    generator=self._append_generators[stream],
                )
                selected = scores.argsort(dim=-1)[..., :n_append]
            selected = selected.sort(dim=-1).values
            mask = torch.zeros_like(all_idx, dtype=torch.bool).scatter(-1, selected, True)
            remaining = all_idx.masked_select(~mask).view(*all_idx.shape[:2], -1)
        return selected, remaining

    def _update_state_from_overflow_tokens(
        self, s_k, s_v, s_vlen, overflow_k, overflow_v, merge_gate,
        ctx_len, available_context,
    ):
        if self.config.kvm_merge_strategy == "nearest":
            return super()._update_state_from_overflow_tokens(
                s_k, s_v, s_vlen, overflow_k, overflow_v, merge_gate,
                ctx_len, available_context,
            )
        if overflow_k.size(2) == 0:
            return s_k, s_v, s_vlen
        target = self._desired_state_len(ctx_len, available_context, s_k.size(2))
        protected = min(self.sink_len, s_k.size(2))
        old_len = s_k.size(2) - protected
        incoming_k = self._prepare_state_update_k(overflow_k)
        pool_k = torch.cat([s_k[:, :, protected:], incoming_k], dim=2)
        pool_v = torch.cat([s_v[:, :, protected:], overflow_v], dim=2)
        pool_radius = torch.cat([
            s_vlen[:, :, protected:], overflow_v.float().norm(dim=-1, keepdim=True)
        ], dim=2)
        pool_gate = torch.cat([
            torch.ones_like(s_vlen[:, :, protected:]), merge_gate
        ], dim=2)
        n_merge = pool_k.size(2) - (target - protected)
        if n_merge == 0:
            return (
                torch.cat([s_k[:, :, :protected], pool_k], dim=2),
                torch.cat([s_v[:, :, :protected], pool_v], dim=2),
                torch.cat([s_vlen[:, :, :protected], pool_radius], dim=2),
            )
        src, dst, keep = self._bipartite_indices(pool_k, n_merge)
        src_k = _gather_by_idx(pool_k, src)
        src_v = _gather_by_idx(pool_v, src)
        weights = _gather_by_idx(pool_gate, src)
        is_incoming = (src >= old_len).unsqueeze(-1)
        if self.config.kvm_use_merge_gate_keys:
            src_k = src_k * torch.where(is_incoming, weights, 1.0)
        if self.config.kvm_use_merge_gate_values:
            src_v = src_v * torch.where(is_incoming, weights, 1.0)
        # Match the reference's one-hot batched-matmul accumulation arithmetic.
        routes = pool_k.new_zeros((*src.shape, pool_k.size(2)))
        routes.scatter_(-1, dst.unsqueeze(-1), 1)
        pool_k = pool_k + routes.mT @ src_k.to(pool_k.dtype)
        pool_v = pool_v + routes.mT @ src_v.to(pool_v.dtype)
        return (
            torch.cat([s_k[:, :, :protected], _gather_by_idx(pool_k, keep)], dim=2),
            torch.cat([s_v[:, :, :protected], _gather_by_idx(pool_v, keep)], dim=2),
            torch.cat([s_vlen[:, :, :protected], _gather_by_idx(pool_radius, keep)], dim=2),
        )

    @torch.compiler.disable
    def _bipartite_indices(self, pool_k, n_merge):
        """Match even sources to odd destinations within fixed-size chunks."""
        batch, heads, length, width = pool_k.shape
        chunk = self.config.kvm_bipartite_chunk_size
        remainder = length % chunk
        max_sources = (length // chunk) * (chunk // 2)
        if remainder >= 2:
            max_sources += (remainder + 1) // 2
        if n_merge > max_sources:
            raise ValueError("Requested budget exceeds one-pass bipartite merge capacity")
        padded_len = ((length + chunk - 1) // chunk) * chunk
        with torch.no_grad():
            keys = self.ln_s_k(pool_k)
            keys = F.pad(keys, (0, 0, 0, padded_len - length))
            keys = keys.view(batch, heads, -1, chunk, width)
            scores = keys[..., ::2, :] @ keys[..., 1::2, :].transpose(-1, -2)
            positions = torch.arange(padded_len, device=pool_k.device).view(-1, chunk)
            valid = ((positions[:, ::2, None] < length)
                     & (positions[:, None, 1::2] < length))
            scores = scores.masked_fill(~valid, -torch.inf)
            best_score, best_dst = scores.max(dim=-1)
            selected = best_score.flatten(-2).argsort(dim=-1, descending=True)[..., :n_merge]
            src = selected * 2
            dst_local = best_dst.flatten(-2).gather(-1, selected)
            dst = (selected // (chunk // 2)) * chunk + dst_local * 2 + 1
            indices = _all_idx(pool_k, length)
            removed = torch.zeros_like(indices, dtype=torch.bool).scatter(-1, src, True)
            keep = indices.masked_select(~removed).view(batch, heads, length - n_merge)
        return src, dst, keep
