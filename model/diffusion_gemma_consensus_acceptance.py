"""Sparse-view consensus acceptance for DiffusionGemma LOD decoding."""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Any, Iterator

import torch
from torch import nn


_ORIGINAL_STEP_ATTRIBUTE = "_diffusion_gemma_consensus_original_step"
_OPEN_COUNT_ATTRIBUTE = "_diffusion_gemma_decoder_open_count"


def _decoder_attention_modules(model: nn.Module) -> list[nn.Module]:
    base = getattr(model, "model", model)
    decoder = getattr(base, "decoder", None)
    if decoder is None:
        raise TypeError("expected a DiffusionGemma model with a decoder")
    return [layer.self_attn for layer in decoder.layers]


@contextmanager
def _decoder_open_count(model: nn.Module, open_count: int) -> Iterator[None]:
    modules = _decoder_attention_modules(model)
    previous = [getattr(module, _OPEN_COUNT_ATTRIBUTE, None) for module in modules]
    for module in modules:
        if hasattr(module, "_diffusion_gemma_lod_settings"):
            setattr(module, _OPEN_COUNT_ATTRIBUTE, open_count)
    try:
        yield
    finally:
        for module, old_count in zip(modules, previous, strict=True):
            if old_count is None:
                if hasattr(module, _OPEN_COUNT_ATTRIBUTE):
                    delattr(module, _OPEN_COUNT_ATTRIBUTE)
            else:
                setattr(module, _OPEN_COUNT_ATTRIBUTE, old_count)


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    return torch.distributions.Categorical(logits=logits).entropy()


class DiffusionGemmaConsensusAcceptance:
    """Re-rank top-8 acceptance with a wider sparse decoder probe."""

    def __init__(
        self,
        model: nn.Module,
        *,
        probe_open_count: int = 16,
        mode: str = "apply",
    ) -> None:
        if probe_open_count < 1:
            raise ValueError("probe open count must be positive")
        if mode not in ("observe", "apply"):
            raise ValueError(f"unknown consensus acceptance mode {mode!r}")
        self.model = model
        self.probe_open_count = probe_open_count
        self.mode = mode
        self.steps = 0
        self.positions = 0
        self.top1_disagreements = 0
        self.original_accepted = 0
        self.original_accepted_top1_disagreements = 0
        self.final_accepted = 0
        self.veto_shortfall = 0
        self.replacements = 0
        self.primary_entropy_original_accepted_sum = 0.0
        self.probe_entropy_original_accepted_sum = 0.0
        self.effective_entropy_final_accepted_sum = 0.0

    def install(self) -> None:
        if hasattr(self.model, _ORIGINAL_STEP_ATTRIBUTE):
            raise RuntimeError("consensus acceptance is already installed")
        original_step = self.model._denoising_step
        setattr(self.model, _ORIGINAL_STEP_ATTRIBUTE, original_step)
        controller = self

        def consensus_step(model_self: nn.Module, *args: Any, **kwargs: Any):
            current_canvas = kwargs["current_canvas"]
            self_conditioning_logits = kwargs["self_conditioning_logits"]
            mask_mapping = kwargs["mask_mapping"]
            past_key_values = kwargs["past_key_values"]
            decoder_position_ids = kwargs["decoder_position_ids"]
            logits_processor = kwargs["logits_processor"]
            input_ids = kwargs["input_ids"]
            cur_step = kwargs["cur_step"]
            sampler = kwargs["sampler"]
            active_batch = ~kwargs["finished_denoising"]
            model_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key
                not in {
                    "current_canvas",
                    "self_conditioning_logits",
                    "mask_mapping",
                    "past_key_values",
                    "decoder_position_ids",
                    "logits_processor",
                    "input_ids",
                    "cur_step",
                    "sampler",
                    "argmax_canvas",
                    "finished_denoising",
                    "diffusion_stopping_criteria",
                    "decoder_forward",
                }
            }
            with _decoder_open_count(model_self, controller.probe_open_count):
                probe_outputs = model_self(
                    decoder_input_ids=current_canvas,
                    self_conditioning_logits=self_conditioning_logits,
                    decoder_attention_mask=mask_mapping,
                    past_key_values=past_key_values,
                    decoder_position_ids=decoder_position_ids,
                    **model_kwargs,
                )
            step_tensor = torch.tensor(
                cur_step, device=current_canvas.device, dtype=torch.int32
            )
            probe_logits = logits_processor(
                input_ids, probe_outputs.logits, cur_step=step_tensor
            )
            probe_entropy = _entropy(probe_logits)
            probe_top1 = probe_logits.argmax(dim=-1)
            del probe_outputs

            original_accept = sampler.accept_canvas

            def consensus_accept(
                canvas: torch.Tensor,
                denoiser_canvas: torch.Tensor,
                primary_logits: torch.Tensor,
                step: int,
            ) -> torch.Tensor:
                original_canvas = original_accept(
                    canvas, denoiser_canvas, primary_logits, step
                )
                original_mask = sampler.accepted_token_mask.clone()
                primary_entropy = _entropy(primary_logits)
                primary_top1 = primary_logits.argmax(dim=-1)
                eligible = primary_top1.eq(probe_top1)
                effective_entropy = torch.maximum(primary_entropy, probe_entropy)
                acceptance_count = original_mask.sum(dim=-1)
                ranked_scores = effective_entropy.masked_fill(
                    ~eligible, float("inf")
                )
                ranked_indices = ranked_scores.argsort(dim=-1)
                selected_by_rank = (
                    torch.arange(
                        ranked_indices.size(1), device=ranked_indices.device
                    ).unsqueeze(0)
                    < acceptance_count.unsqueeze(-1)
                )
                consensus_mask = torch.scatter(
                    torch.zeros_like(selected_by_rank),
                    dim=-1,
                    index=ranked_indices,
                    src=selected_by_rank,
                )
                consensus_mask &= eligible
                final_mask = (
                    consensus_mask if controller.mode == "apply" else original_mask
                )
                sampler.accepted_token_mask = final_mask

                valid = active_batch.unsqueeze(-1).expand_as(original_mask)
                original_active = original_mask & valid
                final_active = final_mask & valid
                controller.steps += 1
                controller.positions += int(valid.sum().item())
                controller.top1_disagreements += int(
                    (primary_top1.ne(probe_top1) & valid).sum().item()
                )
                controller.original_accepted += int(original_active.sum().item())
                controller.original_accepted_top1_disagreements += int(
                    (original_active & ~eligible).sum().item()
                )
                controller.final_accepted += int(final_active.sum().item())
                controller.veto_shortfall += int(
                    (acceptance_count - consensus_mask.sum(dim=-1))
                    .masked_select(active_batch)
                    .sum()
                    .item()
                )
                controller.replacements += int(
                    (final_active & ~original_mask).sum().item()
                )
                if original_active.any():
                    controller.primary_entropy_original_accepted_sum += float(
                        primary_entropy.masked_select(original_active).sum().item()
                    )
                    controller.probe_entropy_original_accepted_sum += float(
                        probe_entropy.masked_select(original_active).sum().item()
                    )
                if final_active.any():
                    controller.effective_entropy_final_accepted_sum += float(
                        effective_entropy.masked_select(final_active).sum().item()
                    )
                if controller.mode == "observe":
                    return original_canvas
                return torch.where(final_mask, denoiser_canvas, canvas)

            sampler.accept_canvas = consensus_accept
            try:
                return original_step(*args, **kwargs)
            finally:
                sampler.accept_canvas = original_accept

        self.model._denoising_step = MethodType(consensus_step, self.model)

    def uninstall(self) -> None:
        original = getattr(self.model, _ORIGINAL_STEP_ATTRIBUTE, None)
        if original is None:
            return
        self.model._denoising_step = original
        delattr(self.model, _ORIGINAL_STEP_ATTRIBUTE)

    def summary(self) -> dict[str, Any]:
        positions = self.positions
        original = self.original_accepted
        final = self.final_accepted
        return {
            "mode": self.mode,
            "probe_open_count": self.probe_open_count,
            "steps": self.steps,
            "positions": positions,
            "top1_disagreements": self.top1_disagreements,
            "top1_disagreement_rate": (
                self.top1_disagreements / positions if positions else None
            ),
            "original_accepted": original,
            "original_accepted_top1_disagreements": self.original_accepted_top1_disagreements,
            "original_accepted_top1_disagreement_rate": (
                self.original_accepted_top1_disagreements / original
                if original
                else None
            ),
            "final_accepted": final,
            "veto_shortfall": self.veto_shortfall,
            "replacements": self.replacements,
            "mean_primary_entropy_on_original_accepted": (
                self.primary_entropy_original_accepted_sum / original
                if original
                else None
            ),
            "mean_probe_entropy_on_original_accepted": (
                self.probe_entropy_original_accepted_sum / original
                if original
                else None
            ),
            "mean_effective_entropy_on_final_accepted": (
                self.effective_entropy_final_accepted_sum / final if final else None
            ),
        }


__all__ = ["DiffusionGemmaConsensusAcceptance"]
