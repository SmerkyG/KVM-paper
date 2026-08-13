"""Drive DiffusionGemma acceptance with a paired native/native entropy view."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from types import MethodType
from typing import Any, Iterator

import torch
from torch import nn

from .diffusion_gemma_acceptance_compare import _native_decoder_attention
from .diffusion_gemma_phase_compare import (
    _acceptance_mask,
    _encoder_attention_modules,
    _entropy,
    _outer_encoder,
)


_ORIGINAL_ENCODER_ATTRIBUTE = "_diffusion_gemma_native_entropy_original_encoder"
_ORIGINAL_STEP_ATTRIBUTE = "_diffusion_gemma_native_entropy_original_step"
_SHADOW_CACHE_ATTRIBUTE = "_diffusion_gemma_native_entropy_shadow_cache"
_ATTENTION_ORIGINAL_FORWARD_ATTRIBUTE = "_diffusion_gemma_lod_original_forward"


@contextmanager
def _unpatched_native_encoder_attention(model: nn.Module) -> Iterator[None]:
    """Run exact encoder attention without constructing a second LOD sidecar."""

    modules = _encoder_attention_modules(model)
    previous = [module.forward for module in modules]
    originals = [
        getattr(module, _ATTENTION_ORIGINAL_FORWARD_ATTRIBUTE, None)
        for module in modules
    ]
    if any(original is None for original in originals):
        raise RuntimeError("native-entropy control requires installed LOD attention")
    for module, original in zip(modules, originals, strict=True):
        module.forward = original
    try:
        yield
    finally:
        for module, forward in zip(modules, previous, strict=True):
            module.forward = forward


class DiffusionGemmaNativeEntropyAcceptance:
    """Use native/native entropy ordering while preserving the LOD trajectory.

    The causal prompt is encoded twice into independent caches.  The primary
    cache and all sampled tokens, logits, self-conditioning, and stopping
    decisions remain LOD-controlled.  A shadow native encoder plus native
    decoder produces only the token acceptance mask used by the sampler.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.encoder_calls = 0
        self.steps = 0
        self.positions = 0
        self.lod_accept_count = 0
        self.native_accept_count = 0
        self.accept_mask_disagreements = 0
        self.accept_intersection = 0
        self.accept_union = 0
        self.native_only_accepts = 0
        self.lod_only_accepts = 0
        self.top1_disagreements = 0
        self.native_accepted_sample_top1_disagreements = 0
        self.lod_entropy_on_native_accepted_sum = 0.0
        self.native_entropy_on_native_accepted_sum = 0.0
        self._native_masks_applied = 0

    @staticmethod
    def _model_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        denoising_keys = {
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
        return {
            key: value for key, value in kwargs.items() if key not in denoising_keys
        }

    def install(self) -> None:
        outer_encoder = _outer_encoder(self.model)
        if hasattr(outer_encoder, _ORIGINAL_ENCODER_ATTRIBUTE):
            raise RuntimeError("native-entropy acceptance is already installed")
        if hasattr(self.model, _ORIGINAL_STEP_ATTRIBUTE):
            raise RuntimeError("native-entropy denoising hook is already installed")
        original_encoder = outer_encoder.forward
        original_step = self.model._denoising_step
        setattr(outer_encoder, _ORIGINAL_ENCODER_ATTRIBUTE, original_encoder)
        setattr(self.model, _ORIGINAL_STEP_ATTRIBUTE, original_step)
        controller = self

        def dual_encoder_forward(
            encoder_self: nn.Module, *args: Any, **kwargs: Any
        ) -> Any:
            primary_cache = kwargs.get("past_key_values")
            if primary_cache is None:
                raise RuntimeError("native-entropy acceptance requires an encoder cache")
            shadow_cache = getattr(primary_cache, _SHADOW_CACHE_ATTRIBUTE, None)
            if shadow_cache is None:
                shadow_cache = copy.deepcopy(primary_cache)
                setattr(primary_cache, _SHADOW_CACHE_ATTRIBUTE, shadow_cache)
            shadow_kwargs = dict(kwargs)
            shadow_kwargs["past_key_values"] = shadow_cache
            with _unpatched_native_encoder_attention(controller.model):
                original_encoder(*args, **shadow_kwargs)
            controller.encoder_calls += 1
            return original_encoder(*args, **kwargs)

        def controlled_step(model_self: nn.Module, *args: Any, **kwargs: Any):
            if args:
                raise TypeError(
                    "native-entropy acceptance expects keyword denoising inputs"
                )
            current_canvas = kwargs["current_canvas"]
            self_conditioning_logits = kwargs["self_conditioning_logits"]
            mask_mapping = kwargs["mask_mapping"]
            primary_cache = kwargs["past_key_values"]
            shadow_cache = getattr(primary_cache, _SHADOW_CACHE_ATTRIBUTE, None)
            if shadow_cache is None:
                raise RuntimeError("native encoder shadow cache is missing")
            decoder_position_ids = kwargs["decoder_position_ids"]
            logits_processor = kwargs["logits_processor"]
            input_ids = kwargs["input_ids"]
            cur_step = kwargs["cur_step"]
            sampler = kwargs["sampler"]
            active_batch = ~kwargs["finished_denoising"]
            model_kwargs = controller._model_kwargs(kwargs)
            step_tensor = torch.tensor(
                cur_step, device=current_canvas.device, dtype=torch.int32
            )

            # This shadow branch observes the exact canvas and the previous
            # LOD self-conditioning logits.  Only its entropy ordering is used.
            with _native_decoder_attention(model_self):
                native_outputs = model_self(
                    decoder_input_ids=current_canvas,
                    self_conditioning_logits=self_conditioning_logits,
                    decoder_attention_mask=mask_mapping,
                    past_key_values=shadow_cache,
                    decoder_position_ids=decoder_position_ids,
                    **model_kwargs,
                )
            native_logits = logits_processor(
                input_ids, native_outputs.logits, cur_step=step_tensor
            )
            native_entropy = _entropy(native_logits)
            native_accept = _acceptance_mask(
                native_entropy, float(sampler.entropy_bound)
            )
            native_top1 = native_logits.argmax(dim=-1)

            captured: dict[str, torch.Tensor] = {}
            original_accept = sampler.accept_canvas

            def accept_with_native_entropy(
                canvas: torch.Tensor,
                denoiser_canvas: torch.Tensor,
                lod_logits: torch.Tensor,
                step: int,
            ) -> torch.Tensor:
                del step
                lod_entropy = _entropy(lod_logits)
                lod_accept = _acceptance_mask(
                    lod_entropy, float(sampler.entropy_bound)
                )
                sampler.accepted_token_mask = native_accept
                captured["lod_entropy"] = lod_entropy
                captured["lod_accept"] = lod_accept
                captured["lod_top1"] = lod_logits.argmax(dim=-1)
                captured["sampled_tokens"] = denoiser_canvas
                controller._native_masks_applied += 1
                return torch.where(native_accept, denoiser_canvas, canvas)

            sampler.accept_canvas = accept_with_native_entropy
            try:
                result = original_step(*args, **kwargs)
            finally:
                sampler.accept_canvas = original_accept

            if "lod_accept" not in captured:
                raise RuntimeError("native entropy mask was not applied")
            if not torch.equal(sampler.accepted_token_mask, native_accept):
                raise RuntimeError("sampler did not retain the native entropy mask")

            valid = active_batch.unsqueeze(-1).expand_as(native_accept)
            lod_accept = captured["lod_accept"] & valid
            applied_accept = native_accept & valid
            disagreement = lod_accept.ne(applied_accept) & valid
            controller.steps += 1
            controller.positions += int(valid.sum().item())
            controller.lod_accept_count += int(lod_accept.sum().item())
            controller.native_accept_count += int(applied_accept.sum().item())
            controller.accept_mask_disagreements += int(disagreement.sum().item())
            controller.accept_intersection += int(
                (lod_accept & applied_accept).sum().item()
            )
            controller.accept_union += int((lod_accept | applied_accept).sum().item())
            controller.native_only_accepts += int(
                (applied_accept & ~lod_accept).sum().item()
            )
            controller.lod_only_accepts += int(
                (lod_accept & ~applied_accept).sum().item()
            )
            controller.top1_disagreements += int(
                (captured["lod_top1"].ne(native_top1) & valid).sum().item()
            )
            controller.native_accepted_sample_top1_disagreements += int(
                (
                    captured["sampled_tokens"].ne(native_top1)
                    & applied_accept
                ).sum().item()
            )
            accepted = int(applied_accept.sum().item())
            if accepted:
                controller.lod_entropy_on_native_accepted_sum += float(
                    captured["lod_entropy"].masked_select(applied_accept).sum().item()
                )
                controller.native_entropy_on_native_accepted_sum += float(
                    native_entropy.masked_select(applied_accept).sum().item()
                )
            del native_logits, native_outputs
            return result

        outer_encoder.forward = MethodType(dual_encoder_forward, outer_encoder)
        self.model._denoising_step = MethodType(controlled_step, self.model)

    def uninstall(self) -> None:
        outer_encoder = _outer_encoder(self.model)
        original_encoder = getattr(outer_encoder, _ORIGINAL_ENCODER_ATTRIBUTE, None)
        if original_encoder is not None:
            outer_encoder.forward = original_encoder
            delattr(outer_encoder, _ORIGINAL_ENCODER_ATTRIBUTE)
        original_step = getattr(self.model, _ORIGINAL_STEP_ATTRIBUTE, None)
        if original_step is not None:
            self.model._denoising_step = original_step
            delattr(self.model, _ORIGINAL_STEP_ATTRIBUTE)

    def summary(self) -> dict[str, Any]:
        accepted = self.native_accept_count
        return {
            "entropy_source": "native_encoder_native_decoder",
            "sample_source": "lod_categorical_logits",
            "self_conditioning_source": "lod_processed_logits",
            "stopping_source": "lod_argmax_and_logits",
            "state_size_changed": False,
            "encoder_calls": self.encoder_calls,
            "steps": self.steps,
            "native_masks_applied": self._native_masks_applied,
            "positions": self.positions,
            "hypothetical_lod_accepts": self.lod_accept_count,
            "applied_native_accepts": accepted,
            "accept_mask_disagreements": self.accept_mask_disagreements,
            "accept_mask_disagreement_rate": (
                self.accept_mask_disagreements / self.positions
                if self.positions
                else None
            ),
            "accept_mask_jaccard": (
                self.accept_intersection / self.accept_union
                if self.accept_union
                else None
            ),
            "native_only_accepts": self.native_only_accepts,
            "lod_only_accepts": self.lod_only_accepts,
            "top1_disagreements": self.top1_disagreements,
            "top1_disagreement_rate": (
                self.top1_disagreements / self.positions if self.positions else None
            ),
            "sample_native_top1_disagreements_on_applied_accepts": (
                self.native_accepted_sample_top1_disagreements
            ),
            "sample_native_top1_disagreement_rate_on_applied_accepts": (
                self.native_accepted_sample_top1_disagreements / accepted
                if accepted
                else None
            ),
            "mean_lod_entropy_on_applied_accepts": (
                self.lod_entropy_on_native_accepted_sum / accepted
                if accepted
                else None
            ),
            "mean_native_entropy_on_applied_accepts": (
                self.native_entropy_on_native_accepted_sum / accepted
                if accepted
                else None
            ),
        }


__all__ = ["DiffusionGemmaNativeEntropyAcceptance"]
