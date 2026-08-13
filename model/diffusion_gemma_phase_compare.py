"""Four-way attention-phase comparison on one DiffusionGemma trajectory."""

from __future__ import annotations

import copy
from contextlib import contextmanager, nullcontext
from types import MethodType
from typing import Any, Iterator

import torch
from torch import nn

from .diffusion_gemma_acceptance_compare import _native_decoder_attention


_ENCODER_ATTENTION_ATTRIBUTE = "_diffusion_gemma_encoder_attention_mode"
_ORIGINAL_ENCODER_ATTRIBUTE = "_diffusion_gemma_phase_original_encoder"
_ORIGINAL_STEP_ATTRIBUTE = "_diffusion_gemma_phase_original_step"
_SHADOW_CACHE_ATTRIBUTE = "_diffusion_gemma_native_encoder_shadow_cache"


def _base_model(model: nn.Module) -> nn.Module:
    return getattr(model, "model", model)


def _outer_encoder(model: nn.Module) -> nn.Module:
    encoder = getattr(_base_model(model), "encoder", None)
    if encoder is None:
        raise TypeError("expected a DiffusionGemma model with an encoder")
    return encoder


def _encoder_attention_modules(model: nn.Module) -> list[nn.Module]:
    language_model = getattr(_outer_encoder(model), "language_model", None)
    if language_model is None:
        raise TypeError("expected encoder.language_model")
    return [
        layer.self_attn
        for layer in language_model.layers
        if hasattr(layer.self_attn, "_diffusion_gemma_lod_settings")
    ]


@contextmanager
def _encoder_attention(model: nn.Module, mode: str) -> Iterator[None]:
    modules = _encoder_attention_modules(model)
    previous = [getattr(module, _ENCODER_ATTENTION_ATTRIBUTE, "lod") for module in modules]
    for module in modules:
        setattr(module, _ENCODER_ATTENTION_ATTRIBUTE, mode)
    try:
        yield
    finally:
        for module, old_mode in zip(modules, previous, strict=True):
            setattr(module, _ENCODER_ATTENTION_ATTRIBUTE, old_mode)


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return torch.logsumexp(logits.float(), dim=-1) - (
        probabilities * logits.float()
    ).sum(dim=-1)


def _acceptance_mask(entropy: torch.Tensor, entropy_bound: float) -> torch.Tensor:
    sorted_entropy, sorted_indices = torch.sort(entropy, dim=-1)
    selected = torch.cumsum(sorted_entropy, dim=-1) - sorted_entropy <= entropy_bound
    return torch.scatter(
        torch.zeros_like(selected), dim=-1, index=sorted_indices, src=selected
    )


class DiffusionGemmaPhaseComparator:
    """Compare encoder/decoder LOD/native combinations on an LOD trajectory."""

    branch_names = ("lod_lod", "lod_native", "native_lod", "native_native")
    repair_names = ("encoder_only", "decoder_only", "either_single", "both_needed")

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.steps = 0
        self.positions = 0
        self.reference_accepted = 0
        self.branch_entropy_sum = {name: 0.0 for name in self.branch_names}
        self.branch_accepted_entropy_sum = {name: 0.0 for name in self.branch_names}
        self.branch_accept_count = {name: 0 for name in self.branch_names}
        self.branch_accept_disagree_native_native = {
            name: 0 for name in self.branch_names
        }
        self.branch_accept_intersection_native_native = {
            name: 0 for name in self.branch_names
        }
        self.branch_accept_union_native_native = {
            name: 0 for name in self.branch_names
        }
        self.top1_disagree_native_native = {name: 0 for name in self.branch_names}
        self.accepted_top1_disagree_native_native = {
            name: 0 for name in self.branch_names
        }
        self.pairwise_top1_disagreements = {
            f"{left}__{right}": 0
            for index, left in enumerate(self.branch_names)
            for right in self.branch_names[index + 1 :]
        }
        self.repair_counts = {name: 0 for name in self.repair_names}
        self.accepted_repair_counts = {name: 0 for name in self.repair_names}
        self.low_entropy_accepted_repair_counts = {
            name: 0 for name in self.repair_names
        }
        self.first128_accepted_repair_counts = {
            name: 0 for name in self.repair_names
        }
        self.reference_native_disagreements = 0
        self.accepted_reference_native_disagreements = 0
        self.low_entropy_accepted_reference_native_disagreements = 0
        self.events: list[dict[str, Any]] = []
        self._trajectory_batches: list[tuple[torch.Tensor, int]] = []
        self._next_trajectory = 0

    def _trajectory_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        for previous, first_id in self._trajectory_batches:
            if previous is input_ids:
                return torch.arange(
                    first_id,
                    first_id + int(input_ids.size(0)),
                    device=input_ids.device,
                )
        first_id = self._next_trajectory
        self._next_trajectory += int(input_ids.size(0))
        self._trajectory_batches.append((input_ids, first_id))
        return torch.arange(
            first_id,
            first_id + int(input_ids.size(0)),
            device=input_ids.device,
        )

    def _record(
        self,
        branches: dict[str, dict[str, torch.Tensor]],
        reference_accepted: torch.Tensor,
        active_batch: torch.Tensor,
        trajectory_ids: torch.Tensor,
    ) -> None:
        valid = active_batch.unsqueeze(-1).expand_as(reference_accepted)
        reference_accepted = reference_accepted & valid
        reference_entropy = branches["lod_lod"]["entropy"]
        low_entropy_accepted = reference_accepted & reference_entropy.lt(0.001)
        native_top1 = branches["native_native"]["top1"]
        native_accept = branches["native_native"]["accept"]

        self.steps += 1
        self.positions += int(valid.sum().item())
        self.reference_accepted += int(reference_accepted.sum().item())
        for name in self.branch_names:
            entropy = branches[name]["entropy"]
            top1 = branches[name]["top1"]
            accepted = branches[name]["accept"] & valid
            self.branch_entropy_sum[name] += float(entropy.masked_select(valid).sum().item())
            self.branch_accepted_entropy_sum[name] += float(
                entropy.masked_select(reference_accepted).sum().item()
            )
            self.branch_accept_count[name] += int(accepted.sum().item())
            self.top1_disagree_native_native[name] += int(
                (top1.ne(native_top1) & valid).sum().item()
            )
            self.accepted_top1_disagree_native_native[name] += int(
                (top1.ne(native_top1) & reference_accepted).sum().item()
            )
            self.branch_accept_disagree_native_native[name] += int(
                (accepted.ne(native_accept) & valid).sum().item()
            )
            self.branch_accept_intersection_native_native[name] += int(
                (accepted & native_accept & valid).sum().item()
            )
            self.branch_accept_union_native_native[name] += int(
                ((accepted | native_accept) & valid).sum().item()
            )

        for index, left in enumerate(self.branch_names):
            for right in self.branch_names[index + 1 :]:
                key = f"{left}__{right}"
                self.pairwise_top1_disagreements[key] += int(
                    (
                        branches[left]["top1"].ne(branches[right]["top1"])
                        & valid
                    ).sum().item()
                )

        lod_top1 = branches["lod_lod"]["top1"]
        encoder_top1 = branches["native_lod"]["top1"]
        decoder_top1 = branches["lod_native"]["top1"]
        disagreement = lod_top1.ne(native_top1) & valid
        encoder_repairs = encoder_top1.eq(native_top1)
        decoder_repairs = decoder_top1.eq(native_top1)
        repair_masks = {
            "encoder_only": disagreement & encoder_repairs & ~decoder_repairs,
            "decoder_only": disagreement & decoder_repairs & ~encoder_repairs,
            "either_single": disagreement & encoder_repairs & decoder_repairs,
            "both_needed": disagreement & ~encoder_repairs & ~decoder_repairs,
        }
        self.reference_native_disagreements += int(disagreement.sum().item())
        self.accepted_reference_native_disagreements += int(
            (disagreement & reference_accepted).sum().item()
        )
        self.low_entropy_accepted_reference_native_disagreements += int(
            (disagreement & low_entropy_accepted).sum().item()
        )
        positions = torch.arange(valid.size(1), device=valid.device).unsqueeze(0)
        for name, mask in repair_masks.items():
            self.repair_counts[name] += int(mask.sum().item())
            self.accepted_repair_counts[name] += int(
                (mask & reference_accepted).sum().item()
            )
            self.low_entropy_accepted_repair_counts[name] += int(
                (mask & low_entropy_accepted).sum().item()
            )
            self.first128_accepted_repair_counts[name] += int(
                (mask & reference_accepted & positions.lt(128)).sum().item()
            )

        event_rows = (disagreement & reference_accepted).nonzero(as_tuple=False)
        for batch_index, canvas_position in event_rows.tolist():
            if len(self.events) >= 2048:
                break
            repair = next(
                name
                for name, mask in repair_masks.items()
                if bool(mask[batch_index, canvas_position])
            )
            self.events.append(
                {
                    "trajectory": int(trajectory_ids[batch_index].item()),
                    "step": self.steps,
                    "position": canvas_position,
                    "repair": repair,
                    "top1": {
                        name: int(branches[name]["top1"][batch_index, canvas_position].item())
                        for name in self.branch_names
                    },
                    "entropy": {
                        name: float(branches[name]["entropy"][batch_index, canvas_position].item())
                        for name in self.branch_names
                    },
                    "accepted": {
                        name: bool(branches[name]["accept"][batch_index, canvas_position].item())
                        for name in self.branch_names
                    },
                }
            )

    def install(self) -> None:
        outer_encoder = _outer_encoder(self.model)
        if hasattr(outer_encoder, _ORIGINAL_ENCODER_ATTRIBUTE):
            raise RuntimeError("phase comparison is already installed")
        if hasattr(self.model, _ORIGINAL_STEP_ATTRIBUTE):
            raise RuntimeError("phase comparison denoising hook is already installed")
        original_encoder = outer_encoder.forward
        original_step = self.model._denoising_step
        setattr(outer_encoder, _ORIGINAL_ENCODER_ATTRIBUTE, original_encoder)
        setattr(self.model, _ORIGINAL_STEP_ATTRIBUTE, original_step)
        comparator = self

        def dual_encoder_forward(encoder_self: nn.Module, *args: Any, **kwargs: Any):
            primary_cache = kwargs.get("past_key_values")
            if primary_cache is None:
                raise RuntimeError("phase comparison requires an encoder cache")
            shadow_cache = getattr(primary_cache, _SHADOW_CACHE_ATTRIBUTE, None)
            if shadow_cache is None:
                shadow_cache = copy.deepcopy(primary_cache)
                setattr(primary_cache, _SHADOW_CACHE_ATTRIBUTE, shadow_cache)
            shadow_kwargs = dict(kwargs)
            shadow_kwargs["past_key_values"] = shadow_cache
            with _encoder_attention(comparator.model, "native"):
                original_encoder(*args, **shadow_kwargs)
            with _encoder_attention(comparator.model, "lod"):
                primary_outputs = original_encoder(*args, **kwargs)
            return primary_outputs

        def compared_step(model_self: nn.Module, *args: Any, **kwargs: Any):
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
            step_tensor = torch.tensor(
                cur_step, device=current_canvas.device, dtype=torch.int32
            )

            def branch(cache: Any, *, native_decoder: bool) -> dict[str, torch.Tensor]:
                context = (
                    _native_decoder_attention(model_self)
                    if native_decoder
                    else nullcontext()
                )
                with context:
                    outputs = model_self(
                        decoder_input_ids=current_canvas,
                        self_conditioning_logits=self_conditioning_logits,
                        decoder_attention_mask=mask_mapping,
                        past_key_values=cache,
                        decoder_position_ids=decoder_position_ids,
                        **model_kwargs,
                    )
                logits = logits_processor(
                    input_ids, outputs.logits, cur_step=step_tensor
                )
                entropy = _entropy(logits)
                result = {
                    "top1": logits.argmax(dim=-1),
                    "entropy": entropy,
                    "accept": _acceptance_mask(entropy, float(sampler.entropy_bound)),
                }
                del logits, outputs
                return result

            diagnostics = {
                "lod_native": branch(primary_cache, native_decoder=True),
                "native_lod": branch(shadow_cache, native_decoder=False),
                "native_native": branch(shadow_cache, native_decoder=True),
            }
            captured: dict[str, torch.Tensor] = {}
            original_accept = sampler.accept_canvas

            def capture_accept(
                canvas: torch.Tensor,
                denoiser_canvas: torch.Tensor,
                logits: torch.Tensor,
                step: int,
            ) -> torch.Tensor:
                accepted_canvas = original_accept(canvas, denoiser_canvas, logits, step)
                captured["logits"] = logits
                captured["accepted_mask"] = sampler.accepted_token_mask
                return accepted_canvas

            sampler.accept_canvas = capture_accept
            try:
                result = original_step(*args, **kwargs)
            finally:
                sampler.accept_canvas = original_accept

            reference_logits = captured["logits"]
            reference_entropy = _entropy(reference_logits)
            diagnostics["lod_lod"] = {
                "top1": reference_logits.argmax(dim=-1),
                "entropy": reference_entropy,
                "accept": captured["accepted_mask"],
            }
            comparator._record(
                diagnostics,
                captured["accepted_mask"],
                active_batch,
                comparator._trajectory_ids(input_ids),
            )
            return result

        outer_encoder.forward = MethodType(dual_encoder_forward, outer_encoder)
        self.model._denoising_step = MethodType(compared_step, self.model)

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
        positions = self.positions
        accepted = self.reference_accepted
        branches = {}
        for name in self.branch_names:
            union = self.branch_accept_union_native_native[name]
            branches[name] = {
                "mean_entropy": (
                    self.branch_entropy_sum[name] / positions if positions else None
                ),
                "mean_entropy_on_reference_accepted": (
                    self.branch_accepted_entropy_sum[name] / accepted
                    if accepted
                    else None
                ),
                "top1_disagreements_vs_native_native": self.top1_disagree_native_native[name],
                "top1_disagreement_rate_vs_native_native": (
                    self.top1_disagree_native_native[name] / positions
                    if positions
                    else None
                ),
                "reference_accepted_top1_disagreements_vs_native_native": self.accepted_top1_disagree_native_native[name],
                "accept_count": self.branch_accept_count[name],
                "accept_mask_disagreements_vs_native_native": self.branch_accept_disagree_native_native[name],
                "accept_mask_jaccard_vs_native_native": (
                    self.branch_accept_intersection_native_native[name] / union
                    if union
                    else None
                ),
            }
        return {
            "trajectory_controller": "lod_encoder_lod_decoder",
            "steps": self.steps,
            "positions": positions,
            "reference_accepted": accepted,
            "branches": branches,
            "pairwise_top1_disagreements": self.pairwise_top1_disagreements,
            "lod_lod_vs_native_native": {
                "top1_disagreements": self.reference_native_disagreements,
                "reference_accepted_top1_disagreements": self.accepted_reference_native_disagreements,
                "low_entropy_reference_accepted_top1_disagreements": self.low_entropy_accepted_reference_native_disagreements,
                "repair_categories": self.repair_counts,
                "reference_accepted_repair_categories": self.accepted_repair_counts,
                "low_entropy_reference_accepted_repair_categories": self.low_entropy_accepted_repair_counts,
                "first128_reference_accepted_repair_categories": self.first128_accepted_repair_counts,
            },
            "accepted_disagreement_events": self.events,
        }


__all__ = ["DiffusionGemmaPhaseComparator"]
