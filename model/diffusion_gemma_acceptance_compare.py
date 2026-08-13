"""Coupled native-vs-LOD diagnostics on DiffusionGemma acceptance steps."""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Any, Iterator

import torch
from torch import nn


_NATIVE_ACTIVE_ATTRIBUTE = "_diffusion_gemma_native_attention_active"
_ORIGINAL_STEP_ATTRIBUTE = "_diffusion_gemma_acceptance_original_step"
_HYBRID_ORIGINAL_STEP_ATTRIBUTE = "_diffusion_gemma_hybrid_original_step"


def _decoder_attention_modules(model: nn.Module) -> list[nn.Module]:
    base = getattr(model, "model", model)
    decoder = getattr(base, "decoder", None)
    if decoder is None:
        raise TypeError("expected a DiffusionGemma model with a decoder")
    return [layer.self_attn for layer in decoder.layers]


@contextmanager
def _native_decoder_attention(model: nn.Module) -> Iterator[None]:
    modules = _decoder_attention_modules(model)
    for module in modules:
        setattr(module, _NATIVE_ACTIVE_ATTRIBUTE, True)
    try:
        yield
    finally:
        for module in modules:
            if hasattr(module, _NATIVE_ACTIVE_ATTRIBUTE):
                delattr(module, _NATIVE_ACTIVE_ATTRIBUTE)


class DiffusionGemmaAcceptanceComparator:
    """Record native disagreement on the exact LOD sampling trajectory."""

    entropy_edges = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
    position_limits = (32, 64, 128, 256)

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.steps = 0
        self.positions = 0
        self.accepted = 0
        self.top1_disagreements = 0
        self.accepted_top1_disagreements = 0
        self.accepted_sample_native_top1_disagreements = 0
        self.accepted_lod_entropy_sum = 0.0
        self.accepted_native_entropy_sum = 0.0
        self.accepted_native_nll_lod_top1_sum = 0.0
        self.accepted_native_nll_sample_sum = 0.0
        self.bin_total = [0 for _ in range(len(self.entropy_edges))]
        self.bin_disagree = [0 for _ in range(len(self.entropy_edges))]
        self.position_accepted = {limit: 0 for limit in self.position_limits}
        self.position_disagree = {limit: 0 for limit in self.position_limits}
        self.position_low_entropy_disagree = {
            limit: 0 for limit in self.position_limits
        }
        self.accepted_disagreement_locations: set[tuple[int, int]] = set()
        self.low_entropy_disagreement_locations: set[tuple[int, int]] = set()
        self.accepted_disagreement_events: list[dict[str, Any]] = []
        self._trajectory_batches: list[tuple[torch.Tensor, int]] = []
        self._next_trajectory = 0

    @staticmethod
    def _entropy(logits: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
        return torch.logsumexp(logits.float(), dim=-1) - (
            probabilities * logits.float()
        ).sum(dim=-1)

    def _record(
        self,
        lod_logits: torch.Tensor,
        native_logits: torch.Tensor,
        accepted_mask: torch.Tensor,
        sampled_tokens: torch.Tensor,
        active_batch: torch.Tensor,
        trajectory_ids: torch.Tensor,
    ) -> None:
        lod_entropy = self._entropy(lod_logits)
        native_entropy = self._entropy(native_logits)
        lod_top1 = lod_logits.argmax(dim=-1)
        native_top1 = native_logits.argmax(dim=-1)
        disagreement = lod_top1.ne(native_top1)
        valid_mask = active_batch.unsqueeze(-1).expand_as(accepted_mask)
        accepted_mask = accepted_mask & valid_mask
        accepted_count = int(accepted_mask.sum().item())

        self.steps += 1
        self.positions += int(valid_mask.sum().item())
        self.accepted += accepted_count
        self.top1_disagreements += int(
            (disagreement & valid_mask).sum().item()
        )
        self.accepted_top1_disagreements += int(
            (disagreement & accepted_mask).sum().item()
        )
        self.accepted_sample_native_top1_disagreements += int(
            (sampled_tokens.ne(native_top1) & accepted_mask).sum().item()
        )
        position = torch.arange(
            accepted_mask.size(1), device=accepted_mask.device
        ).unsqueeze(0)
        accepted_disagreement = accepted_mask & disagreement
        low_entropy_disagreement = accepted_disagreement & lod_entropy.lt(0.001)
        for limit in self.position_limits:
            within = position < limit
            self.position_accepted[limit] += int(
                (accepted_mask & within).sum().item()
            )
            self.position_disagree[limit] += int(
                (accepted_disagreement & within).sum().item()
            )
            self.position_low_entropy_disagree[limit] += int(
                (low_entropy_disagreement & within).sum().item()
            )

        event_rows = accepted_disagreement.nonzero(as_tuple=False)
        for batch_index, canvas_position in event_rows.tolist():
            trajectory = int(trajectory_ids[batch_index].item())
            location = (trajectory, canvas_position)
            self.accepted_disagreement_locations.add(location)
            event_entropy = float(lod_entropy[batch_index, canvas_position].item())
            if event_entropy < 0.001:
                self.low_entropy_disagreement_locations.add(location)
            if len(self.accepted_disagreement_events) < 2048:
                self.accepted_disagreement_events.append(
                    {
                        "trajectory": trajectory,
                        "step": self.steps,
                        "position": canvas_position,
                        "lod_entropy": event_entropy,
                        "native_entropy": float(
                            native_entropy[batch_index, canvas_position].item()
                        ),
                        "lod_top1": int(lod_top1[batch_index, canvas_position].item()),
                        "native_top1": int(
                            native_top1[batch_index, canvas_position].item()
                        ),
                        "sampled_token": int(
                            sampled_tokens[batch_index, canvas_position].item()
                        ),
                    }
                )
        if accepted_count:
            self.accepted_lod_entropy_sum += float(
                lod_entropy.masked_select(accepted_mask).sum().item()
            )
            self.accepted_native_entropy_sum += float(
                native_entropy.masked_select(accepted_mask).sum().item()
            )
            native_log_normalizer = torch.logsumexp(
                native_logits.float(), dim=-1
            )
            native_nll_lod_top1 = native_log_normalizer - native_logits.gather(
                -1, lod_top1.unsqueeze(-1)
            ).squeeze(-1).float()
            native_nll_sample = native_log_normalizer - native_logits.gather(
                -1, sampled_tokens.unsqueeze(-1)
            ).squeeze(-1).float()
            self.accepted_native_nll_lod_top1_sum += float(
                native_nll_lod_top1.masked_select(accepted_mask).sum().item()
            )
            self.accepted_native_nll_sample_sum += float(
                native_nll_sample.masked_select(accepted_mask).sum().item()
            )

        accepted_entropy = lod_entropy.masked_select(accepted_mask)
        accepted_disagreement = disagreement.masked_select(accepted_mask)
        if accepted_count:
            edges = torch.tensor(
                self.entropy_edges[1:],
                dtype=accepted_entropy.dtype,
                device=accepted_entropy.device,
            )
            bins = torch.bucketize(accepted_entropy, edges)
            totals = torch.bincount(bins, minlength=len(self.entropy_edges))
            disagrees = torch.bincount(
                bins,
                weights=accepted_disagreement.float(),
                minlength=len(self.entropy_edges),
            )
            for index in range(len(self.entropy_edges)):
                self.bin_total[index] += int(totals[index].item())
                self.bin_disagree[index] += int(disagrees[index].item())

    def install(self) -> None:
        if hasattr(self.model, _ORIGINAL_STEP_ATTRIBUTE):
            raise RuntimeError("acceptance comparison is already installed")
        original_step = self.model._denoising_step
        setattr(self.model, _ORIGINAL_STEP_ATTRIBUTE, original_step)
        comparator = self

        def trajectory_ids(input_ids: torch.Tensor) -> torch.Tensor:
            for previous, first_id in comparator._trajectory_batches:
                if previous is input_ids:
                    return torch.arange(
                        first_id,
                        first_id + int(input_ids.size(0)),
                        device=input_ids.device,
                    )
            first_id = comparator._next_trajectory
            comparator._next_trajectory += int(input_ids.size(0))
            comparator._trajectory_batches.append((input_ids, first_id))
            return torch.arange(
                first_id,
                first_id + int(input_ids.size(0)),
                device=input_ids.device,
            )

        def compared_step(model_self: nn.Module, *args: Any, **kwargs: Any):
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
            with _native_decoder_attention(model_self):
                native_outputs = model_self(
                    decoder_input_ids=current_canvas,
                    self_conditioning_logits=self_conditioning_logits,
                    decoder_attention_mask=mask_mapping,
                    past_key_values=past_key_values,
                    decoder_position_ids=decoder_position_ids,
                    **model_kwargs,
                )
            native_raw_logits = native_outputs.logits

            captured: dict[str, torch.Tensor] = {}
            original_accept = sampler.accept_canvas

            def capture_accept(
                canvas: torch.Tensor,
                denoiser_canvas: torch.Tensor,
                logits: torch.Tensor,
                step: int,
            ) -> torch.Tensor:
                accepted_canvas = original_accept(
                    canvas, denoiser_canvas, logits, step
                )
                captured["logits"] = logits
                captured["sampled_tokens"] = denoiser_canvas
                captured["accepted_mask"] = sampler.accepted_token_mask
                return accepted_canvas

            sampler.accept_canvas = capture_accept
            try:
                result = original_step(*args, **kwargs)
            finally:
                sampler.accept_canvas = original_accept

            step_tensor = torch.tensor(
                cur_step, device=current_canvas.device, dtype=torch.int32
            )
            native_logits = logits_processor(
                input_ids, native_raw_logits, cur_step=step_tensor
            )
            comparator._record(
                captured["logits"],
                native_logits,
                captured["accepted_mask"],
                captured["sampled_tokens"],
                active_batch,
                trajectory_ids(input_ids),
            )
            return result

        self.model._denoising_step = MethodType(compared_step, self.model)

    def uninstall(self) -> None:
        original = getattr(self.model, _ORIGINAL_STEP_ATTRIBUTE, None)
        if original is None:
            return
        self.model._denoising_step = original
        delattr(self.model, _ORIGINAL_STEP_ATTRIBUTE)

    def summary(self) -> dict[str, Any]:
        accepted = self.accepted
        bins = []
        for index, lower in enumerate(self.entropy_edges):
            upper = (
                self.entropy_edges[index + 1]
                if index + 1 < len(self.entropy_edges)
                else None
            )
            total = self.bin_total[index]
            disagree = self.bin_disagree[index]
            bins.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "accepted": total,
                    "top1_disagreements": disagree,
                    "top1_disagreement_rate": disagree / total if total else None,
                }
            )
        return {
            "steps": self.steps,
            "positions": self.positions,
            "accepted": accepted,
            "acceptance_rate": accepted / self.positions if self.positions else None,
            "all_position_top1_disagreement_rate": (
                self.top1_disagreements / self.positions if self.positions else None
            ),
            "accepted_top1_disagreements": self.accepted_top1_disagreements,
            "accepted_top1_disagreement_rate": (
                self.accepted_top1_disagreements / accepted if accepted else None
            ),
            "accepted_sample_native_top1_disagreement_rate": (
                self.accepted_sample_native_top1_disagreements / accepted
                if accepted
                else None
            ),
            "accepted_mean_lod_entropy": (
                self.accepted_lod_entropy_sum / accepted if accepted else None
            ),
            "accepted_mean_native_entropy": (
                self.accepted_native_entropy_sum / accepted if accepted else None
            ),
            "accepted_mean_native_nll_lod_top1": (
                self.accepted_native_nll_lod_top1_sum / accepted
                if accepted
                else None
            ),
            "accepted_mean_native_nll_sample": (
                self.accepted_native_nll_sample_sum / accepted
                if accepted
                else None
            ),
            "accepted_lod_entropy_bins": bins,
            "position_prefixes": {
                str(limit): {
                    "accepted": self.position_accepted[limit],
                    "top1_disagreements": self.position_disagree[limit],
                    "top1_disagreement_rate": (
                        self.position_disagree[limit]
                        / self.position_accepted[limit]
                        if self.position_accepted[limit]
                        else None
                    ),
                    "low_entropy_top1_disagreements": (
                        self.position_low_entropy_disagree[limit]
                    ),
                }
                for limit in self.position_limits
            },
            "unique_accepted_disagreement_locations": len(
                self.accepted_disagreement_locations
            ),
            "unique_low_entropy_disagreement_locations": len(
                self.low_entropy_disagreement_locations
            ),
            "accepted_disagreement_events": self.accepted_disagreement_events,
        }


class DiffusionGemmaEarlyNativeController:
    """Use native decoder attention for the first few steps of each canvas.

    The causal encoder remains on its installed attention backend.  Only the
    decoder call inside selected denoising steps is switched, so this is a
    diagnostic of high-noise canvas routing rather than a native-prefill run.
    """

    def __init__(self, model: nn.Module, *, early_steps: int) -> None:
        if early_steps < 1:
            raise ValueError("early native steps must be positive")
        self.model = model
        self.early_steps = early_steps
        self.native_step_calls = 0
        self.lod_step_calls = 0
        self.canvases = 0
        self._current_input_ids: torch.Tensor | None = None
        self._canvas_step = 0

    def install(self) -> None:
        if hasattr(self.model, _HYBRID_ORIGINAL_STEP_ATTRIBUTE):
            raise RuntimeError("early-native control is already installed")
        if hasattr(self.model, _ORIGINAL_STEP_ATTRIBUTE):
            raise RuntimeError(
                "early-native control cannot be combined with acceptance comparison"
            )
        original_step = self.model._denoising_step
        setattr(self.model, _HYBRID_ORIGINAL_STEP_ATTRIBUTE, original_step)
        controller = self

        def hybrid_step(model_self: nn.Module, *args: Any, **kwargs: Any):
            input_ids = kwargs["input_ids"]
            if input_ids is not controller._current_input_ids:
                controller._current_input_ids = input_ids
                controller._canvas_step = 0
                controller.canvases += 1
            use_native = controller._canvas_step < controller.early_steps
            controller._canvas_step += 1
            if use_native:
                controller.native_step_calls += 1
                with _native_decoder_attention(model_self):
                    return original_step(*args, **kwargs)
            controller.lod_step_calls += 1
            return original_step(*args, **kwargs)

        self.model._denoising_step = MethodType(hybrid_step, self.model)

    def uninstall(self) -> None:
        original = getattr(self.model, _HYBRID_ORIGINAL_STEP_ATTRIBUTE, None)
        if original is None:
            return
        self.model._denoising_step = original
        delattr(self.model, _HYBRID_ORIGINAL_STEP_ATTRIBUTE)

    def summary(self) -> dict[str, int]:
        return {
            "early_native_steps": self.early_steps,
            "canvases": self.canvases,
            "native_step_calls": self.native_step_calls,
            "lod_step_calls": self.lod_step_calls,
        }


__all__ = [
    "DiffusionGemmaAcceptanceComparator",
    "DiffusionGemmaEarlyNativeController",
]
