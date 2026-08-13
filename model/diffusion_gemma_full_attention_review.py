"""Primary-first full-attention review for DiffusionGemma LOD acceptance."""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
from torch import nn

from .diffusion_gemma_acceptance_compare import _native_decoder_attention


_ORIGINAL_STEP_ATTRIBUTE = "_diffusion_gemma_full_review_original_step"


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return torch.logsumexp(logits.float(), dim=-1) - (
        probabilities * logits.float()
    ).sum(dim=-1)


def _acceptance_mask(entropy: torch.Tensor, entropy_bound: float) -> torch.Tensor:
    sorted_entropy, sorted_indices = torch.sort(entropy, dim=-1)
    selected = (
        torch.cumsum(sorted_entropy, dim=-1) - sorted_entropy
        <= entropy_bound
    )
    return torch.scatter(
        torch.zeros_like(selected),
        dim=-1,
        index=sorted_indices,
        src=selected,
    )


class DiffusionGemmaFullAttentionReviewer:
    """Review false-confident LOD acceptances with native decoder attention."""

    def __init__(
        self,
        model: nn.Module,
        *,
        lod_entropy_threshold: float = 0.001,
        mode: str = "apply",
        policy: str = "sample_top1",
    ) -> None:
        if lod_entropy_threshold < 0.0:
            raise ValueError("LOD entropy threshold cannot be negative")
        if mode not in ("observe", "apply"):
            raise ValueError(f"unknown full-attention review mode {mode!r}")
        if policy not in ("sample_top1", "native_acceptance"):
            raise ValueError(f"unknown full-attention review policy {policy!r}")
        self.model = model
        self.lod_entropy_threshold = lod_entropy_threshold
        self.mode = mode
        self.policy = policy
        self.steps = 0
        self.full_attention_passes = 0
        self.positions = 0
        self.original_accepted = 0
        self.reviewed = 0
        self.reviewed_top1_disagreements = 0
        self.reviewed_sample_disagreements = 0
        self.reviewed_native_rejections = 0
        self.hypothetical_vetoes = 0
        self.final_accepted = 0
        self.primary_entropy_reviewed_sum = 0.0
        self.native_entropy_reviewed_sum = 0.0

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
        if hasattr(self.model, _ORIGINAL_STEP_ATTRIBUTE):
            raise RuntimeError("full-attention review is already installed")
        original_step = self.model._denoising_step
        setattr(self.model, _ORIGINAL_STEP_ATTRIBUTE, original_step)
        reviewer = self

        def reviewed_step(model_self: nn.Module, *args: Any, **kwargs: Any):
            if args:
                raise TypeError("full-attention review expects keyword denoising inputs")
            decoder_forward = kwargs["decoder_forward"]
            current_canvas = kwargs["current_canvas"]
            argmax_canvas = kwargs["argmax_canvas"]
            input_ids = kwargs["input_ids"]
            decoder_position_ids = kwargs["decoder_position_ids"]
            self_conditioning_logits = kwargs["self_conditioning_logits"]
            mask_mapping = kwargs["mask_mapping"]
            past_key_values = kwargs["past_key_values"]
            finished_denoising = kwargs["finished_denoising"]
            sampler = kwargs["sampler"]
            logits_processor = kwargs["logits_processor"]
            diffusion_stopping_criteria = kwargs["diffusion_stopping_criteria"]
            model_kwargs = reviewer._model_kwargs(kwargs)
            active_batch = ~finished_denoising
            step_tensor = torch.tensor(
                kwargs["cur_step"],
                device=current_canvas.device,
                dtype=torch.int32,
            )
            torch.compiler.cudagraph_mark_step_begin()

            # Primary LOD attention always runs first. In particular, no
            # diagnostic/native kernel can perturb this step's LOD logits.
            primary_outputs = decoder_forward(
                decoder_input_ids=current_canvas,
                self_conditioning_logits=self_conditioning_logits,
                decoder_attention_mask=mask_mapping,
                past_key_values=past_key_values,
                decoder_position_ids=decoder_position_ids,
                **model_kwargs,
            )
            primary_logits = logits_processor(
                input_ids, primary_outputs.logits, cur_step=step_tensor
            )
            primary_entropy = _entropy(primary_logits)
            primary_top1 = primary_logits.argmax(dim=-1)
            probabilities = torch.softmax(
                primary_logits, dim=-1, dtype=torch.float32
            )
            vocab_size = int(model_self.config.text_config.vocab_size)
            batch_size, canvas_length = current_canvas.shape
            denoiser_canvas = torch.multinomial(
                probabilities.view(-1, vocab_size), num_samples=1
            ).view(batch_size, canvas_length)

            original_canvas = sampler.accept_canvas(
                current_canvas, denoiser_canvas, primary_logits, step_tensor
            )
            original_mask = sampler.accepted_token_mask.clone()
            valid = active_batch.unsqueeze(-1).expand_as(original_mask)
            review_mask = (
                original_mask
                & valid
                & primary_entropy.le(reviewer.lod_entropy_threshold)
            )

            native_logits = None
            native_entropy = None
            native_accept = None
            native_top1 = None
            if bool(review_mask.any()):
                with _native_decoder_attention(model_self):
                    native_outputs = model_self(
                        decoder_input_ids=current_canvas,
                        self_conditioning_logits=self_conditioning_logits,
                        decoder_attention_mask=mask_mapping,
                        past_key_values=past_key_values,
                        decoder_position_ids=decoder_position_ids,
                        **model_kwargs,
                    )
                native_logits = logits_processor(
                    input_ids, native_outputs.logits, cur_step=step_tensor
                )
                native_entropy = _entropy(native_logits)
                native_top1 = native_logits.argmax(dim=-1)
                native_accept = _acceptance_mask(
                    native_entropy, float(sampler.entropy_bound)
                )
                reviewer.full_attention_passes += 1
                del native_outputs

            if native_logits is None:
                hypothetical_veto = torch.zeros_like(original_mask)
                mixed_logits = primary_logits
            else:
                sample_disagreement = denoiser_canvas.ne(native_top1)
                if reviewer.policy == "sample_top1":
                    hypothetical_veto = review_mask & sample_disagreement
                else:
                    native_approved = native_accept & ~sample_disagreement
                    hypothetical_veto = review_mask & ~native_approved
                mixed_logits = torch.where(
                    hypothetical_veto.unsqueeze(-1), native_logits, primary_logits
                )

            final_mask = original_mask & ~hypothetical_veto
            if reviewer.mode == "observe":
                final_mask = original_mask
                mixed_logits = primary_logits
                accepted_canvas = original_canvas
            else:
                accepted_canvas = torch.where(
                    final_mask, denoiser_canvas, current_canvas
                )
            sampler.accepted_token_mask = final_mask
            accepted_canvas = accepted_canvas.clone()
            new_current_canvas = sampler.renoise_canvas(
                accepted_canvas, step_tensor
            ).clone()
            new_argmax_canvas = mixed_logits.argmax(dim=-1)

            if diffusion_stopping_criteria is not None:
                if finished_denoising.any():
                    new_argmax_canvas = torch.where(
                        finished_denoising[:, None],
                        argmax_canvas,
                        new_argmax_canvas,
                    )
                    new_current_canvas = torch.where(
                        finished_denoising[:, None],
                        current_canvas,
                        new_current_canvas,
                    )
                    mixed_logits = torch.where(
                        finished_denoising[:, None, None],
                        self_conditioning_logits,
                        mixed_logits,
                    )
                finished_denoising |= diffusion_stopping_criteria(
                    new_argmax_canvas, mixed_logits
                )

            reviewer.steps += 1
            reviewer.positions += int(valid.sum().item())
            reviewer.original_accepted += int((original_mask & valid).sum().item())
            reviewer.reviewed += int(review_mask.sum().item())
            reviewer.hypothetical_vetoes += int(hypothetical_veto.sum().item())
            reviewer.final_accepted += int((final_mask & valid).sum().item())
            if native_logits is not None:
                reviewer.reviewed_top1_disagreements += int(
                    (review_mask & primary_top1.ne(native_top1)).sum().item()
                )
                reviewer.reviewed_sample_disagreements += int(
                    (review_mask & denoiser_canvas.ne(native_top1)).sum().item()
                )
                reviewer.reviewed_native_rejections += int(
                    (review_mask & ~native_accept).sum().item()
                )
                reviewer.primary_entropy_reviewed_sum += float(
                    primary_entropy.masked_select(review_mask).sum().item()
                )
                reviewer.native_entropy_reviewed_sum += float(
                    native_entropy.masked_select(review_mask).sum().item()
                )

            embeddings_dtype = model_self.model.decoder.embed_tokens.weight.dtype
            next_self_conditioning = mixed_logits.to(embeddings_dtype)
            del primary_outputs
            return (
                new_current_canvas,
                new_argmax_canvas,
                next_self_conditioning,
                finished_denoising,
            )

        self.model._denoising_step = MethodType(reviewed_step, self.model)

    def uninstall(self) -> None:
        original = getattr(self.model, _ORIGINAL_STEP_ATTRIBUTE, None)
        if original is None:
            return
        self.model._denoising_step = original
        delattr(self.model, _ORIGINAL_STEP_ATTRIBUTE)

    def summary(self) -> dict[str, Any]:
        reviewed = self.reviewed
        original = self.original_accepted
        return {
            "mode": self.mode,
            "policy": self.policy,
            "lod_entropy_threshold": self.lod_entropy_threshold,
            "steps": self.steps,
            "full_attention_passes": self.full_attention_passes,
            "positions": self.positions,
            "original_accepted": original,
            "reviewed": reviewed,
            "review_rate_of_original_accepted": (
                reviewed / original if original else None
            ),
            "reviewed_top1_disagreements": self.reviewed_top1_disagreements,
            "reviewed_top1_disagreement_rate": (
                self.reviewed_top1_disagreements / reviewed if reviewed else None
            ),
            "reviewed_sample_disagreements": self.reviewed_sample_disagreements,
            "reviewed_sample_disagreement_rate": (
                self.reviewed_sample_disagreements / reviewed if reviewed else None
            ),
            "reviewed_native_rejections": self.reviewed_native_rejections,
            "hypothetical_vetoes": self.hypothetical_vetoes,
            "hypothetical_veto_rate": (
                self.hypothetical_vetoes / reviewed if reviewed else None
            ),
            "final_accepted": self.final_accepted,
            "mean_primary_entropy_reviewed": (
                self.primary_entropy_reviewed_sum / reviewed if reviewed else None
            ),
            "mean_native_entropy_reviewed": (
                self.native_entropy_reviewed_sum / reviewed if reviewed else None
            ),
        }


__all__ = ["DiffusionGemmaFullAttentionReviewer"]
