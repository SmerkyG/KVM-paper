"""lm-eval adapters for Hugging Face DiffusionGemma generation tasks.

DiffusionGemma is a block-diffusion model, so the ordinary ``HFLM`` causal
log-likelihood path is not valid.  This adapter deliberately supports only
``generate_until`` tasks (RULER, GSM8K generation, etc.).  It reuses HFLM's
tokenization, batching, left padding, stop-string handling, and result ordering,
while replacing the model-generation call with DiffusionGemma's API.

The checkpoint also exposes a causal encoder whose hidden states are used for
an auxiliary autoregressive loss during training.  ``DiffusionGemmaARLM``
provides the greedy generation loop that Transformers does not currently
expose for those states.
"""

from __future__ import annotations

from typing import Any

import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils_hf import stop_sequences_criteria


class DiffusionGemmaLM(HFLM):
    """Generation-only lm-eval wrapper for ``DiffusionGemmaForBlockDiffusion``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # DiffusionGemma's composite config nests the actual text limit.  HFLM
        # otherwise falls back to 2K because it cannot discover that field.
        model = kwargs.get("pretrained", args[0] if args else None)
        if kwargs.get("max_length") is None and not isinstance(model, str):
            text_config = model.config.get_text_config(decoder=True)
            kwargs["max_length"] = int(text_config.max_position_embeddings)
        kwargs["backend"] = "causal"
        kwargs["logits_cache"] = False
        super().__init__(*args, **kwargs)
        self._diffusion_generation_requests = 0
        self._diffusion_tokens_per_forward_sum = 0.0

    @staticmethod
    def _unsupported_likelihood() -> NotImplementedError:
        return NotImplementedError(
            "DiffusionGemmaLM only supports generation tasks. AR next-token "
            "log-likelihood is not defined for block diffusion; use a "
            "diffusion-specific Monte Carlo likelihood estimator instead."
        )

    def loglikelihood(self, requests):
        raise self._unsupported_likelihood()

    def loglikelihood_rolling(self, requests):
        raise self._unsupported_likelihood()

    @torch.no_grad()
    def _model_generate(
        self,
        context: torch.Tensor,
        max_length: int,
        stop: list[str],
        **generation_kwargs: Any,
    ) -> torch.Tensor:
        """Call diffusion generation and return an HFLM-compatible tensor.

        DiffusionGemma emits complete canvases, even when ``max_new_tokens`` is
        shorter than one canvas.  Truncating the returned tensor to lm-eval's
        requested length is therefore essential for fair task evaluation.
        """
        attention_mask = generation_kwargs.pop("attention_mask", None)
        requested_new_tokens = max_length - context.shape[1]
        if requested_new_tokens <= 0:
            raise ValueError("lm-eval requested no room for generated tokens")

        # These are conventional AR controls that some task configs inject.
        # DiffusionGemma samples/refines according to its own generation config.
        temperature = generation_kwargs.pop("temperature", None)
        do_sample = generation_kwargs.pop("do_sample", None)
        use_cache = generation_kwargs.pop("use_cache", None)
        if temperature not in (None, 0, 0.0):
            raise ValueError(
                "AR `temperature` is unsupported; configure DiffusionGemma "
                "with `t_min` and `t_max` instead"
            )
        if do_sample not in (None, False):
            raise ValueError(
                "AR `do_sample` is unsupported by DiffusionGemma generation"
            )
        if use_cache not in (None, True):
            raise ValueError("DiffusionGemma evaluation requires its KV cache")

        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        output = self.model.generate(
            input_ids=context,
            attention_mask=attention_mask,
            max_new_tokens=requested_new_tokens,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            **generation_kwargs,
        )
        if isinstance(output, torch.Tensor):
            sequences = output
        else:
            sequences = output.sequences
            tokens_per_forward = getattr(output, "tokens_per_forward", None)
            if tokens_per_forward is not None:
                self._diffusion_generation_requests += int(
                    tokens_per_forward.numel()
                )
                self._diffusion_tokens_per_forward_sum += float(
                    tokens_per_forward.float().sum().item()
                )
        return sequences[:, :max_length]

    @property
    def diffusion_generation_statistics(self) -> dict[str, float | int | None]:
        """Return DiffusionGemma's model-reported generation efficiency."""
        count = self._diffusion_generation_requests
        return {
            "requests": count,
            "mean_tokens_per_forward": (
                self._diffusion_tokens_per_forward_sum / count if count else None
            ),
        }


class DiffusionGemmaARLM(DiffusionGemmaLM):
    """Greedy autoregressive generation through DiffusionGemma's causal encoder."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ar_generation_requests = 0
        self._ar_generated_tokens = 0
        self._ar_forward_passes = 0

    def _encoder_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.model.lm_head(hidden_states).float()
        softcap = float(self.model.final_logit_softcapping)
        return torch.tanh(logits / softcap) * softcap

    @torch.no_grad()
    def _model_generate(
        self,
        context: torch.Tensor,
        max_length: int,
        stop: list[str],
        **generation_kwargs: Any,
    ) -> torch.Tensor:
        """Greedily decode one token at a time from the causal encoder states."""
        del stop  # lm-eval also truncates decoded strings at every stop sequence.
        attention_mask = generation_kwargs.pop("attention_mask", None)
        temperature = generation_kwargs.pop("temperature", None)
        do_sample = generation_kwargs.pop("do_sample", None)
        use_cache = generation_kwargs.pop("use_cache", None)
        if temperature not in (None, 0, 0.0):
            raise ValueError(
                "DiffusionGemma AR evaluation currently supports greedy decoding only"
            )
        if do_sample not in (None, False):
            raise ValueError(
                "DiffusionGemma AR evaluation currently supports greedy decoding only"
            )
        if use_cache not in (None, True):
            raise ValueError(
                "DiffusionGemma AR evaluation requires its causal KV cache"
            )
        if generation_kwargs:
            unknown = ", ".join(sorted(generation_kwargs))
            raise ValueError(
                f"unsupported DiffusionGemma AR generation arguments: {unknown}"
            )

        requested_new_tokens = max_length - context.shape[1]
        if requested_new_tokens <= 0:
            raise ValueError("lm-eval requested no room for generated tokens")
        if attention_mask is None:
            attention_mask = torch.ones_like(context, dtype=torch.long)
        else:
            attention_mask = attention_mask.to(device=context.device)

        encoder = self.model.model.encoder
        encoder_outputs = encoder(
            input_ids=context,
            attention_mask=attention_mask,
        )
        past_key_values = encoder_outputs.past_key_values
        next_logits = self._encoder_logits(encoder_outputs.last_hidden_state[:, -1])
        sequences = context
        batch_size = context.shape[0]
        pad_token_id = int(self.tokenizer.pad_token_id)
        eos_token_ids = self.model.generation_config.eos_token_id
        if eos_token_ids is None:
            eos_token_ids = [self.tokenizer.eos_token_id]
        elif isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        eos = torch.tensor(eos_token_ids, device=context.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=context.device)
        forward_passes = 1
        generated = 0

        for token_index in range(requested_new_tokens):
            next_token = next_logits.argmax(dim=-1)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, pad_token_id),
                next_token,
            )
            sequences = torch.cat((sequences, next_token[:, None]), dim=-1)
            generated += int((~finished).sum().item())
            finished |= (next_token[:, None] == eos[None, :]).any(dim=-1)
            if bool(finished.all()) or token_index + 1 == requested_new_tokens:
                break

            active = (~finished).to(attention_mask.dtype)[:, None]
            attention_mask = torch.cat((attention_mask, active), dim=-1)
            position_ids = torch.full(
                (1, 1),
                context.shape[1] + token_index,
                dtype=torch.long,
                device=context.device,
            )
            encoder_outputs = encoder(
                input_ids=next_token[:, None],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
            )
            past_key_values = encoder_outputs.past_key_values
            next_logits = self._encoder_logits(
                encoder_outputs.last_hidden_state[:, -1]
            )
            forward_passes += 1

        if sequences.shape[1] < max_length:
            padding = torch.full(
                (batch_size, max_length - sequences.shape[1]),
                pad_token_id,
                dtype=sequences.dtype,
                device=sequences.device,
            )
            sequences = torch.cat((sequences, padding), dim=-1)

        self._ar_generation_requests += batch_size
        self._ar_generated_tokens += generated
        self._ar_forward_passes += forward_passes
        return sequences

    @property
    def ar_generation_statistics(self) -> dict[str, float | int | None]:
        requests = self._ar_generation_requests
        return {
            "requests": requests,
            "generated_tokens": self._ar_generated_tokens,
            "forward_passes": self._ar_forward_passes,
            "mean_generated_tokens_per_request": (
                self._ar_generated_tokens / requests if requests else None
            ),
        }


__all__ = ["DiffusionGemmaARLM", "DiffusionGemmaLM"]
