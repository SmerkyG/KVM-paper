"""Session-affine prefix reuse for the sequential HF Serve path."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any

import torch
from transformers.cli.serving.utils import (
    DirectStreamer,
    GenerateManager,
    GenerationState,
    _GenerationCancelled,
    _StreamError,
)
from transformers.generation.continuous_batching.cache_manager import BlockManager

from .hf_pytorch_lod_attention import new_hf_lod_cache


LOD_SESSION_HEADER = "X-LOD-Session-ID"
# A NUL cannot arrive in an HTTP header, so automatic identifiers cannot
# collide with an explicit X-LOD-Session-ID.
_AUTOMATIC_SESSION_PREFIX = "\x00lod-auto:"
_AUTOMATIC_HASH_BLOCK_SIZE = 8
_LOD_SESSION_ID: ContextVar[str | None] = ContextVar(
    "hf_lod_session_id", default=None
)


def bind_lod_session(session_id: str | None) -> Token:
    """Bind a request's session ID until its generation call is submitted."""
    return _LOD_SESSION_ID.set(session_id)


def reset_lod_session(token: Token) -> None:
    _LOD_SESSION_ID.reset(token)


def current_lod_session() -> str | None:
    return _LOD_SESSION_ID.get()


@dataclass(frozen=True)
class LODSessionCacheConfig:
    """Bounds for GPU-resident, conversation-affine LOD caches."""

    max_sessions: int = 8
    ttl_seconds: float = 3600.0
    automatic_discovery: bool = True

    def __post_init__(self) -> None:
        if self.max_sessions < 0:
            raise ValueError("max_sessions cannot be negative")
        if self.ttl_seconds < 0:
            raise ValueError("ttl_seconds cannot be negative")


@dataclass
class _SessionEntry:
    cache: Any
    tokens: torch.Tensor
    pending_tokens: torch.Tensor
    request_tokens: torch.Tensor
    generated_tokens: torch.Tensor
    last_used: float


@dataclass
class _PreparedGeneration:
    inputs: dict[str, Any]
    cache: Any
    full_input_ids: torch.Tensor
    logical_input_ids: torch.Tensor
    generation_input_length: int


class _DeferredEndStreamer(DirectStreamer):
    """Publish the SSE terminator only after the session cache is committed."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._finished = False

    def end(self) -> None:
        # ``model.generate`` calls this before returning. The manager still
        # needs that return value to update the session's represented tokens.
        return

    def finish(self) -> None:
        if not self._finished:
            self._finished = True
            super().end()


class LODSessionGenerateManager(GenerateManager):
    """Reuse one mutable LOD cache along explicit or inferred request streams."""

    def __init__(self, config: LODSessionCacheConfig) -> None:
        super().__init__()
        self.config = config
        self._sessions: OrderedDict[str, _SessionEntry] = OrderedDict()
        # Transformers continuous batching uses this same chained block hash
        # to locate shareable paged-cache prefixes. LOD cannot share those
        # physical blocks, but can reuse the index to locate its own mutable
        # per-conversation caches before doing strict token verification.
        self._prefix_hasher = BlockManager(
            num_blocks=0,
            block_size=_AUTOMATIC_HASH_BLOCK_SIZE,
            tp_on=False,
        )
        self._prefix_index: dict[int, set[str]] = {}
        self._session_hashes: dict[str, tuple[int, ...]] = {}
        self._automatic_sessions: set[str] = set()
        self._next_automatic_session = 0
        self._hits = 0
        self._misses = 0
        self._prefix_mismatches = 0
        self._evictions = 0
        self._reused_tokens = 0
        self._turn_reconciliations = 0
        self._automatic_hits = 0
        self._automatic_misses = 0

    @property
    def enabled(self) -> bool:
        return self.config.max_sessions > 0

    def cache_stats(self) -> dict[str, int]:
        return {
            "active_sessions": len(self._sessions),
            "hits": self._hits,
            "misses": self._misses,
            "prefix_mismatches": self._prefix_mismatches,
            "evictions": self._evictions,
            "reused_tokens": self._reused_tokens,
            "turn_reconciliations": self._turn_reconciliations,
            "automatic_sessions": len(self._automatic_sessions),
            "automatic_hits": self._automatic_hits,
            "automatic_misses": self._automatic_misses,
        }

    def clear_sessions(self) -> None:
        self._sessions.clear()
        self._prefix_index.clear()
        self._session_hashes.clear()
        self._automatic_sessions.clear()

    def _unindex_session(self, session_id: str) -> None:
        for prefix_hash in self._session_hashes.pop(session_id, ()):
            session_ids = self._prefix_index.get(prefix_hash)
            if session_ids is None:
                continue
            session_ids.discard(session_id)
            if not session_ids:
                self._prefix_index.pop(prefix_hash, None)

    def _remove_session(self, session_id: str) -> None:
        self._unindex_session(session_id)
        self._sessions.pop(session_id, None)
        self._automatic_sessions.discard(session_id)

    def _drop_session(self, session_id: str) -> None:
        self._remove_session(session_id)

    def _evict_expired(self, now: float) -> None:
        if self.config.ttl_seconds == 0:
            expired = list(self._sessions)
        else:
            expired = [
                session_id
                for session_id, entry in self._sessions.items()
                if now - entry.last_used >= self.config.ttl_seconds
            ]
        for session_id in expired:
            self._remove_session(session_id)
            self._evictions += 1

    def _make_room(self, session_id: str, now: float) -> None:
        self._evict_expired(now)
        if session_id in self._sessions:
            return
        while len(self._sessions) >= self.config.max_sessions:
            oldest_session_id = next(iter(self._sessions))
            self._remove_session(oldest_session_id)
            self._evictions += 1

    def _prefix_hashes(self, tokens: torch.Tensor) -> tuple[int, ...]:
        token_ids = tokens.detach().cpu().tolist()
        parent_hash = None
        hashes = []
        for start in range(
            0,
            len(token_ids) - _AUTOMATIC_HASH_BLOCK_SIZE + 1,
            _AUTOMATIC_HASH_BLOCK_SIZE,
        ):
            block = token_ids[start : start + _AUTOMATIC_HASH_BLOCK_SIZE]
            parent_hash = self._prefix_hasher.compute_hash(
                parent_hash, block, group_id=0
            )
            hashes.append(parent_hash)
        return tuple(hashes)

    def _index_automatic_session(
        self, session_id: str, entry: _SessionEntry
    ) -> None:
        self._unindex_session(session_id)
        hashes = self._prefix_hashes(entry.request_tokens)
        self._session_hashes[session_id] = hashes
        for prefix_hash in hashes:
            self._prefix_index.setdefault(prefix_hash, set()).add(session_id)

    def _entry_matches_request(
        self,
        entry: _SessionEntry,
        inputs: dict[str, Any],
        full_input_ids: torch.Tensor,
        processor,
    ) -> bool:
        cached_length = int(entry.tokens.numel())
        full_length = int(full_input_ids.size(-1))
        if (
            0 < cached_length < full_length
            and entry.tokens.device == full_input_ids.device
            and torch.equal(full_input_ids[0, :cached_length], entry.tokens)
        ):
            return True
        return (
            self._reconcile_chat_turn(
                entry, inputs, full_input_ids, processor
            )
            is not None
        )

    def _new_automatic_session_id(self) -> str:
        while True:
            self._next_automatic_session += 1
            session_id = (
                f"{_AUTOMATIC_SESSION_PREFIX}{self._next_automatic_session}"
            )
            if session_id not in self._sessions:
                self._automatic_sessions.add(session_id)
                return session_id

    def _resolve_automatic_session(
        self, inputs: dict[str, Any], processor
    ) -> str:
        full_input_ids = inputs.get("input_ids")
        if not isinstance(full_input_ids, torch.Tensor):
            raise TypeError("LOD session caching requires tensor input_ids")
        if full_input_ids.ndim != 2 or int(full_input_ids.size(0)) != 1:
            raise ValueError("LOD session caching requires a single prompt row")

        self._evict_expired(time.monotonic())
        checked: set[str] = set()
        prefix_hashes = self._prefix_hashes(full_input_ids[0])
        for prefix_hash in reversed(prefix_hashes):
            indexed = self._prefix_index.get(prefix_hash, ())
            for session_id in reversed(self._sessions):
                if session_id in checked or session_id not in indexed:
                    continue
                checked.add(session_id)
                entry = self._sessions[session_id]
                if self._entry_matches_request(
                    entry, inputs, full_input_ids, processor
                ):
                    self._automatic_hits += 1
                    return session_id

        # Short prompts and chat templates may diverge inside the first hash
        # block. The cache is deliberately small, so a verified fallback scan
        # preserves correctness without turning normal lookup into O(history).
        for session_id in reversed(self._sessions):
            if (
                session_id in checked
                or session_id not in self._automatic_sessions
            ):
                continue
            entry = self._sessions[session_id]
            if self._entry_matches_request(
                entry, inputs, full_input_ids, processor
            ):
                self._automatic_hits += 1
                return session_id

        self._automatic_misses += 1
        return self._new_automatic_session_id()

    @staticmethod
    def _slice_prompt_inputs(
        inputs: dict[str, Any], prefix_length: int, full_length: int
    ) -> dict[str, Any]:
        sliced = dict(inputs)
        sliced["input_ids"] = inputs["input_ids"][..., prefix_length:]
        # The attention mask describes both the retained cache and new suffix.
        # Position-like tensors, in contrast, align only with new input IDs.
        for name in ("position_ids", "token_type_ids"):
            value = sliced.get(name)
            if (
                isinstance(value, torch.Tensor)
                and value.ndim >= 2
                and int(value.size(-1)) == full_length
            ):
                sliced[name] = value[..., prefix_length:]
        return sliced

    @staticmethod
    def _common_prefix_length(left: torch.Tensor, right: torch.Tensor) -> int:
        limit = min(int(left.numel()), int(right.numel()))
        if limit == 0:
            return 0
        differences = torch.nonzero(left[:limit] != right[:limit])
        return limit if differences.numel() == 0 else int(differences[0, 0])

    @staticmethod
    def _continuation_inputs(
        inputs: dict[str, Any], suffix: torch.Tensor, cached_length: int
    ) -> dict[str, Any]:
        continued = dict(inputs)
        continued["input_ids"] = suffix.unsqueeze(0)
        total_length = cached_length + int(suffix.numel())
        attention_mask = continued.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor):
            if attention_mask.ndim != 2:
                raise ValueError(
                    "LOD session continuation requires a 2D attention mask"
                )
            continued["attention_mask"] = attention_mask.new_ones(
                (1, total_length)
            )
        position_ids = continued.get("position_ids")
        if isinstance(position_ids, torch.Tensor):
            continued["position_ids"] = torch.arange(
                cached_length,
                total_length,
                dtype=position_ids.dtype,
                device=position_ids.device,
            ).unsqueeze(0)
        cache_position = continued.get("cache_position")
        if isinstance(cache_position, torch.Tensor):
            continued["cache_position"] = torch.arange(
                cached_length,
                total_length,
                dtype=cache_position.dtype,
                device=cache_position.device,
            )
        return continued

    def _reconcile_chat_turn(
        self,
        entry: _SessionEntry,
        inputs: dict[str, Any],
        full_input_ids: torch.Tensor,
        processor,
    ) -> tuple[dict[str, Any], torch.Tensor] | None:
        """Continue past a chat template whose assistant prefix is not stable.

        Some templates put control tokens in the generation prompt but omit
        them when the same assistant answer is rendered as history.  The raw
        generated answer nevertheless begins exactly where the two renderings
        first diverge.  Retain the state that actually produced that answer,
        consume its one-token cache lag, and append only the new turn.
        """
        previous_request = entry.request_tokens
        generated = entry.generated_tokens
        current = full_input_ids[0]
        shared = self._common_prefix_length(previous_request, current)
        generated_length = int(generated.numel())
        rendered_match = self._rendered_generated_end(
            processor, generated, current[shared:]
        )
        if (
            shared == 0
            or generated_length == 0
            or rendered_match is None
        ):
            return None
        generated_end, matched_generated_length = rendered_match
        generated_end += shared
        dropped_generated = generated_length - matched_generated_length
        pending_length = int(entry.pending_tokens.numel())
        if dropped_generated > pending_length:
            # The cache already contains a terminal token omitted by the chat
            # history, and LOD caches cannot be rolled back partially.
            return None
        pending = entry.pending_tokens[: pending_length - dropped_generated]
        suffix = torch.cat((pending, current[generated_end:]))
        if suffix.numel() == 0:
            return None
        cached_length = int(entry.tokens.numel())
        continued = self._continuation_inputs(inputs, suffix, cached_length)
        logical_input_ids = torch.cat((entry.tokens, suffix)).unsqueeze(0)
        return continued, logical_input_ids

    @staticmethod
    def _rendered_generated_end(
        processor, generated: torch.Tensor, rendered_tail: torch.Tensor
    ) -> tuple[int, int] | None:
        generated_length = int(generated.numel())
        if generated_length == 0:
            return None
        if (
            generated_length <= int(rendered_tail.numel())
            and generated.device == rendered_tail.device
            and torch.equal(rendered_tail[:generated_length], generated)
        ):
            return generated_length, generated_length

        # Tool-call parsers may normalize insignificant whitespace before the
        # structured call is rendered back into chat history. Chat templates
        # also omit the final EOS generated after their own turn delimiter.
        # Accept only these forms at the exact template divergence.
        decode = getattr(processor, "decode", None)
        if not callable(decode):
            return None
        tokenizer = getattr(processor, "tokenizer", processor)
        eos_token_ids = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos_token_ids, int):
            eos_token_ids = {eos_token_ids}
        elif eos_token_ids is None:
            eos_token_ids = set()
        else:
            eos_token_ids = set(eos_token_ids)
        eos_token_ids.update(getattr(tokenizer, "all_special_ids", ()))
        generated_ids = generated.detach().cpu().tolist()
        candidate_lengths = [generated_length]
        trimmed_length = generated_length
        while (
            trimmed_length > 0
            and generated_ids[trimmed_length - 1] in eos_token_ids
        ):
            trimmed_length -= 1
            candidate_lengths.append(trimmed_length)

        scan_limit = min(
            int(rendered_tail.numel()),
            max(generated_length * 2 + 32, generated_length + 256),
        )
        tail_ids = rendered_tail[:scan_limit].detach().cpu().tolist()
        for matched_length in candidate_lengths:
            if matched_length == 0:
                continue
            target_ids = generated_ids[:matched_length]
            if (
                matched_length <= len(tail_ids)
                and tail_ids[:matched_length] == target_ids
            ):
                return matched_length, matched_length
            target = "".join(
                decode(target_ids, skip_special_tokens=False).split()
            )
            if not target:
                continue
            matched_end = None
            for end in range(1, scan_limit + 1):
                candidate = "".join(
                    decode(tail_ids[:end], skip_special_tokens=False).split()
                )
                if candidate == target:
                    matched_end = end
                    continue
                if matched_end is not None:
                    return matched_end, matched_length
                if len(candidate) > len(target):
                    break
            if matched_end is not None:
                return matched_end, matched_length
        return None

    def _prepare_generation(
        self,
        model,
        processor,
        inputs: dict[str, Any],
        session_id: str,
    ) -> _PreparedGeneration:
        full_input_ids = inputs.get("input_ids")
        if not isinstance(full_input_ids, torch.Tensor):
            raise TypeError("LOD session caching requires tensor input_ids")
        if full_input_ids.ndim != 2 or int(full_input_ids.size(0)) != 1:
            raise ValueError("LOD session caching requires a single prompt row")
        full_length = int(full_input_ids.size(-1))
        now = time.monotonic()
        self._make_room(session_id, now)
        entry = self._sessions.get(session_id)
        logical_input_ids = full_input_ids
        if entry is not None:
            cached_length = int(entry.tokens.numel())
            prefix_matches = (
                0 < cached_length < full_length
                and entry.tokens.device == full_input_ids.device
                and torch.equal(
                    full_input_ids[0, :cached_length], entry.tokens
                )
            )
            if prefix_matches:
                cache = entry.cache
                generation_inputs = self._slice_prompt_inputs(
                    inputs, cached_length, full_length
                )
                self._hits += 1
                self._reused_tokens += cached_length
            else:
                reconciled = self._reconcile_chat_turn(
                    entry, inputs, full_input_ids, processor
                )
                if reconciled is not None:
                    generation_inputs, logical_input_ids = reconciled
                    cache = entry.cache
                    self._hits += 1
                    self._reused_tokens += cached_length
                    self._turn_reconciliations += 1
                else:
                    self._remove_session(session_id)
                    self._prefix_mismatches += 1
                    entry = None
        if entry is None:
            cache = new_hf_lod_cache(model)
            generation_inputs = dict(inputs)
            self._misses += 1
        generation_inputs["past_key_values"] = cache
        return _PreparedGeneration(
            inputs=generation_inputs,
            cache=cache,
            full_input_ids=full_input_ids,
            logical_input_ids=logical_input_ids,
            generation_input_length=int(generation_inputs["input_ids"].size(-1)),
        )

    @staticmethod
    def _sequences(result) -> torch.Tensor:
        sequences = getattr(result, "sequences", result)
        if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
            raise TypeError("model.generate returned invalid sequences")
        return sequences

    def _publish_session(
        self,
        session_id: str,
        prepared: _PreparedGeneration,
        generated_ids: torch.Tensor,
    ) -> None:
        logical_tokens = torch.cat(
            (prepared.logical_input_ids[0], generated_ids), dim=0
        )
        cached_length = int(prepared.cache.get_seq_length())
        if not 0 < cached_length <= int(logical_tokens.numel()):
            self._drop_session(session_id)
            raise AssertionError(
                "LOD session cache length exceeds the generated token stream"
            )
        self._sessions[session_id] = _SessionEntry(
            cache=prepared.cache,
            tokens=logical_tokens[:cached_length].detach().clone(),
            pending_tokens=logical_tokens[cached_length:].detach().clone(),
            request_tokens=prepared.full_input_ids[0].detach().clone(),
            generated_tokens=generated_ids.detach().clone(),
            last_used=time.monotonic(),
        )
        self._sessions.move_to_end(session_id)
        if session_id in self._automatic_sessions:
            self._index_automatic_session(session_id, self._sessions[session_id])

    @staticmethod
    def _generation_kwargs(model, processor, inputs, gen_config) -> dict[str, Any]:
        kwargs = {
            **inputs,
            "generation_config": gen_config,
            "tokenizer": processor,
        }
        if hasattr(model, "has_talker"):
            kwargs["generation_mode"] = "text"
        return kwargs

    async def generate_non_streaming(
        self,
        model,
        processor,
        inputs: dict,
        gen_config,
        request_id: str,
    ) -> tuple[str, int, torch.Tensor]:
        session_id = current_lod_session()
        if not self.enabled or (
            session_id is None and not self.config.automatic_discovery
        ):
            return await super().generate_non_streaming(
                model, processor, inputs, gen_config, request_id
            )
        input_length = int(inputs["input_ids"].size(-1))

        def run() -> torch.Tensor:
            resolved_session_id = session_id
            if resolved_session_id is None:
                resolved_session_id = self._resolve_automatic_session(
                    inputs, processor
                )
            prepared = self._prepare_generation(
                model, processor, inputs, resolved_session_id
            )
            try:
                result = model.generate(
                    **self._generation_kwargs(
                        model, processor, prepared.inputs, gen_config
                    )
                )
                sequences = self._sequences(result)
                generated_ids = sequences[
                    0, prepared.generation_input_length :
                ]
                self._publish_session(
                    resolved_session_id, prepared, generated_ids
                )
                return generated_ids
            except Exception:
                self._drop_session(resolved_session_id)
                raise

        generated_ids = await self.async_submit(run)
        text = processor.decode(generated_ids, skip_special_tokens=True)
        return text, input_length, generated_ids

    def generate_streaming(
        self,
        model,
        processor,
        inputs: dict,
        gen_config,
        request_id: str,
        response_parser=None,
    ) -> tuple[asyncio.Queue, DirectStreamer]:
        session_id = current_lod_session()
        if not self.enabled or (
            session_id is None and not self.config.automatic_discovery
        ):
            return super().generate_streaming(
                model,
                processor,
                inputs,
                gen_config,
                request_id,
                response_parser=response_parser,
            )
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        rust_tokenizer = getattr(processor, "tokenizer", processor)._tokenizer
        streamer = _DeferredEndStreamer(
            rust_tokenizer,
            loop,
            queue,
            response_parser=response_parser,
        )

        def run() -> None:
            resolved_session_id = session_id
            if resolved_session_id is None:
                resolved_session_id = self._resolve_automatic_session(
                    inputs, processor
                )
            prepared = self._prepare_generation(
                model, processor, inputs, resolved_session_id
            )
            generation_inputs = dict(prepared.inputs)
            generation_inputs["streamer"] = streamer
            try:
                result = model.generate(
                    **self._generation_kwargs(
                        model, processor, generation_inputs, gen_config
                    )
                )
                sequences = self._sequences(result)
                generated_ids = sequences[
                    0, prepared.generation_input_length :
                ]
                self._publish_session(
                    resolved_session_id, prepared, generated_ids
                )
                streamer.finish()
            except _GenerationCancelled:
                self._drop_session(resolved_session_id)
                streamer.finish()
            except Exception as error:
                self._drop_session(resolved_session_id)
                loop.call_soon_threadsafe(
                    queue.put_nowait, _StreamError(str(error))
                )

        self.submit(run)
        return queue, streamer


class LODSessionGenerationState(GenerationState):
    """HF generation state whose sequential managers retain LOD sessions."""

    def __init__(self, config: LODSessionCacheConfig) -> None:
        super().__init__(continuous_batching=False)
        self.session_config = config

    def get_manager(self, model_id: str, use_cb: bool = False):
        if use_cb:
            raise RuntimeError("LOD session caching cannot use continuous batching")
        if model_id not in self._generate_managers:
            self._generate_managers[model_id] = LODSessionGenerateManager(
                self.session_config
            )
        return self._generate_managers[model_id]

    def session_cache_stats(self) -> dict[str, dict[str, int]]:
        return {
            model_id: manager.cache_stats()
            for model_id, manager in self._generate_managers.items()
            if isinstance(manager, LODSessionGenerateManager)
        }

    def shutdown(self) -> None:
        for manager in self._generate_managers.values():
            if isinstance(manager, LODSessionGenerateManager):
                manager.clear_sessions()
        super().shutdown()


__all__ = [
    "LOD_SESSION_HEADER",
    "LODSessionCacheConfig",
    "LODSessionGenerateManager",
    "LODSessionGenerationState",
    "bind_lod_session",
    "current_lod_session",
    "reset_lod_session",
]
