"""vLLM scheduler policy that preserves a full LoD prefill budget."""

from __future__ import annotations

from lod_attention._config import PREFILL_CHUNK_SIZE
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput


class LODChunkAlignedScheduler(AsyncScheduler):
    """Keep decode rows from carving tokens out of the 16K prefill budget."""

    def _running_decode_tokens(self) -> int:
        next_step = self.current_step + 1
        total = 0
        for request in self.running:
            if (
                request.is_prefill_chunk
                or next_step < request.next_decode_eligible_step
            ):
                continue
            if (
                request.num_output_placeholders > 0
                and request.num_computed_tokens
                + 2
                - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                continue
            tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            tokens = min(
                tokens,
                self.max_model_len
                - request.num_computed_tokens
                - self.num_sampled_tokens_per_step,
            )
            total += max(int(tokens), 0)
        return total

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        if not (self.waiting or self.skipped_waiting):
            return super().schedule(throttle_prefills)

        configured_budget = self.max_num_scheduled_tokens
        self.max_num_scheduled_tokens = min(
            configured_budget,
            PREFILL_CHUNK_SIZE + self._running_decode_tokens(),
        )
        try:
            return super().schedule(throttle_prefills)
        finally:
            self.max_num_scheduled_tokens = configured_budget


__all__ = ["LODChunkAlignedScheduler"]
