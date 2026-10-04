"""vLLM scheduler policy that preserves a full LoD prefill budget."""

from __future__ import annotations

import os

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
        # Benchmark-only cohort barrier.  With a 16K token budget, equal long
        # prompts otherwise enter decode one at a time as their prefills
        # finish.  Request-level timestamps then measure different live batch
        # sizes at different context lengths.  Holding existing decode rows
        # while any prompt remains unfinished makes the first decode step a
        # genuine full-cohort B=N step without changing prefill chunking.
        if os.getenv("LOD_BENCHMARK_SYNCHRONIZED_DECODE", "0") == "1" and (
            self.waiting
            or self.skipped_waiting
            or any(request.is_prefill_chunk for request in self.running)
        ):
            # ``Scheduler.schedule`` increments ``current_step`` before it
            # tests eligibility.  A +1 assignment here would therefore make
            # the request eligible in the very call that was meant to hold
            # it. Keep it one step beyond that increment and refresh the
            # bound on every turn until every prefill is complete.
            next_step = self.current_step + 2
            for request in self.running:
                if not request.is_prefill_chunk:
                    request.next_decode_eligible_step = max(
                        request.next_decode_eligible_step,
                        next_step,
                    )
        # A long prompt remains in ``running`` after its first scheduler turn.
        # Keep applying the aligned budget while any such row is unfinished;
        # otherwise a B=1 run falls back to (16K + one reserved decode token)
        # and produces 16,385-token continuation chunks.  Besides creating a
        # tiny trailing block, that defeats the cross-layer 16K cache builder.
        has_running_prefill = any(
            request.is_prefill_chunk for request in self.running
        )
        if not (self.waiting or self.skipped_waiting or has_running_prefill):
            return super().schedule(throttle_prefills)

        admission_cohort = int(
            os.getenv("LOD_BENCHMARK_ADMISSION_COHORT", "1")
        )
        if admission_cohort < 1:
            raise ValueError("LOD_BENCHMARK_ADMISSION_COHORT must be positive")
        prefill_cohort = int(os.getenv("LOD_BENCHMARK_PREFILL_COHORT", "1"))
        if prefill_cohort < 1:
            raise ValueError("LOD_BENCHMARK_PREFILL_COHORT must be positive")
        configured_budget = self.max_num_scheduled_tokens
        # Offline ``LLM.generate`` submits requests to the asynchronous engine
        # one message at a time.  Without a short admission barrier the core
        # can consume the first long prompt before the other B-1 requests even
        # reach ``waiting``.  This benchmark-only switch holds the initial
        # scheduler turn until the requested cohort is visible. Admission and
        # prefill parallelism are deliberately independent: the release B=8
        # benchmark still has one 16K total scheduler budget, not B * 16K.
        if admission_cohort > 1:
            eligible_requests = [*self.running, *self.waiting, *self.skipped_waiting]
            if (
                len(eligible_requests) < admission_cohort
                and eligible_requests
                and all(request.num_computed_tokens == 0 for request in eligible_requests)
            ):
                self.max_num_scheduled_tokens = 0
                try:
                    return super().schedule(throttle_prefills)
                finally:
                    self.max_num_scheduled_tokens = configured_budget
        self.max_num_scheduled_tokens = min(
            configured_budget,
            PREFILL_CHUNK_SIZE * prefill_cohort
            + self._running_decode_tokens(),
        )
        try:
            return super().schedule(throttle_prefills)
        finally:
            self.max_num_scheduled_tokens = configured_budget


__all__ = ["LODChunkAlignedScheduler"]
