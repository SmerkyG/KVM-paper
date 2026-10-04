"""Benchmark-only request rotation, without changing token/update budgets."""

from __future__ import annotations

from typing import Any


def rotate_prefill_requests(
    running: list[Any], *, has_waiting: bool, max_running: int, next_step: int,
    max_prefills: int | None = None,
) -> None:
    """Admit waiting rows first, then advance the least-progressed prefill.

    vLLM checks ``next_decode_eligible_step`` for all running requests, despite
    the field's name. Using it leaves request/cache ownership untouched and
    does not preempt a row or free its KV blocks. Decode rows are never gated.
    ``max_prefills`` bounds simultaneous unfinished caches, independently of
    the maximum decode batch. This experiment assumes space for that cohort.
    """
    prefills = [row for row in running if row.is_prefill_chunk]
    eligible = [row for row in prefills if row.next_decode_eligible_step <= next_step]
    if max_prefills is not None and max_prefills < 1:
        raise ValueError("max_prefills must be positive")
    admit = (
        has_waiting and len(running) < max_running
        and (max_prefills is None or len(prefills) < max_prefills)
    )
    selected = (
        min(eligible, key=lambda row: row.num_computed_tokens)
        if eligible and not admit else None
    )
    for row in prefills:
        if row is not selected:
            row.next_decode_eligible_step = max(row.next_decode_eligible_step, next_step + 1)
