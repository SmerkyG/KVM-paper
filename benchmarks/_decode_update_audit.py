"""Read existing LoD update counters outside benchmark timing windows.

No CUDA calls, events, profiling, or per-token instrumentation are used.
The serving pools already count completed catch-up batches and request rows.
"""

from __future__ import annotations

from typing import Any


def read_decode_update_counters(worker: Any) -> dict[str, dict[str, int]]:
    runner = getattr(worker, "model_runner", None)
    state = getattr(runner, "model_state", None)
    runtime = getattr(state, "_vllm_lod_runtime", None)
    pools = getattr(runtime, "pools", {}) if runtime is not None else {}
    return {
        str(name): {
            "catch_up_batches": int(pool.catch_up_batches),
            "catch_up_rows": int(pool.catch_up_rows),
        }
        for name, pool in pools.items()
    }


def decode_update_deltas(
    before: list[dict[str, dict[str, int]]] | None,
    after: list[dict[str, dict[str, int]]] | None,
) -> list[dict[str, dict[str, int]]] | None:
    """Subtract counters per worker and layer, never pooling TP/DCP ranks."""
    if before is None or after is None:
        return None
    if len(before) != len(after):
        raise RuntimeError("worker count changed during the speed measurement")
    result = []
    for initial, final in zip(before, after, strict=True):
        if initial.keys() != final.keys():
            raise RuntimeError("LoD layer set changed during the speed measurement")
        worker_delta = {}
        for name, counters in initial.items():
            delta = {
                key: final[name][key] - value for key, value in counters.items()
            }
            if any(value < 0 for value in delta.values()):
                raise RuntimeError("LoD update counters reset during the measurement")
            worker_delta[name] = delta
        result.append(worker_delta)
    return result
