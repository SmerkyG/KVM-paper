"""Inspect actual prefill row chunks during warmup, never during timing."""

from collections import Counter
import json
import os


def prefill_lengths(batch) -> list[int]:
    # Read the assembled batch's live CPU query offsets. These are shared
    # by native MLA and LoD, unlike their backend metadata.
    # Single-token decode rows are not part of this chunked-prefill audit.
    rows = int(batch.num_reqs)
    starts = batch.query_start_loc_np[: rows + 1]
    return [int(b - a) for a, b in zip(starts, starts[1:]) if b - a > 1]


def arm_prefill_batch_audit(worker):
    runner = worker.model_runner
    if hasattr(runner, "_prefill_batch_audit_original"):
        raise RuntimeError("prefill audit already installed")
    # This pinned vLLM image uses GPU model-runner v2 (prepare_inputs returns
    # InputBatch). Do not hook model.forward, which a graph wrapper bypasses,
    # or legacy GPUModelRunner._model_forward, which v2 does not have.
    original = runner.prepare_inputs
    runner._prefill_batch_audit_original = original
    runner._prefill_batch_audit_chunks = []
    runner._prefill_batch_audit_memory = []
    audit_memory = os.environ.get("LOD_BENCHMARK_PREFILL_MEMORY_AUDIT") == "1"

    def audited(*args, **kwargs):
        batch = original(*args, **kwargs)
        lengths = prefill_lengths(batch)
        if lengths:
            runner._prefill_batch_audit_chunks.append(lengths)
            if audit_memory:
                from benchmarks._kimi_prefill_memory import snapshot_prefill_memory
                snapshot = snapshot_prefill_memory(worker)
                runner._prefill_batch_audit_memory.append(snapshot)
                if worker.rank == 0:
                    # Preserve the last completed chunk's accounting even if
                    # the next warmup chunk kills a worker before the final RPC.
                    print("KIMI_PREFILL_MEMORY_CHUNK " + json.dumps({
                        "rank": worker.rank,
                        "chunk_index": len(runner._prefill_batch_audit_chunks) - 1,
                        "input_lengths": lengths, **snapshot,
                    }), flush=True)
        return batch

    runner.prepare_inputs = audited
    return {"rank": worker.rank, "scope": "untimed warmup only"}


def finish_prefill_batch_audit(worker):
    runner = worker.model_runner
    runner.prepare_inputs = runner._prefill_batch_audit_original
    chunks = runner._prefill_batch_audit_chunks
    memory = runner._prefill_batch_audit_memory
    del runner._prefill_batch_audit_original
    del runner._prefill_batch_audit_chunks
    del runner._prefill_batch_audit_memory
    return {"rank": worker.rank, "chunks": chunks,
        "prefill_tokens": sum(map(sum, chunks)),
        "row_count_histogram": dict(Counter(map(len, chunks))),
        "scope": "untimed warmup only; no audit hooks in measured generation",
        **({"memory_snapshots": memory} if memory else {})}


def validate_prefill_batch_audits(audits, *, length, batch_size, row_chunk,
                                  cohort, world_size):
    if {a["rank"] for a in audits} != set(range(world_size)):
        raise RuntimeError("missing prefill batch audit rank")
    for audit in audits:
        chunks = audit["chunks"]
        if not chunks or audit["prefill_tokens"] != length * batch_size:
            raise RuntimeError(f"wrong actual prefill token count: {audit}")
        if max(map(len, chunks)) != cohort or any(
            len(chunk) > cohort or any(n <= 0 or n > row_chunk for n in chunk)
            for chunk in chunks
        ):
            raise RuntimeError(f"unexpected actual prefill batch shape: {audit}")
    if any(a["chunks"] != audits[0]["chunks"] for a in audits):
        raise RuntimeError("TP ranks executed different prefill row chunks")
