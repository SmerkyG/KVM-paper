# Attention-core timing

The preferred validated estimate of attention-region time is a paired
real-minus-dummy benchmark. It preserves normal uninstrumented vLLM execution
and CUDA graphs; do not use framework profiler event totals as the primary
attention-speed number. End-to-end prefill and decode latency remain the
primary performance measurements.

The dummy backend performs no cache update, routing, attention, or LoD state
maintenance. It copies the query to the already allocated attention-output
buffer so the rest of the model receives finite, shape-identical activations.
QKV projections, RoPE, output projection, MLPs, scheduling, and sampling still
run. Consequently,

```text
attention-core time = real wall time - matched dummy wall time
```

includes cache maintenance and all work inside the attention backend. Retaining
the output copy makes the estimate slightly conservative. Because GPU work can
overlap, this subtraction measures the attention region's marginal contribution
to full-model wall time, not a sum of isolated kernel durations.

Do not use this subtraction for a dynamic-MoE model. Replacing attention changes
later hidden states and therefore expert routing, so the real-minus-dummy delta
would mix attention with a different MoE workload. The validator rejects that
case; use matched end-to-end timing for K3 and use kernel profiles only as
diagnostics.

For a cohort, prefill is measured from the earliest request scheduling time to
the latest first token. Decode begins at that latest first token and ends at the
latest final token. Sequential chunked prefill and admission overhead therefore
remain in prefill, while an early row waiting for the remaining prefills is not
misclassified as decode.

## Reproduction requirements

Use the same otherwise idle node and GPU set, checkpoint, tensor and decode
context parallelism, batch, scheduler, scheduler budget, context-length panel,
decode length, seed, prompt cohort, cache reservation, and runtime package
versions for every arm. The explicit cache reservation must be large enough to
keep the entire requested cohort live; a nominal B=8 run that executes in waves
is invalid. Use natural ProLong continuation tokens with
`--fixed-decode-trace`, and use `--synchronized-decode` whenever batch size is
greater than one. Run one unreported warmup and at least three measured
repetitions while keeping CUDA graphs enabled.

For the canonical batch-8 panel, admission synchronization does **not** enlarge
the scheduler's prefill budget: it remains one 16K chunk plus the eight decode
rows. The explicit cache reservation must nevertheless retain all eight
completed prefills until the last prompt finishes; request preemption or
wave-wise decode invalidates the run.

Protocol-schema-10 validation constructs one maximum-length real-token request
panel and uses nested prefixes of those same requests at every context length.
It recomputes medians from the raw repetitions,
checks the per-request metric window against an independent wall clock, verifies
prompt and continuation hashes, checks that batched requests overlap and finish
together, records every request's scheduler-preemption count and rejects any
preemption or prefix-cache hit, audits the worker-side real/dummy/LoD attention
mode and hardware,
records and validates the actual per-worker route counts, prefill/update
schedule, leaf geometry, and cross-layer construction group,
fingerprints the installed AITER route-kernel source on every worker,
fingerprints every loaded Kimi LoD shared object and its exact JIT build
manifest, verifies the fused prefill route flags whenever a prompt exceeds the
16K exact prefix,
reconstructs every cohort total from its individual execution-batch timing
records, requires every measured continuation to reproduce the warmup trace,
rejects a greater than 10% within-run timing span, and rejects mismatched
configurations or runtime packages. Any ``LOD_KIMI_*`` override makes a run
experimental and is rejected unless explicitly allowed. Reported
attention-region speedups include
conservative observed bounds formed from the extrema of the measured arms. It
records each arm's source
fingerprint but does not require fingerprints to match, so an unchanged dense
or dummy control can be reused after a LoD-only source change. The caller
remains responsible for confirming that the reused arm's executed path is
unchanged. Ordinary end-to-end dense-versus-LoD comparisons are stricter: both
arms must have the exact same complete source identity. The resolved Python
executable and the effective Python, ROCm, compiler, loader, and tuning-table
paths are part of that matching contract. Legacy artifacts cannot be used for
validated subtraction or end-to-end comparison.

Run the ordinary full and LoD arms as described in [PROLONG.md](PROLONG.md).
For example:

```bash
uv run python -m benchmarks.prolong \
  --checkpoint Qwen/Qwen3.8-27B-FP8 \
  --mode full \
  --measure speed \
  --lengths 131072 \
  --batch-size 8 \
  --speed-samples 8 \
  --tensor-parallel-size 1 \
  --decode-tokens 1025 \
  --fixed-decode-trace \
  --synchronized-decode \
  --repeats 3 \
  --seed 0 \
  --output results/qwen-full.json

uv run python -m benchmarks.prolong \
  --checkpoint Qwen/Qwen3.8-27B-FP8 \
  --mode two-tier \
  --measure speed \
  --lengths 131072 \
  --batch-size 8 \
  --speed-samples 8 \
  --tensor-parallel-size 1 \
  --decode-tokens 1025 \
  --fixed-decode-trace \
  --synchronized-decode \
  --repeats 3 \
  --seed 0 \
  --output results/qwen-two-tier.json
```

Run the matched dummy control by changing only the attention mode and adding
`--dummy-attention`:

```bash
uv run python -m benchmarks.prolong \
  --checkpoint Qwen/Qwen3.8-27B-FP8 \
  --mode full \
  --dummy-attention \
  --measure speed \
  --lengths 131072 \
  --batch-size 8 \
  --speed-samples 8 \
  --tensor-parallel-size 1 \
  --decode-tokens 1025 \
  --fixed-decode-trace \
  --synchronized-decode \
  --repeats 3 \
  --seed 0 \
  --output results/qwen-dummy.json
```

Validate the pairing and calculate the differences:

```bash
uv run python -m benchmarks.attention_timing \
  --dummy results/qwen-dummy.json \
  --real results/qwen-full.json \
  --real results/qwen-two-tier.json \
  --output results/qwen-attention-time.json
```

Attention speedup is the full-attention difference divided by the LoD
difference. End-to-end speedup must still be reported separately from the
ordinary, unsubtracted wall times.

This benchmark is performance-only. Dummy output tokens have no quality
meaning. Always use normal full and LoD runs for loss and task evaluation.

## Kernel diagnostics

Kernel microbenchmarks are diagnostic only; they must not replace the matched
end-to-end protocol above.  In particular, a wrapper that launches work on a
private CUDA/HIP stream must return its completion stream or event.  The
measuring stream must explicitly wait for that dependency *before* recording
its end event.  A device-wide synchronization after both events have already
been recorded does not repair this mistake: it can make an unfinished private
stream look artificially free.  Every probe must also synchronize before its
first measured interval, warm every compiled specialization, use production
tensor layouts and real dimensions, and report medians from multiple samples.

Per-kernel event sums are not additive when production overlaps local,
coarse, fine, communication, or cross-layer cache-construction streams.  Use
them to identify a critical path, then accept an optimization only when the
unchanged end-to-end protocol confirms it.  Profiling and debug-print modes are
never valid performance configurations.
