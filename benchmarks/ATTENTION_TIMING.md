# Attention-core timing

The preferred validated measurement of attention time is a paired
real-minus-dummy benchmark. It preserves normal uninstrumented vLLM execution
and CUDA graphs; do not use framework profiler event totals as the primary
attention-speed number.

The dummy backend performs no cache update, routing, attention, or LoD state
maintenance. It copies the query to the already allocated attention-output
buffer so the rest of the model receives finite, shape-identical activations.
QKV projections, RoPE, output projection, MLPs, scheduling, and sampling still
run. Consequently,

```text
attention-core time = real wall time - matched dummy wall time
```

includes cache maintenance and all work inside the attention backend. Retaining
the output copy makes the estimate slightly conservative.

## Reproduction requirements

Use the same unloaded GPU type, checkpoint, tensor parallelism, batch,
scheduler, context lengths, decode length, seed, prompt cohort, and runtime
package versions for every run. Run one unreported warmup before measured
repetitions and keep CUDA graphs enabled. The summary command rejects mismatched
recorded configuration, context lengths, prompt-token hashes, or runtime package
versions. It records each arm's source fingerprint but does not require the
fingerprints to match, so an unchanged full-attention or dummy control can be
reused after a LoD-only source change. The caller remains responsible for
confirming that a reused control's executed path is unchanged. A run still
aborts rather than writing an artifact if its own source identity changes while
it is executing. Legacy artifacts without `benchmark_identity.runtime` cannot
be used for validated subtraction.

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
  --repeats 1 \
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
  --repeats 1 \
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
  --repeats 1 \
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
