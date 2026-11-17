# Triton KVM kernels

## Implementation map

KVM maintains a chunked causal sliding window plus a recurrent state
of compressed key/value rows. Overflow tokens are processed with global novelty
ranking, append-before-merge routing, protected sink rows, and a joint softmax
over the recurrent state and recent window.

The entry points are:

| Surface | Entry point |
|---|---|
| Eager semantic reference | `model.kvm_mixer.SequenceMixer` |
| Optimized Triton mixer | `model.kvm_triton_mixer.SequenceMixer` |
| Triton training kernels | `model.kernels.kvm_triton_training_kernels` |
| Optimized decode cache state | `model.kvm_triton_decode.TritonKVMDecodeCache` |
| Triton decode attention/ring kernels | `model.kernels.kvm_triton_decode` |
| FP32 merge-routing reference | `model.kvm_fp32_routing_reference.SequenceMixer` |

## Selecting the optimized training path

The paper KVM configs select the eager reference unless the optimized class is
explicitly layered or supplied on the command line. A fixed256 example is:

```bash
bash run.sh \
  -c configs/prolong/base.yaml \
  -c configs/prolong/tokens_3B.yaml \
  -c configs/prolong/120M/kvm_wd.yaml \
  -c configs/prolong/kvm_triton.yaml
```

Use `configs/prolong/120M/kvm_sqrt16_wd.yaml` for the sqrt16 schedule. The
default validation corpus is `SmerkyG/dclm-10B`.

## Reproducing the kernel benchmarks

The benchmark compares the KVM Triton kernels with PyTorch Flash SDPA
at the post-QKV operator boundary. Both implementations receive BF16
preprojected tensors and return `[B,T,H*D]`; projections, RoPE, the output
projection, and the rest of the transformer layer are excluded.

The paper measurements use an AMD Instinct MI325X with PyTorch 2.9.1 and ROCm
7.2. Non-smoke workers reject other environments. Full attention is restricted
to `SDPBackend.FLASH_ATTENTION`. Fixed-context workers profile both arms before
measurement; decode-series workers profile the selected arm. A result is
rejected unless the expected KVM Triton or ROCm AOTriton attention symbols
appear in the trace. The detected symbols and software versions are stored with
the raw measurements.

### Fixed-context benchmark

This benchmark uses batch 8, 32 heads, head dimension 128, and context lengths
512, 1024, 2048, 4096, 8192, 16384, and 32768. It measures prefill, backward,
and 256-token
autoregressive-decode segments for fixed-256 and `16 sqrt(T)` KVM state
schedules against full causal attention.

Inputs are deterministically repopulated between measurements. Before each
timed operator call, the harness streams through a 512 MiB buffer and stages
the inputs outside the timing interval. Kernel warmup and autotuning also occur
before measurement. Decode uses a separate event pair for each generated token;
cache updates and output layout conversion remain inside the timed interval.

By default, each coordinate uses seeds 0 through 9, five warmup calls, and four
reported samples per seed. A pilot call selects how many iterations are
averaged into each sample, targeting 250 ms and capping the count at 50. No
samples are discarded. The plots use the median of the four samples for each
seed.

### Decode trajectory

The decode-series benchmark measures every generated-token position from 2
through 32768 for fixed-256 KVM, `16 sqrt(T)` KVM, and full attention. Each run
first traverses the complete trajectory without timing so all kernels are
compiled and tuned. It then resets the cache, repopulates the inputs, evicts
hardware cache before each position outside the event interval, and records one
timing per position. The default is ten runs per implementation.

### Commands and outputs

Run the workers and analyzers on the reference GPU host:

```bash
.venv/bin/python scripts/benchmark_kvm.py submit \
  --max-parallel 8 \
  --root experiments/kvm_benchmark_b8h32

.venv/bin/python scripts/benchmark_kvm.py analyze \
  --root experiments/kvm_benchmark_b8h32

.venv/bin/python scripts/benchmark_kvm.py decode-series-submit \
  --max-parallel 8 \
  --root experiments/kvm_decode_series

.venv/bin/python scripts/benchmark_kvm.py decode-series-analyze \
  --root experiments/kvm_decode_series
```

Alternatively, `scripts/run_kvm_end_to_end.sh [output-root] [gpu-count]` runs
all four commands.

The fixed-context output contains one JSON file per coordinate under `raw/`,
`runs.csv`, `summary.csv`, and four figures under `plots/`. The decode-series
output contains one JSON file per run under `raw/`,
`decode_series_runs.csv`, `decode_series_mean.csv`,
`decode_series_speedup.csv`, and its figure under `plots/`.

### Full-model inference

```bash
cluster-run --num-gpus 1 -- uv run --no-sync python scripts/benchmark_kvm.py inference \
  --checkpoint /path/to/gpta2-checkpoint --contexts 1024 8192 32768 \
  --batch-size 1 --new-tokens 32 --cudagraph --output experiments/gpta2-inference.json
```

The benchmark starts with device-resident token IDs and times the entire
greedy generation path, including embedding, token shift, attention, FFN,
cache updates, and the output head. It reports prefill through the first
generated token, the remaining decode steps, and synchronized wall time.
The JSON includes every timed trial, capture time, and peak allocator memory.

Use the same command with a Triton KVM checkpoint to benchmark KVM. Both models
report ordinary `eager` timings. GPTA-2 `static_eager` uses a preallocated KV
cache. Triton KVM `prepared_eager` retains the prefill schedule tensors on the
device between requests and uses the existing preallocated decode ring.

`--cudagraph` captures prefill and the full decode trajectory with those caches
(`static_graph` for GPTA-2 and `prepared_graph` for Triton KVM). KVM replay
includes any recurrent state updates during prefill and decode. The benchmark
captures at most 32 decode steps per graph and replays the graphs in order
with the same growing cache. Replay uses a fixed batch size, prompt length,
and number of generated tokens. Graph setup
and warmup are outside the timed trials. The benchmark checks greedy tokens and
logits on two prompts before timing replay.

Add `--prefill-chunk-size 2048` to process the prompt in model-wide chunks.
GPTA-2 carries the full-attention KV cache and uses a lower-right causal mask.
Triton KVM carries its recurrent state and recent-token window, and its
model-wide chunk size must be a multiple of the recurrent chunk length.
Both carry token-shift state and position IDs across chunks and use the same
decode cache layouts as one-shot prefill.
