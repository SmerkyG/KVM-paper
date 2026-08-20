# Final two-tier BF16 and INT8 LongBench v2 comparison

Run date: 2026-08-20

This evaluates the latest uncapped two-tier LOD vLLM paths on all 503
LongBench v2 examples. Two complete repeats were run for each precision after
the final cooperative decode changes because answer-choice decisions show
material run-to-run numerical sensitivity.

## Main result

| Mode | Run 1 | Run 2 | Two-run mean |
|---|---:|---:|---:|
| Historical full attention | 250/503 (49.70%) | - | 49.70% |
| Historical recursive INT4 LOD | 236/503 (46.92%) | - | 46.92% |
| Latest two-tier BF16 | 230/503 (45.73%) | 240/503 (47.71%) | 235/503 (46.72%) |
| Latest two-tier INT8 | 240/503 (47.71%) | 236/503 (46.92%) | 238/503 (47.32%) |

INT8 does not show an accuracy penalty relative to BF16 in these runs. It is
10 answers better in repeat 1 and four answers worse in repeat 2. The paired
exact p-values are 0.110 and 0.541, respectively. The two-run mean difference
of three answers (0.60 percentage points) is much smaller than run-to-run
variation and should not be interpreted as an INT8 quality gain.

BF16 repeated at 230 and 240 correct with 460/503 identical predictions;
INT8 repeated at 240 and 236 with 458/503 identical predictions. This confirms
that one LongBench pass is too noisy to resolve a few-answer precision effect
for this batched execution path. The historical two-tier BF16 result was
243/503, which is within the broader observed variation but above both latest
repeats.

## End-to-end speed

Times exclude model loading, compilation, one long warmup batch, and the
evaluator's short warmup. Latest BF16 and INT8 values are means of the two
complete runs.

| Metric | Full attention | Two-tier BF16 | Two-tier INT8 |
|---|---:|---:|---:|
| Slowest-shard / eight-GPU wall | 1,355.02 s | 527.95 s (2.567x) | 460.29 s (2.944x) |
| Aggregate shard wall | 9,080.45 s | 3,583.18 s (2.534x) | 3,135.21 s (2.896x) |
| Summed request latency | 51,369.47 s | 21,314.59 s (2.410x) | 18,843.92 s (2.726x) |

Relative to BF16, INT8 reduces slowest-shard wall time by 12.82%, aggregate
shard time by 12.50%, and summed request latency by 11.59%.

By LongBench length class, the summed-request speedups versus full attention
are:

| Length | Two-tier BF16 | Two-tier INT8 | INT8 vs BF16 |
|---|---:|---:|---:|
| Short | 1.318x | 1.283x | 2.70% slower |
| Medium | 2.066x | 2.341x | 11.75% faster |
| Long | 2.928x | 3.407x | 14.05% lower latency |

The short subset does not amortize INT8 quantization, while medium and long
contexts do. LongBench as a whole benefits because most runtime is in the
medium and long requests.

## Configuration

- Model: `Qwen/Qwen3.5-35B-A3B`, BF16 weights
- vLLM 0.27.1 on gfx942, eight one-GPU shards per precision
- Eight resident requests, `max_num_batched_tokens=16384`
- Native 262,144-token context; prompts capped at 262,016 tokens
- Direct 4,096-token-chunk two-tier prefill, raw top-eight routing,
  `16 sqrt(T)` state schedule, and uncapped physical leaf pages
- Latest GQA-cooperative HIP decode with adaptive route splits
- BF16 mode stores BF16 K/V leaves
- INT8 mode stores signed INT8 K/V leaves with per-token scales and uses the
  INT8 leaf/coarse attention paths
- Thinking disabled, guided A-D decoding, at most 32 output tokens

All four runs completed 503 unique examples with exit code zero and no memory
faults or backend failures. Raw run-1 outputs and logs are in `bf16/` and
`int8/`; repeat-2 outputs are in the sibling
`20260820_two_level_final_bf16_int8_budget16k_repeat2/` directory. Detailed
machine-readable headline results are in `summary.json`.
