# Native three-tier top-2 decode

Optimized top-2 is now the default three-tier decode policy for **Qwen3.8 and
K2 Horizon**, in both **BF16 and INT4** modes. The shared recursive kernel
entrypoint selects two distinct semantic pages per routed centroid in one
summary scan, opens their leaves, and leaves one count-corrected disjoint
parent residual. Centroids with only one available page open that page once.

## Implementation

`DECODE_PAGE_COUNT = 2` in `lod_attention/_config.py` supplies the native
default for `query_major_residual_page_attention`. Its indexed wrapper is
used by both the fused vLLM decode path and the Hugging Face engine paths.
The optimized Triton kernel body retains the previously validated single-scan
winner/runner-up implementation.

The BF16 PyTorch reference also opens two pages and subtracts both pages
before constructing its single residual. Its existing chronological tie
order is preserved; the optimized kernel retains its established physical
page tie order.

Two-tier attention continues to refine every leaf in its selected centroids.
Prefill continues to refine all leaves in selected centroids, and short decode
retains its existing full-cache fallback. All modes retain eight centroid
routes. The low-level benchmark option still supports explicit budgets of
one, two, four, and eight pages. The benchmark hook now honors an explicit
one-page control even though the native default is two.

The production kernel source digest is:

```text
789c39bdb1ad901ca0ea706a2d29f1e8351fbd222529b06de8e282a35ad26868
```

Earlier quality and speed artifacts retain their frozen kernel digest
`19aa6136...`. Updating the native default and its import/documentation
changes the source digest without changing the optimized top-2 kernel body.

The initial native-default validation used `99f9c4d6...`. The subsequent
K2 batch-8 tuning changes only the AMD compiler
occupancy hint for native D128/GQA8, one-warp, 16-page-scan, two-page decode
with at least eight rows. BF16 leaves with BF16 summaries use hint 4; INT4
leaves with INT8 summaries use automatic hint 0. The Triton arithmetic and
reduction layout stay unchanged. Explicit nondefault hints are respected.
Qwen's D256 launch geometry retains its existing settings.

## Validation

All GPU validation used `cluster-run` and the existing ROCm runtime through
`uv`. All validation jobs below finished with exit code 0.

| Check | Run | Result |
| --- | ---: | --- |
| Full CPU suite, GPUs hidden | local `uv` | 73 passed, 50 GPU cases skipped |
| GPU frontier/hash references, two-warp page launches | 21826 | 48 passed |
| GPU frontier/hash references, production model launch geometry | 21829 | 48 passed |
| Native K2 three-tier BF16, 16K NIAH-S3 smoke | 21827 | 4/4 correct |
| Native K2 three-tier INT4, 16K NIAH-S3 smoke | 21828 | 4/4 correct |
| Final K2 occupancy candidate, GPU frontier/hash references | 21867 | 48 passed |
| K2 BF16 batch-8, same-GPU previous/tuned top-2 | 21869 | 1.76% lower decode latency |
| K2 INT4 batch-8, previous/tuned top-2 | 21859 | 2.08% lower decode latency |

The final GPU references include calls that omit the page-budget argument,
so they exercise the production default. They cover D128/GQA8 K2 with one
warp and D256/GQA6 Qwen with two warps, BF16 and INT4 leaves, BF16 and INT8
summaries, partial pages, fewer pages than the budget, invalid routes, tied
winners, runner-up selection across scan blocks, and bounded overflow-hash
misses.

The PyTorch tests check one-page and two-page centroids as well as a third
page retained in the residual. Existing two-tier tests continue to pass.
Benchmark-hook tests check explicit budgets and preserve prefill behavior.

The K2 smoke runs use IFM/K2-Horizon-32B-FP8 weights, two concurrent requests,
four UUID needles per precision, and 128 output tokens. They exercise native
vLLM serving without the page-count ablation hook. Their prompt generator is
the runtime's lm-eval 0.4.12. These are serving smoke tests; a full K2 RULER
quality panel is being collected separately; its results will be reported
after the BF16 and INT4 panels complete.

## Performance

On MI325X with IFM/K2-Horizon-32B-FP8 weights, batch size 8, 64K ProLong
prompts, and 1,025 forced decode steps, the occupancy tuning reduced median
three-tier top-2 decode latency across three measured repetitions:

| Leaf precision | Previous top-2 | Tuned top-2 | Reduction |
| --- | ---: | ---: | ---: |
| BF16 | 52.386 ms/batch step | 51.462 ms/batch step | 1.76% |
| INT4 | 53.413 ms/batch step | 52.304 ms/batch step | 2.08% |

Each comparison used the same GPU, one warm-up repetition, identical cohorts,
and disabled prefix caching. The INT4 result is within 0.14% of the matched
top-1 control (52.375 ms/batch step). BF16's final tuning comparison did not
include a new top-1 control. The GPU references verify bitwise attention
outputs and log-sum-exp values for identical cached inputs across the tuned
and untuned batch geometries.

## Reproduction

Experiment logs and quality panels are recorded separately from this serving
change. Use `uv` and the self-documenting `cluster-run` command for GPU runs.

Run the GPU references from the release worktree:

```bash
cluster-run --detach --name production-top2-tests-rerun --num-gpus 1 -- \
  env LOD_RUN_GPU_TESTS=1 PYTHONPATH="$PWD" \
  uv run --no-project \
  --python /home/dan/subusers/agent/.venvs/vllm-rocm-0.27.1/bin/python \
  python -m pytest -q tests/test_page_opening.py tests/test_paged_hash_safety.py
```

Run a fresh native K2 task with a new output path:

```bash
cluster-run --detach --name k2-top2-int4-smoke-rerun --num-gpus 1 -- \
  env PYTHONPATH="$PWD:$PWD/integrations/vllm_lod" \
  uv run --no-project \
  --python /home/dan/subusers/agent/.venvs/vllm-rocm-0.27.1/bin/python \
  python -m benchmarks.niah_s3 --checkpoint IFM/K2-Horizon-32B-FP8 \
  --mode three-tier-int4 --lengths 16384 --samples 4 --batch-size 2 \
  --max-new-tokens 128 --output results/production-top2/k2-int4-smoke-rerun.json
```
