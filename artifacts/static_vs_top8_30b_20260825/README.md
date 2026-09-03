# Static cohort versus unrestricted top-8 decode at 64K

## Scope and freshness

This panel answers which 30B-class models are faster at 64K with either:

- unrestricted query-dependent top-8 routing (retaining the ordinary
  1024-leaf overfull-centroid guard); or
- the query-independent static cohort, opening all leaves of centroids with
  posting-list length at most `max(16, ceil(sqrt(T) / 16))` and representing
  larger centroids by their coarse entry.

At 64K the static cap is 16. Every timing used below was produced after the
custom-cache slowdown fix. Older timing artifacts were excluded. The Muse
pair is the immediately preceding post-fix two-length run; its 64K arm uses
cap 16 (the artifact's final cap diagnostic is 23 because its 128K arm ran
last).

## Method

- context 65,536, batch 8;
- distinct real `Seerkfang/prolong-64k-512-new` documents, not repeated
  synthetic text;
- 16K maximum batched prefill, 64 generated tokens;
- median of three measured prefill/decode repetitions after warmup;
- BF16 two-level LOD storage and raw routing geometry, as recorded by the
  saved execution diagnostics;
- identical model, prompt, vLLM, cache, and prefill settings within each pair;
- execution audits required the requested fixed-mask top-8 or compact-static
  final-attention path to have actually executed.

Gemma's global attention has GQA8 and D=512, which the existing indexed final
kernel did not support. Its pair therefore uses the new common D=512 path:
materialized FP16 QK scores followed by dimension-split MFMA PV and a split
final reduction. Compact and masked variants were checked against dense
reference attention before the model run; maximum BF16 output error was
0.001953 and maximum LSE error was 0.000158.

## Decode results

Lower latency is better. The parenthesized values are speedups versus the
historical full-attention control; static delta is `(static / top-8 - 1)`.

| model | historical full attention | unrestricted top-8 | static cap 16 | static delta | latency winner | high-quality reference |
|---|---:|---:|---:|---:|---|---|
| Qwen3.8-27B-FP8 | 52.030 ms | **36.247 ms (1.44x)** | 39.893 ms (1.30x) | +10.06% | top-8 | **top-8** |
| Gemma-4-26B-A4B | 11.694 ms | **10.421 ms (1.12x)** | 10.930 ms (1.07x) | +4.88% | top-8 | **top-8** |
| Phi-4 TP5 | 9.970 ms | 10.913 ms (0.91x) | **9.231 ms (1.08x)** | -15.42% | static | **top-8** |
| Muse-Glimmer-30B | 19.215 ms | 19.349 ms (0.99x) | **18.696 ms (1.03x)** | -3.37% | static | **top-8** |
| OLMo-3-1125-32B | 30.481 ms | 28.878 ms (1.06x) | **26.864 ms (1.13x)** | -6.97% | static | **top-8** |

The Muse control is the newer post-cache-fix matched full-attention run. The
other full-attention values are the recorded historical controls; the custom
cache fix does not participate in the native full-attention path.

Thus, at 64K static wins the latency-only comparison on Phi, Muse, and OLMo;
unrestricted top-8 wins on Qwen and Gemma. Top-8 remains the high-quality
reference for all five models because the static arms do not have comparable
quality validation. The repeat dispersion is much smaller than each difference
(per-arm CV at most 0.70%, except Gemma top-8's one-time first-repeat JIT;
its two stable repeats agree closely).

This is not a universal static-to-top-8 crossover. It says that at 64K the
serial routing/mask work costs more than the static cohort's additional exact
attention on three models, but saves enough leaf work to pay for itself on
Qwen and Gemma. It does not establish static as a quality-equivalent policy.
A longer-context sweep is still needed to locate the actual latency crossover
for the latter two.

## Prefill observations

Top-8 and static are decode-selection policies; both arms use the same
two-level LOD prefill algorithm. The table nevertheless reports both measured
end-to-end prefill arms. Parentheses are throughput speedups versus historical
full attention, equivalently `full seconds / LOD seconds`.

| model | historical full attention | top-8-run LOD prefill | static-run LOD prefill |
|---|---:|---:|---:|
| Qwen3.8-27B-FP8 | 110.565 s | 81.190 s (1.36x) | 80.945 s (1.37x) |
| Gemma-4-26B-A4B | 40.063 s | 18.909 s (2.12x) | 18.957 s (2.11x) |
| Phi-4 TP5 | 28.119 s | 32.930 s (0.85x) | 33.751 s (0.83x) |
| Muse-Glimmer-30B | 51.933 s | 56.508 s (0.92x) | 55.040 s (0.94x) |
| OLMo-3-1125-32B | 67.892 s | 74.457 s (0.91x) | 97.710 s (0.69x) |

The first four top-8/static differences are small enough to be run-to-run
effects rather than distinct prefill algorithms. OLMo's 31% static-arm
regression is exceptional and should be rerun/profiled before treating static
as an end-to-end default there, even though static decode is 6.97% faster.

## Artifacts

- `qwen38_top8_64k_b8_r3.json`
- `qwen38_static_auto_64k_b8_r3.json`
- `gemma4_top8_64k_b8_r3.json`
- `gemma4_static_auto_64k_b8_r3.json`
- `phi4_tp5_top8_64k_b8_r3.json`
- `phi4_tp5_static_auto_64k_b8_r3.json`
- `olmo_top8_64k_b8_r3.json`
- `olmo_static_auto_64k_b8_r3.json`
- Muse: `../muse_cohort_speed_20260825/muse_unrestricted_top8_b8_r3.json`
  and `../muse_cohort_speed_20260825/muse_all_cohort_leaves_b8_r3.json`

The failed Gemma IPC-weight-cache startup was excluded: vLLM failed in its
MoE warmup before executing attention. Both valid Gemma arms used the standard
loader on the same node and GPU.
