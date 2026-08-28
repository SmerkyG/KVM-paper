# Fixed adjacent-32 coarse groups (Qwen3.5-0.8B)

Quality diagnostic for fixed, non-overlapping groups of 32 consecutive remote
tokens. Each selected parent is fully opened, so top-8 evaluates 256 exact
remote leaves per query and top-16 evaluates 512. All settings match the T/16
all-leaves panel: BF16, 512 exact local tokens, exact leaf storage, coarse mean
K/V with `log(count)`, chat-formatted NIAH-S3 with thinking disabled, greedy
decoding, and `state_growth_factor=128` to prevent later similarity merges.
ProLong uses the same eight deterministic 8K documents and 65,528 prediction
tokens as the preceding fixed-pooling diagnostics.

## Results

| Method | Exact remote leaves/query | 8K NIAH-S3 | 64K NIAH-S3 | ProLong CE | ProLong PPL | PPL vs full |
|---|---:|---:|---:|---:|---:|---:|
| Full attention | all | 64/64 | - | 3.252045 | 25.8431 | baseline |
| Similarity-routed KVM LOD, top-8 | variable | 64/64 | - | 3.254731 | 25.9126 | +0.27% |
| Fixed T/16, top-8, all leaves | 128 | 63/64 | 7/8 | 3.273635 | 26.4072 | +2.18% |
| **Fixed T/32, top-8, all leaves** | **256** | **62/64** | **6/8** | **3.273451** | **26.4023** | **+2.16%** |
| Fixed T/16, top-16, all leaves | 256 | 64/64 | 8/8 | 3.265883 | 26.2032 | +1.39% |
| **Fixed T/32, top-16, all leaves** | **512** | **64/64** | **8/8** | **3.264797** | **26.1748** | **+1.28%** |

## Interpretation

T/32 top-16 preserves retrieval and is the lowest-loss fixed-pooling variant
tested so far, but its ProLong improvement over T/16 top-16 is only 0.11% in
perplexity while it doubles exact-leaf work. It halves the coarse-field length,
so whether that trade is worthwhile depends on the optimized coarse-versus-leaf
kernel crossover at the target context length.

The equal-leaf-work comparison is more diagnostic. T/32 top-8 and T/16 top-16
both attend to 256 exact remote leaves, but T/16 top-16 gives better ProLong
PPL (26.2032 versus 26.4023) and better retrieval (64/64 and 8/8 versus 62/64
and 6/8). Finer parents plus more routes use the same exact-leaf budget more
effectively than blurrier parents plus fewer routes.

Top-16 currently uses the ordinary top-k and packed-leaf path because the
fused routing specialization supports at most eight routes. These results are
quality measurements, not representative optimized top-16 timings.
