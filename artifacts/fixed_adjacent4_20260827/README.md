# Fixed adjacent-4 coarse groups (Qwen3.5-0.8B)

Quality diagnostic for replacing similarity-based KVM updates with fixed,
non-overlapping groups of four consecutive remote tokens. Each coarse entry
stores the summed K/V and a count of four (apart from a possible tail); coarse
attention consumes the mean and applies the usual `log(count)` mass correction.
Opening an entry restores attention to its four original exact leaves. The
512-token local window remains exact.

The HF prototype uses `state_premerge_factor=4` and
`state_growth_factor=128`. At the tested lengths, the state target is larger
than the number of groups, so every group is appended independently and no
similarity-based KVM merge occurs. This deliberately tests quality rather than
an optimized direct-append implementation.

## NIAH-S3

Chat template applied, thinking disabled, greedy decode.

| Context | Opening policy | Score |
|---:|---|---:|
| 8K | full-attention control | 64/64 |
| 8K | similarity-routed KVM LOD top-8 control | 64/64 |
| 8K | coarse means only | 0/8 |
| 8K | top-8 groups (32 exact remote tokens) | 64/64 |
| 8K | top-4 groups (16 exact remote tokens) | 64/64 |
| 64K | top-4 groups (16 exact remote tokens) | 8/8 |
| 8K | mass threshold 1/16, cap 16, prefill + decode | 7/8 |
| 8K | mass threshold 1/16 prefill, top-8 decode | 8/8 |
| 8K | top-8 prefill, mass threshold 1/16 decode | 8/8 |
| 8K | mass threshold 1/128, cap 16, prefill + decode | 8/8 |

The paged exact-leaf backend scored 5/8 even with ordinary top-8, while the
packed backend scored 64/64. Therefore the earlier paged mass-cutoff scores are
an implementation/layout confound and are not algorithmic mass-routing results.

## ProLong loss

Eight matched deterministic 8K documents, 65,528 prediction tokens total.
The aggregate CE is token-weighted and perplexity is `exp(CE)`.

| Method | CE | PPL | PPL vs full |
|---|---:|---:|---:|
| Full attention | 3.252045 | 25.8431 | baseline |
| Similarity-based KVM LOD, top-8 | 3.254731 | 25.9126 | +0.27% |
| Fixed adjacent-4, top-8 | 3.272414 | 26.3749 | +2.06% |
| Fixed adjacent-4, top-4 | 3.283240 | 26.6620 | +3.17% |
| Fixed adjacent-4, mass 1/128, cap 16 | 3.301290 | 27.1476 | +5.05% |

All eight fixed-adjacent top-8 documents had higher loss than their matched
full-attention counterpart. Thus this scheme is strong on sparse retrieval but
is materially worse than similarity-routed KVM centroids on average
next-token modeling.

## Interpretation

Fixed adjacent pooling is simple and preserves exact retrieval once the
relevant group is opened. It does not preserve enough token-level information
in the coarse-only branch, and its language-modeling degradation remains much
larger than ordinary KVM LOD even with top-8 opening.

It also keeps `T/4` coarse entries, so it has linear state rather than the
`O(sqrt(T))` state of scheduled LOD. Decode attends to a quarter-length coarse
field and naive prefill remains quadratic with a one-quarter coefficient. It
may still be attractive as a simple regular/GPU-friendly baseline, but it is
not a replacement for subquadratic LOD in its current form.
