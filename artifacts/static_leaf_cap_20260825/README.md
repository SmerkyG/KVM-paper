# Static leaf-count cutoff diagnostic

This diagnostic tests `Qwen/Qwen3.5-0.8B` at 64K with batch size 8, BF16
two-level LoD, the `16sqrt(T)` state schedule, chat formatting, and greedy
NIAH-S3 decoding.  It replaces query-dependent opening for the final decode
attention with a static rule:

- if a nonempty centroid has at most `C` remote leaves, attend to all of its
  leaves exactly and omit its coarse entry;
- otherwise, omit its leaves and retain its coarse mean with `log(count)`;
- sink and local-window tokens remain exact in every case.

Thus "discarded" below means discarded from *exact leaf attention*, not that
the centroid's represented probability mass is removed.  The fixed-list AITER
execution audit passed for every run.  No speed measurements were made.

## Leaf-count distribution

The distribution below is from one representative batch of eight 64K NIAH-S3
prompts, aggregated across the model's six LoD layers and two KV heads.  It
contains 390,816 nonempty centroid instances representing 6,193,152 remote
leaves.  The last column is the fraction of leaves retained if every centroid
through that bin is opened exactly.

| Leaves per centroid | Centroids | Leaves | Cumulative leaves retained |
|---:|---:|---:|---:|
| 1 | 4.109% | 0.259% | 0.259% |
| 2 | 4.398% | 0.555% | 0.814% |
| 3-4 | 12.045% | 2.683% | 3.498% |
| 5-8 | 23.314% | 9.435% | 12.933% |
| 9-16 | 27.216% | 20.540% | 33.473% |
| 17-32 | 18.553% | 26.645% | 60.117% |
| 33-64 | 7.770% | 21.471% | 81.588% |
| 65-128 | 2.061% | 11.038% | 92.626% |
| 129-256 | 0.418% | 4.465% | 97.091% |
| 257-512 | 0.099% | 2.130% | 99.221% |
| 513-1024 | 0.017% | 0.722% | 99.943% |
| >1024 | 0.0008% | 0.057% | 100.000% |

The tail is skewed, but the extremely large centroids are too rare to matter
by themselves.  For example, dropping exact leaves only for centroids above
128 leaves removes just 7.37% of leaf work.  The aggressive cutoff `C=16`
opens 71.1% of centroids but only 33.5% of leaves; `C=9` opens 48.6% of
centroids and only 15.6% of leaves.

## NIAH-S3 quality

All rows use the same 64 examples.  The cutoffs were screened on the first
eight examples and then extended over the remaining 56.

| Policy | Centroids opened | Remote leaves exact | NIAH-S3 at 64K |
|---|---:|---:|---:|
| Full attention | n/a | 100% | 64/64 |
| Static `C=8` | 43.9% | 12.9% | 62/64 |
| Static `C=9` | 48.6% | 15.6% | 64/64 |
| Static `C=10` | 52.8% | 18.3% | 63/64 |
| Static `C=12` | 60.3% | 23.7% | 64/64 |
| Static `C=14` | 66.3% | 28.8% | 64/64 |
| Static `C=16` | 71.1% | 33.5% | 64/64 |

Quality is not strictly monotonic in the cutoff: opening another set of
centroids replaces their count-corrected coarse approximation with exact
leaves, which can change later greedy choices in either direction.  Therefore
`C=9` is the smallest cutoff tested that matched full attention on this panel,
not a general proof that 15.6% is sufficient on arbitrary tasks.  `C=12` and
`C=16` are more conservative perfect points.

The raw JSON artifacts are named
`qwen08_staticcap{C}_64k_s8.json` and
`qwen08_staticcap{C}_64k_s56_offset8.json`; the matched full-attention result
is `qwen08_full_64k_s64.json`.  The detailed histogram is in
`qwen08_staticcap16_64k_s8_dist.json`.
