# Fixed adjacent T/16, top-16 on Phi, Muse, and OLMo (2026-08-27)

## Result

Fixed adjacent T/16 centroids with top-16 opening do **not** improve 64K
prefill speed on any of the three previously difficult model families.  In a
fresh, matched vLLM panel, the fixed method is 3.9% slower than learned top-3
on Phi, 8.4% slower on Muse, and 7.1% slower on OLMo.

All rows use batch 8, eight distinct real ProLong documents, BF16 attention
state, one generated token to isolate prefill, a 64-token prompt reserve, one
warmup, and the median of three measured repetitions.  The scheduler permits
a 16,384-token chunk per request, so the aggregate batch-8 token budget is
131,072.  Full attention uses `ROCM_AITER_UNIFIED_ATTN`; Phi uses TP5 and the
other models use TP1.

| model | full attention | learned two-tier top-3 | fixed T/16 top-16 | fixed vs learned |
|---|---:|---:|---:|---:|
| Phi-4 TP5 | **24.385 s** | 31.888 s | 33.131 s | **+3.90% slower** |
| Muse-Glimmer-30B | 49.638 s | **49.407 s** | 53.552 s | **+8.39% slower** |
| OLMo-3-1125-32B | 66.368 s | **64.913 s** | 69.540 s | **+7.13% slower** |

Measured repetitions:

| model/method | repetitions (seconds) |
|---|---|
| Phi full | 24.437, 24.385, 24.202 |
| Phi learned | 32.833, 31.888, 31.877 |
| Phi fixed | 33.827, 33.131, 33.129 |
| Muse full | 49.690, 49.638, 49.631 |
| Muse learned | 49.407, 50.361, 48.365 |
| Muse fixed | 54.098, 52.053, 53.552 |
| OLMo full | 66.333, 66.368, 66.406 |
| OLMo learned | 65.644, 64.888, 64.913 |
| OLMo fixed | 69.540, 69.143, 69.611 |

The prompt hashes match within every model's three-way comparison, and every
artifact reports eight unique prompt hashes.  The fixed LOD dispatch audits
report `state_premerge_factor=16` and an effective prefill open count of 16;
the learned controls report factor 1 and open count 3.  Backend startup logs
and inference JIT logs confirm the full controls used the requested AITER
unified-attention path.  Their optional post-timing `apply_model` inspection
failed because insecure function serialization was not enabled, after the
complete timing JSON had already been persisted; this does not affect the
timing samples.

## Oversized-centroid hypothesis

The learned centroid distributions are very different across these models:

| model | centroids with >1024 leaves | leaf mass in >1024 centroids | leaf mass in <=16 centroids |
|---|---:|---:|---:|
| Muse | 0.0047% | 0.445% | 29.79% |
| Phi | 0.0627% | 6.371% | 19.06% |
| OLMo | 0.1287% | 20.994% | 17.36% |

OLMo genuinely has a severe unconditional tail, whereas Muse does not.  But
replacing every posting list with an exactly 16-token adjacent group still
makes OLMo slower.  This means the global existence of giant centroids is not
the principal end-to-end bottleneck.  The earlier direct diagnostic reaches
the same conclusion more narrowly: retaining learned top-3 routing while
capping each selected posting-list visit at 16 saved only 2.5% on Phi, 3.1%
on Muse, and 2.8% on OLMo.  Oversized selected lists contribute some cost but
cannot explain the original full-attention gap.

At 64K, T/16 and the ordinary `16*sqrt(T)` schedule both create about 4,096
coarse entries.  Fixed grouping therefore does not reduce coarse routing or
coarse-attention width.  It saves learned clustering/state-update work and
removes posting-list variance, but top-16 always opens 16 groups x 16 leaves,
or 256 exact remote leaves per query.  Learned top-3 usually performs much
less exact work.  The added leaf work outweighs the simpler state update on
all three families.

## Interpretation by model

- **Muse:** the latest hierarchical selector and overlap path is now tied
  with full attention under this scheduler.  Its centroid tail was already
  too small to explain a slowdown.  T/16 disables no required feature, but
  its extra exact work makes it clearly slower.
- **OLMo:** native-GQA packing plus the latest learned path is now 2.2% faster
  than full attention.  Its giant-centroid tail is real but is not dominant
  in the selected workload; T/16 loses 7.1% to learned routing.
- **Phi:** learned LOD remains 30.8% slower than full attention, and T/16 makes
  that 35.9% slower.  Phi's residual problem is therefore not posting-list
  imbalance.  It remains a kernel/parallel-efficiency issue for the
  D=128/GQA4 TP5 sparse route, coarse, and exact stages compared with native
  AITER attention.

The fixed layout has already shown acceptable Qwen quality, but no additional
large-model quality runs are justified by this speed screen.  It should not be
promoted as a speed path for these families.

## Scheduler-regime correction

Older large-model tables used a 16,384-token **aggregate** scheduler budget,
which serialized the batch and produced the historical Phi/Muse/OLMo learned
times of 43.216/51.514/69.286 seconds.  The current external-cache execution
contract uses 16,384 tokens **per request**, or 131,072 aggregate for batch 8.
Mixing those values initially made fixed Phi appear 23% faster.  The fresh
same-scheduler control above reverses that conclusion: fixed Phi is 3.9%
slower.  Only the matched table in this report should be used for this test.

## Artifacts

- `phi_full_64k_b8_r3_canonical.json`
- `phi_learned_top3_64k_b8_r3_canonical.json`
- `phi_t16_top16_64k_b8_r3_canonical.json`
- `muse_full_64k_b8_r3_canonical.json`
- `muse_learned_top3_64k_b8_r3_canonical.json`
- `muse_t16_top16_64k_b8_r3_canonical.json`
- `olmo_full_64k_b8_r3_canonical.json`
- `olmo_learned_top3_64k_b8_r3_canonical.json`
- `olmo_t16_top16_64k_b8_r3_canonical.json`

Related evidence:

- `../prefill_leaf_visit_cap_20260826/README.md`
- `../prefill_subkernel_profile_20260826/README.md`
- `../fixed_adjacent16_speed_20260827/README.md`
