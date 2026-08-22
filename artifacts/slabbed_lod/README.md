# Slabbed LOD prototype

This directory records the first quality-oriented prototype of slab-local LOD
construction. It uses Qwen3.5-0.8B without changing model weights.

## Design

- Split the sequence into fixed 4096-token slabs.
- Keep attention within the current and immediately preceding slabs exact and
  causal, matching the two-chunk BSWA geometry used by classic KVM. For slab
  width `B`, this is a blockwise BSWA whose exact span varies from `B` to
  `2B-1` tokens; it is not part of the remote LOD approximation.
- Once a slab is complete, reduce it independently to 256 semantic regions.
- Seed those regions with 256 positions spread uniformly through the completed
  slab, then assign every other key to its highest-dot-product seed centroid.
- Route each later query over older completed-slab summaries, open the top
  regions as exact BF16 leaves, and LSE-merge those leaves with the
  count-corrected coarse remainder and exact two-slab branch.
- During an initial prefill, fold the slab axis into the batch axis so all slab
  reductions are constructed concurrently.
- Optionally reduce groups of four old slabs while retaining the original leaf
  ownership. A 16√T budget trigger leaves all groups untouched through 64K and
  merges only the oldest groups needed to stay within the schedule after that.
  This is the first level of a recursive delayed-merge hierarchy.
- Optionally rescore the leaves of a small centroid-nominated candidate set to
  correct the coarse mass and rerank the regions that are opened.

The implementation in `model/slabbed_lod_attention.py` is deliberately a plain
PyTorch reference. Its timings measure the reference implementation, not the
expected fixed-shape kernel performance. The delayed merge currently implements
one hierarchy level; repeating the same four-child/two-density construction at
older levels gives the intended subquadratic schedule.

## Results

The initial prototype incorrectly kept only the current slab exact. Correcting
it to a two-slab exact field removed nearly all of the 8K and 16K loss gap.
ProLong cross-entropy loss below uses one sample except for the explicitly
marked four-sample 8K comparison:

### Block-size and reference-speed sweep

The following sweep fixes the summary density at one region per 16 tokens,
opens eight regions, and keeps two blocks exact. Times are the median of warm
samples 2--4 on one MI325X. They measure the plain PyTorch slab prototype; the
full-attention control uses the model's optimized attention path.

| Context | Block | Warm prefill | Loss delta vs full |
|---:|---:|---:|---:|
| 16K | 256 | 6.588 s | +0.03915 |
| 16K | 512 | 5.901 s | +0.02608 |
| 16K | 1024 | 5.936 s | +0.01674 |
| 16K | 2048 | 5.676 s | +0.00794 |
| 16K | 4096 | 3.886 s | +0.00371 |
| 16K | 8192 | 1.279 s | +0.00018 |
| 16K | full | 0.159 s | - |
| 32K | 1024 | 13.754 s | +0.03053 |
| 32K | 2048 | 13.634 s | +0.01759 |
| 32K | 4096 | 12.405 s | +0.00955 |
| 32K | 8192 | 9.711 s | +0.00380 |
| 32K | full | 0.551 s | - |
| 64K | 4096 | 35.454 s | +0.01913 |
| 64K | 8192 | 33.251 s | +0.01082 |
| 64K | 16384 | 27.240 s | +0.00399 |
| 64K | full | 0.971 s | - |

At 16K, a 4096-token block with only the current block exact took 5.600 s and
had loss 3.14855, versus 3.886 s and loss 3.14016 with the preceding block also
exact. In the reference engine, moving that preceding block from remote routing
to the dense local branch is therefore both faster and more accurate.

Larger blocks remain faster throughout this reference sweep, but that is an
implementation-overhead result rather than a production optimum. For long
sequences the exact branch performs about `1.5*T*B` query-key pairs, so its
arithmetic grows linearly with block width `B`. A fused/kernel implementation
must be swept again: it should eventually trade reduced state/routing overhead
against this increasing dense-local cost. The present absolute timings mainly
show that the PyTorch remote path needs kernels before it can compete with full
attention.

The optimized 4096-block path removes that reference overhead by using full
slab routing/construction tiles, split flash attention for the preceding and
causal-current slabs, a fused Triton route/coarse scan, and MoE-style packed
varlen flash attention over the union of the eight opened regions. The same
four-sample protocol gives:

| Context | Optimized slabbed | Full | Prefill-time change | Loss delta |
|---:|---:|---:|---:|---:|
| 16K | 0.184 s | 0.159 s | 15.6% slower | +0.00395 |
| 32K | 0.409 s | 0.551 s | 25.8% faster | +0.00973 |
| 64K | 0.923 s | 0.971 s | 5.0% faster | +0.01997 |

Packing routed queries by semantic region is the decisive change: at 16K it
reduces the otherwise optimized path from 0.753 s to 0.184 s. Precomputing the
posting order for the entire prefill was rejected; it increased the 16K time to
0.234 s and only reduced the 64K time from 0.923 s to 0.911 s. Sorting only the
remote prefixes that are actually queried is the better simple implementation.

| Context | Attention | Open regions | Loss | Delta vs full |
|---:|---|---:|---:|---:|
| 8K, 4 samples | full | - | 3.08219 | - |
| 8K, 4 samples | slabbed 256, two exact slabs | 8 | 3.08221 | +0.00002 |
| 16K | full | - | 3.33124 | - |
| 16K | slabbed 256, two exact slabs | 8 | 3.33204 | +0.00081 |
| 16K | 128 mass candidates | 32 | 3.33086 | -0.00038 |
| 32K | full | - | 3.30752 | - |
| 32K | slabbed 256 | 8 | 3.31231 | +0.00479 |
| 32K | slabbed 256 | 32 | 3.30988 | +0.00236 |
| 32K | slabbed 256 | 64 | 3.30942 | +0.00190 |
| 32K | delayed merge, 256 then 1024->512 | 32 | 3.30973 | +0.00221 |
| 32K | delayed merge, 256 then 1024->512 | 64 | 3.30863 | +0.00111 |
| 32K | delayed merge, 256 then 1024->512 | 128 | 3.30830 | +0.00078 |
| 32K | delayed merge, 512 then 2048->1024 | 64 | 3.30911 | +0.00159 |
| 32K | delayed merge, 512 then 2048->1024 | 128 | 3.30817 | +0.00065 |
| 32K | delayed 512, 64 mass candidates | 32 | 3.30888 | +0.00136 |
| 32K | budgeted 256, 128 mass candidates | 32 | 3.30830 | +0.00078 |
| 32K | eager-merged 256, 128 mass candidates | 32 | 3.30813 | +0.00061 |
| 32K | exact region-mass oracle | 8 | 3.30934 | +0.00182 |
| 32K | exact region-mass oracle | 16 | 3.30858 | +0.00106 |
| 32K | exact region-mass oracle | 32 | 3.30798 | +0.00046 |
| 64K | full | - | 2.81446 | - |
| 64K | budgeted 256, 128 mass candidates | 32 | 2.81727 | +0.00280 |
| 64K | eager-merged 256, 128 mass candidates | 32 | 2.81675 | +0.00229 |
| 64K | budgeted 256, exact-mass oracle | 64 | 2.81674 | +0.00228 |
| 64K | budgeted 256, exact-mass oracle | 128 | 2.81573 | +0.00127 |
| 64K | eager-merged 256, exact-mass oracle | 64 | 2.81652 | +0.00206 |
| 64K | budgeted 256, three exact slabs, exact mass | 128 | 2.81536 | +0.00090 |
| 64K | budgeted 256, four exact slabs, exact mass | 128 | 2.81500 | +0.00054 |
| 64K | budgeted 256, four exact slabs, 128 candidates | 32 | 2.81570 | +0.00124 |
| 64K | budgeted 256, four exact slabs, 128 candidates | 64 | 2.81559 | +0.00113 |
| 64K | eager-merged 512, exact mass | 128 | 2.81554 | +0.00108 |
| 64K | eager-merged 512, 128 candidates | 64 | 2.81728 | +0.00281 |

The exact-mass rows are diagnostic and score all remote leaves. They show that
the dominant remaining error is the Jensen gap in each centroid's coarse
softmax mass, not loss of the important leaves. A diagonal second-moment
correction did not capture this anisotropic error. Rescoring 128 nominated
regions and opening only 32 nearly matches the exact-mass/top-32 oracle while
loading values for only one quarter of the checked regions. It is the strongest
non-oracle result so far.

At 64K, the dominant residual shifts from coarse mass to the mean value of the
unopened regions. Doubling the base slab summaries helps the exact oracle only
slightly and hurts candidate routing. Keeping a wider fixed exact field is the
cleaner knob: four exact slabs lower the practical 128-candidate/32-open gap to
+0.00124, 64 opens lower it to +0.00113, and the exact-mass/top-128 upper
bound is +0.00054. The small gain from 32 to 64 practical opens indicates that
candidate recall/coarse mass, rather than value loading alone, is the next
thing an optimized hierarchy would need to improve.

The following NIAH scores are retained as historical results from the original
one-exact-slab prototype; they should not be used as the quality comparison for
the corrected two-slab implementation:

| Context | Attention | Open regions | NIAH-1 | NIAH-2 | NIAH-3 |
|---:|---|---:|---:|---:|---:|
| 8K | slabbed, prefix seeds | 8 | 8/8 | 8/8 | 8/8 |
| 16K | full | - | 8/8 | 8/8 | 8/8 |
| 16K | slabbed, prefix seeds | 8 | 6/8 | 7/8 | 8/8 |
| 16K | slabbed, prefix seeds | 16 | 5/8 | 8/8 | - |
| 16K | slabbed, strided seeds | 8 | 8/8 | 8/8 | 8/8 |
| 16K | slabbed, strided seeds | 16 | 8/8 | - | - |
| 32K | slabbed, strided seeds | 8 | 8/8 | 7/8 | 7/8 |
| 32K | slabbed, strided seeds | 16 | - | 7/8 | 8/8 |
| 32K, 512 regions/slab | slabbed, strided seeds | 8 | - | 8/8 | 8/8 |
| 32K | full | - | - | 8/8 | 8/8 |

Strided seeds remain the default. They are causal because a slab summary is not
exposed until the slab is complete. The delayed merge does not materially harm
the exact-mass upper bound, which supports retaining it as the route to a
recursive schedule. The next quality/performance decision is whether to open
more small regions or retain multi-representative mass information inside each
closed region.

## Reproduction

The Qwen3.5 optional fast-path dependencies are required:

```bash
uv run --extra qwen35-fast-path python -m scripts.compare_hf_lod_loss \
  --checkpoint Qwen/Qwen3.5-0.8B --sequence-length 16384 --samples 1 \
  --mode lod --engine-backend torch --slabbed --slab-size 4096 \
  --slots-per-slab 256 --slab-local-slabs 2 \
  --slab-seed-selection strided --open-count 8 \
  --output artifacts/slabbed_lod/prolong_slab_strided_4096x256_top8_16k_s1.json

uv run --extra qwen35-fast-path python -m scripts.eval_hf_lod_lmeval \
  --checkpoint Qwen/Qwen3.5-0.8B --mode lod --tasks niah_single_3 \
  --batch-size 8 --ruler-length 16384 --limit 8 --disable-thinking \
  --engine-backend torch --slabbed --slab-size 4096 --slots-per-slab 256 \
  --slab-local-slabs 2 --slab-seed-selection strided --open-count 8 \
  --left-padding-mode exact \
  --output artifacts/slabbed_lod/niah3_slab_strided_4096x256_top8_16k_n8.json
```
