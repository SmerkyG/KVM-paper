# INT4 family validation and sub-4-bit study

Date: 2026-08-27

## INT4 result

The recursive three-tier INT4 default (16-channel groups, page-wide token
group, two-step least-squares scale refinement, and INT8 page summaries)
generalizes to both Gemma-4 and Muse-Glimmer in this panel.

All NIAH-S3 runs use 64 matched examples at 64K, batch eight, the model chat
template, thinking disabled, a 16K scheduler token budget, and direct LOD
prefill and decode. ProLong uses eight concatenated real 8K documents and
direct LOD prefill. The ProLong timing is evaluator wall time, not a
controlled speed benchmark.

| Model | Storage | NIAH-S3 | ProLong CE | CE delta vs BF16 |
|---|---|---:|---:|---:|
| Gemma-4-26B-A4B | BF16 | 61/64 | 7.398160 | reference |
| Gemma-4-26B-A4B | legacy INT4, G32 max | not run | 7.405527 | +0.007366 |
| Gemma-4-26B-A4B | **INT4, G16 L2** | **63/64** | **7.362806** | **-0.035354** |
| Muse-Glimmer-30B | BF16 | 64/64 | 2.160909 | reference |
| Muse-Glimmer-30B | legacy INT4, G32 max | not run | 2.161216 | +0.000307 |
| Muse-Glimmer-30B | **INT4, G16 L2** | **64/64** | **2.161019** | **+0.000110** |

Muse's CE penalty falls by 64.2% relative to legacy INT4. Gemma's raw-document
CE is too noisy to interpret as a quantization improvement: the instruction
model has high loss on this unformatted corpus, and the quantized run happened
to score lower than BF16. Its matched NIAH result is the useful evidence here.

## Sub-4-bit Qwen diagnostic

These rows simulate quantize/dequantize (QDQ) while retaining BF16 physical
storage. They measure quality only. The effective payload estimates assume
native packed codes plus one BF16 scale per token-by-channel group; page
summaries and shared LOD metadata are excluded.

No native odd-bit pack/unpack attention kernel was built: three-bit packing
would require eight codes per three bytes and must still prove that its added
bit extraction does not erase the memory-bandwidth benefit.

| Format | Scale group | Effective bits/value | ProLong CE | 48-choice score | Agreement with BF16 | RMS margin drift |
|---|---|---:|---:|---:|---:|---:|
| BF16 LOD | n/a | 16.000 | 1.925134 | 11/48 | reference | reference |
| improved INT4 K/V | 16 tokens x 16 channels | 4.063 | 1.925276 | 12/48 | **91.67%** | **0.3292** |
| K3/V4 | 16 tokens x 16 channels | 3.563 average | **1.925279** | 12/48 | 79.17% | 0.4719 |
| K3/V3 | 4 tokens x 32 channels | 3.125 | 1.925362 | **13/48** | 79.17% | 0.4341 |
| K3/V3, separate positive/negative scales | 4 tokens x 32 channels | 3.250 | 1.925306 | 11/48 | 75.00% | 0.4296 |
| K2/V2 | 1 token x 16 channels | 3.000 | 1.925814 | not run | not run | not run |

K3/V4 is the most plausible next format: it cuts estimated leaf payload by
12.3% versus improved INT4 and is essentially tied on ProLong CE. Pure 3-bit
would cut it by 23.1%. Neither is yet quality-neutral, however: both create
substantially more choice-distribution drift than INT4. The separate-sign
scale adds metadata and did not help the harder panel, so it was removed from
the implementation rather than retained as another mode. Two-bit storage is
rejected by the CE test.

Both K3/V4 and pure K3/V3 nevertheless scored 64/64 on the matched Qwen
NIAH-S3 8K panel. This confirms that the candidate formats preserve easy exact
retrieval, but does not override the more sensitive choice-distribution test.

## Similar-leaf grouping

Leaves may be reordered after positional encoding without changing exact
attention, so grouping similar leaves within each centroid could reduce the
range of page residuals and sharpen page summaries. A practical online design
would keep two to four open 16-leaf page builders per centroid and append to
the closest page mean; an initial prefill could instead repack once and leave
decode appends online.

This is not the first choice yet. It adds routing/update work and irregular
metadata, and most short-context centroids contain no more than one page.
Token-group scaling obtains a similar reduction in quantization range while
keeping the regular page layout and attention kernel. Similar-leaf grouping
should therefore be tested first as an offline residual-error diagnostic on
the multi-page centroid tail, and implemented only if it closes a gap that
regular K3/V4 scaling cannot.

## Artifacts

The JSON files in this directory contain the Gemma and Muse results. The Qwen
sub-4-bit ProLong, margin-panel, and NIAH artifacts are in
`../sub4_quality_20260827/`.
