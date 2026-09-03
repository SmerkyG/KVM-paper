# Fixed-list block sparsity at 64K

This diagnostic evaluates the proposed page-size-1 decode list on
`Qwen/Qwen3.5-0.8B`, using the actual two-tier top-8 routes from all six LOD
layers and unioning routes across the four query heads sharing each KV head.
It covers 64 greedy decode route steps, eight samples, and both ProLong and
RULER NIAH-S3.

The fixed logical list contains only valid indices, in this order:

1. protected sink;
2. active local BSWA entries;
3. active coarse centroids;
4. all valid leaves, packed in centroid-major order.

Sink and local entries are enabled. An unopened centroid's coarse entry is
enabled; an opened centroid's leaves are enabled instead. Blocks are formed
across the resulting *complete consecutive list*, so leaves from several small
centroids can share one block. There are no allocated page slots or per-page
padding in these counts.

## Prompts

- ProLong uses eight distinct 65,536-token prompts, each made from two real,
  distinct ProLong documents. The user turn ends with
  `Please summarize the foregoing documents.` and is rendered with Qwen's chat
  template (`enable_thinking=False`) before the assistant response is elicited.
- NIAH uses the first eight canonical RULER NIAH-S3 records with the same chat
  formatting used by the quality harness. Their natural rendered lengths are
  64,967--64,979 tokens. They are evaluated without padding; all eight target
  UUIDs occur in the greedy output.

## Raw GQA-unioned top-8

| Workload | Mean opened centroids | Mean opened leaves | N=16 zero blocks | N=16 issued / full attention | Useful lanes in issued N=16 blocks | N=64 issued / full attention | N=256 issued / full attention |
|---|---:|---:|---:|---:|---:|---:|---:|
| ProLong | 18.01 | 1,156.80 | 91.78% | 8.74% | 95.92% | 9.66% | 12.53% |
| NIAH-S3 | 16.45 | 693.09 | 92.22% | 8.27% | 97.00% | 8.85% | 10.67% |

The fixed list is longer than full attention because it includes both 4,095
coarse entries and the exact-token archive: 69,631 entries for 65,536-token
ProLong, and 69,049 versus 64,972 tokens on average for NIAH. The ratios above
therefore compare executed padded lanes against the original full-attention
token count, not merely against the longer fixed list.

The leaf section is even more structured: 97.90% of leaf-touching N=16 blocks
fast-fail on ProLong and 98.69% on NIAH. Once a block is live, its lanes are
usually all useful because the leaf indices belonging to an opened centroid
are consecutive.

## Refusing centroids with 1,024 or more leaves

| Workload | Mean opened leaves | N=16 zero blocks | N=16 issued / full attention | N=64 issued / full attention | N=256 issued / full attention |
|---|---:|---:|---:|---:|---:|
| ProLong | 1,049.10 | 91.94% | 8.57% | 9.49% | 12.35% |
| NIAH-S3 | 686.05 | 92.23% | 8.26% | 8.84% | 10.65% |

The 1,024-leaf rule barely changes NIAH and saves only 0.17 percentage points
of full-attention-equivalent N=16 work on ProLong. Very large selected
centroids are uncommon enough that this safeguard is not driving the main
sparsity result.

## Interpretation

N=16 is a viable inner fast-fail granularity here. It preserves almost all of
the ideal entry-level sparsity without launching tiny independent attention
operations: the mean issued work is about 5.72K lanes per ProLong query group
and 5.38K per NIAH query group. N=64 remains reasonably efficient, but N=256
alone loses materially more sparsity. A practical kernel should therefore use
a large resident/program tile (optionally with an N=256 summary OR mask) while
testing and executing its fixed list in N=16 subblocks.

Artifacts:

- `qwen35_08b_prolong_b8_64k.json`
- `qwen35_08b_niah_s3_64k.json`

Reproduction is implemented in
`scripts/analyze_fixed_list_block_sparsity.py`.
