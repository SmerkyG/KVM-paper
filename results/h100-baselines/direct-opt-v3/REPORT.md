# H100 threshold prefill: optimized retime

Llama-3.1-8B-Instruct, BF16 weights/KV, batch one, same H100 80GB. All GPU runs use uv and immutable cluster-run snapshots.

## Full-model prefill

One warmup and three measured repetitions per arm; medians below exclude warmup. Timing includes local/sink attention, the complete initial exact field, state/cache updates and deferred cache completion. Initial allocation, loading and decode graph capture are excluded consistently. Prefill is eager for every arm; these are not graph-replay or attention-only timings.

| Context | Dense | Top-eight | Previous optimized threshold | New threshold | New versus dense |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 1.553 s | 1.622 s | 1.756 s | 1.663 s | 7.1% more time |
| 65,536 | 4.749 s | 4.101 s | 4.737 s | 4.518 s | 4.9% less time |
| 128,000 | 15.697 s | 9.420 s | 11.816 s | 11.528 s | 26.6% less time |

Uncapped equal-per-region c=16, the 1,024-entry region eligibility limit and the 16,384-token initial exact field are unchanged. Internal scratch tiles now use 16,384 queries through 2,048 active regions, 8,192 through 4,096 regions and 4,096 above that (previously 4,096 throughout). The 128,000-token run explicitly retains 4,096-query scratch tiles throughout for memory headroom; this does not change the exact initial field or routing. MLP chunks remain 8,192. This is not the per-entry bounded rule.

## Implementation and limits

- Native QK is saved in FP32. Final threshold selection and closed-weight preparation share the same saved-score pass, retaining the previously validated globally shifted FP16 coarse projection.
- Selection bits are compacted and packed without sorting, count readbacks or a region cap. Query scaling retains its explicit rounded multiplication before log-count addition.
- Common exact frontiers use a block diagonal tensor-core reduction with a three-part BF16 expansion of FP32 weights. Frontiers larger than 32 regions use the uncapped original reduction.
- Producer pipelining uses two stages through 2,048 regions and one stage for larger states. The QK tile remains 64 queries by 128 regions with four warps.
- Cache-key RMS normalization fuses the existing PyTorch 2.5 operations while reproducing its reduction order and rounding. The guarded BF16/D128 path falls back to PyTorch for unsupported shapes or versions.
- State assignment keeps the original cuBLAS score matrix. A packed unsigned reduction returns its BF16 maximum and first winning index without separate value/index reductions. NaN positions and indices match; NaN payload bits are outside the comparison contract.
- All cache work remains inside full-prefill timing. The fresh top-eight control uses the same new cache normalization and max reductions.

The increased scratch tile trades memory for speed. Peak PyTorch allocation below includes weights and excludes allocations made outside PyTorch. Dense and top-eight controls are freshly rerun at every length; the previous optimized-threshold column uses earlier completed matched retimes. The longest prompt is 128,000 tokens, not 131,072. No 128K quality panel has been rerun. No complete-execution subquadratic bound is claimed. The remaining 32K gap is reported explicitly; this result does not establish a tie with dense.

| Context | Previous threshold peak | New threshold peak | New measured samples |
|---:|---:|---:|---|
| 32,768 | 33.43 GiB | 64.20 GiB | 1.6626 s, 1.6626 s, 1.6654 s |
| 65,536 | 41.06 GiB | 70.68 GiB | 4.5176 s, 4.5175 s, 4.5208 s |
| 128,000 | 66.44 GiB | 66.44 GiB | 11.5267 s, 11.5450 s, 11.5277 s |

## Validation

The [full 32K RULER panel](RULER_32K.md) scored **87.95%** using the accepted timing snapshot and kernel settings. The table below retains the targeted checks used during tuning.

| Task | Examples | Fresh previous-selector control | Candidate |
|---|---:|---:|---:|
| cwe | 32 | 34.38% | 35.94% |
| niah_multikey_2 | 2 | 100.00% | 100.00% |
| niah_multikey_3 | 2 | 100.00% | 100.00% |
| qa_1 | 2 | 100.00% | 100.00% |

Across these 38 examples: 29 scores are unchanged, 5 improve and 4 are lower than the matched previous-selector control. These differences are numerical generation outcomes; they are not evidence that this implementation improves the routing rule.

At 32,768 tokens: 32 checked frontiers, zero selection differences outside tolerance, maximum frontier-reference error 0.0078125, and exact same-cache eager/graph decode logits.

At 65,536 tokens: 96 checked frontiers, zero selection differences outside tolerance, maximum frontier-reference error 0.0078125, and exact same-cache eager/graph decode logits.

At 128,000 tokens: 224 checked frontiers, zero selection differences outside tolerance, maximum frontier-reference error 0.0156250, and exact same-cache eager/graph decode logits.

Independent PyTorch checks find zero BF16 output-bit differences across 4,391,936 normalized values, with identical FP32 means, square roots and reciprocals. An ordinary Triton reduction is retained as a negative control and does differ. The packed maximum check covers 1,520 rows, including ties, signed zeros, infinities and NaNs, with identical indices, non-NaN value bits and NaN positions.

The 3 uncapped synthetic stress cases have zero selection mismatches and reach 45 opened regions, exercising both the tensor-core reducer and its larger-frontier fallback.

The faster online-coarse and tiled-cache candidates were rejected after quality checks. With both changes CWE scored 29.38%; restoring the original cache update alone scored 30.00%. Keeping the original coarse projection with tensor-core reduction scored 36.56% in the intermediate full CWE control. The table above uses the final full 32-example rerun, with no diagnostic source-row filter.

Apply these overrides to the existing experimental scaled-FP16 direct configuration:

```text
LOD_H100_CLOSED_REDUCE=tc-wide
LOD_H100_TC_REDUCE_ROWS=4
LOD_H100_TC_REDUCE_PARTS=3
LOD_H100_FAST_TILE=auto
LOD_H100_FAST_PRODUCE_STAGES=auto
LOD_H100_STATE_FUSED_RMS=1
LOD_H100_STATE_PACKED_MAX=1
```

Use `LOD_H100_FAST_TILE=4096` at 128,000 tokens. The scratch dispatch is separate from the unchanged 16K initial exact field. Serving defaults and other models are unchanged.


## Diagnostic trials

The following are one-repetition instrumented profiles, with warmup excluded. Nested stage timings overlap and must not be added to their parent timings. They are separate from the three-repetition full-model table.

<details>
<summary>Profiles, including regressing variants</summary>

| Raw profile | Context | Wall seconds | Routing pass | Closed pass | Exact kernel | Exact reduction |
|---|---:|---:|---:|---:|---:|---:|
| [threshold-directopt-accepted-v3-select-32768-profiles.json](threshold-directopt-accepted-v3-select-32768-profiles.json) | 32768 | 1.6678 | 0.0952 | 0.1508 | 0.0910 | 0.0507 |
| [threshold-directopt-accepted-v3-select-65536-profiles.json](threshold-directopt-accepted-v3-select-65536-profiles.json) | 65536 | 4.5550 | 0.4063 | 0.7459 | 0.5200 | 0.1790 |
| [threshold-directopt-auxiliary-v3-select-32768-profiles.json](threshold-directopt-auxiliary-v3-select-32768-profiles.json) | 32768 | 1.7660 | 0.0967 | 0.1567 | 0.1028 | 0.0968 |
| [threshold-directopt-auxiliary-v3-select-65536-profiles.json](threshold-directopt-auxiliary-v3-select-65536-profiles.json) | 65536 | 4.7700 | 0.4227 | 0.7514 | 0.5471 | 0.2955 |
| [threshold-directopt-combined-v3-select-32768-profiles.json](threshold-directopt-combined-v3-select-32768-profiles.json) | 32768 | 1.6868 | 0.0608 | 0.1627 | 0.1026 | 0.0582 |
| [threshold-directopt-combined-v3-select-65536-profiles.json](threshold-directopt-combined-v3-select-65536-profiles.json) | 65536 | 4.4285 | 0.2445 | 0.7405 | 0.5507 | 0.2041 |
| [threshold-directopt-compactready-v3-select-32768-profiles.json](threshold-directopt-compactready-v3-select-32768-profiles.json) | 32768 | 1.7771 | 0.0964 | 0.1556 | 0.1151 | 0.0993 |
| [threshold-directopt-compactready-v3-select-65536-profiles.json](threshold-directopt-compactready-v3-select-65536-profiles.json) | 65536 | 4.8431 | 0.4225 | 0.7551 | 0.6064 | 0.3061 |
| [threshold-directopt-decoupled-v3-select-32768-profiles.json](threshold-directopt-decoupled-v3-select-32768-profiles.json) | 32768 | 1.6825 | 0.0564 | 0.1633 | 0.1031 | 0.0584 |
| [threshold-directopt-decoupled-v3-select-65536-profiles.json](threshold-directopt-decoupled-v3-select-65536-profiles.json) | 65536 | 4.3893 | 0.2204 | 0.7394 | 0.5480 | 0.2035 |
| [threshold-directopt-decoupledwide-v3-select-32768-profiles.json](threshold-directopt-decoupledwide-v3-select-32768-profiles.json) | 32768 | 1.6480 | 0.0522 | 0.1545 | 0.0904 | 0.0567 |
| [threshold-directopt-decoupledwide-v3-select-65536-profiles.json](threshold-directopt-decoupledwide-v3-select-65536-profiles.json) | 65536 | 4.3317 | 0.2082 | 0.7368 | 0.5165 | 0.1984 |
| [threshold-directopt-defer-v3-select-32768-profiles.json](threshold-directopt-defer-v3-select-32768-profiles.json) | 32768 | 1.7322 | 0.0955 | 0.1446 | 0.0901 | 0.0958 |
| [threshold-directopt-defer-v3-select-65536-profiles.json](threshold-directopt-defer-v3-select-65536-profiles.json) | 65536 | 4.7058 | 0.4516 | 0.7233 | 0.5151 | 0.2928 |
| [threshold-directopt-detailedstate-v3-select-32768-profiles.json](threshold-directopt-detailedstate-v3-select-32768-profiles.json) | 32768 | 1.6858 | 0.0952 | 0.1501 | 0.0898 | 0.0503 |
| [threshold-directopt-final-v3-select-32768-profiles.json](threshold-directopt-final-v3-select-32768-profiles.json) | 32768 | 1.6907 | 0.0954 | 0.1504 | 0.0906 | 0.0505 |
| [threshold-directopt-final-v3-select-65536-profiles.json](threshold-directopt-final-v3-select-65536-profiles.json) | 65536 | 4.6118 | 0.4291 | 0.7422 | 0.5160 | 0.1773 |
| [threshold-directopt-fusedrms-v3-select-32768-profiles.json](threshold-directopt-fusedrms-v3-select-32768-profiles.json) | 32768 | 1.6608 | 0.0951 | 0.1506 | 0.0905 | 0.0507 |
| [threshold-directopt-fusedrms-v3-select-65536-profiles.json](threshold-directopt-fusedrms-v3-select-65536-profiles.json) | 65536 | 4.5551 | 0.4294 | 0.7435 | 0.5161 | 0.1774 |
| [threshold-directopt-globalfused-v3-select-32768-profiles.json](threshold-directopt-globalfused-v3-select-32768-profiles.json) | 32768 | 1.7386 | 0.0954 | 0.1979 | 0.0908 | 0.0504 |
| [threshold-directopt-globalfused-v3-select-65536-profiles.json](threshold-directopt-globalfused-v3-select-65536-profiles.json) | 65536 | 4.7519 | 0.4249 | 0.9169 | 0.5126 | 0.1763 |
| [threshold-directopt-groupreduce2-v3-select-32768-profiles.json](threshold-directopt-groupreduce2-v3-select-32768-profiles.json) | 32768 | 1.7654 | 0.0964 | 0.1563 | 0.1024 | 0.1045 |
| [threshold-directopt-groupreduce2-v3-select-65536-profiles.json](threshold-directopt-groupreduce2-v3-select-65536-profiles.json) | 65536 | 4.8036 | 0.4242 | 0.7539 | 0.5498 | 0.3201 |
| [threshold-directopt-groupreduce4-v3-select-32768-profiles.json](threshold-directopt-groupreduce4-v3-select-32768-profiles.json) | 32768 | 1.7542 | 0.0964 | 0.1563 | 0.1024 | 0.0909 |
| [threshold-directopt-groupreduce4-v3-select-65536-profiles.json](threshold-directopt-groupreduce4-v3-select-65536-profiles.json) | 65536 | 4.7436 | 0.4227 | 0.7517 | 0.5481 | 0.2824 |
| [threshold-directopt-inneraudit-v3-select-32768-profiles.json](threshold-directopt-inneraudit-v3-select-32768-profiles.json) | 32768 | 1.6968 | 0.0957 | 0.1505 | 0.0905 | 0.0506 |
| [threshold-directopt-inneraudit-v3-select-65536-profiles.json](threshold-directopt-inneraudit-v3-select-65536-profiles.json) | 65536 | 4.6152 | 0.4516 | 0.7420 | 0.5145 | 0.1767 |
| [threshold-directopt-leaf16w2-v3-select-32768-profiles.json](threshold-directopt-leaf16w2-v3-select-32768-profiles.json) | 32768 | 1.7940 | 0.0963 | 0.1558 | 0.1377 | 0.0962 |
| [threshold-directopt-leaf16w2-v3-select-65536-profiles.json](threshold-directopt-leaf16w2-v3-select-65536-profiles.json) | 65536 | 4.9968 | 0.4231 | 0.7551 | 0.7759 | 0.2960 |
| [threshold-directopt-leafw2-v3-select-32768-profiles.json](threshold-directopt-leafw2-v3-select-32768-profiles.json) | 32768 | 1.8308 | 0.0961 | 0.1554 | 0.1790 | 0.0957 |
| [threshold-directopt-leafw2-v3-select-65536-profiles.json](threshold-directopt-leafw2-v3-select-65536-profiles.json) | 65536 | 5.2133 | 0.4228 | 0.7545 | 0.9913 | 0.2959 |
| [threshold-directopt-matched-v3-select-32768-profiles.json](threshold-directopt-matched-v3-select-32768-profiles.json) | 32768 | 1.9228 | 0.1964 | 0.0000 | 0.1009 | 0.0951 |
| [threshold-directopt-matched-v3-select-65536-profiles.json](threshold-directopt-matched-v3-select-65536-profiles.json) | 65536 | 5.3311 | 0.9413 | 0.0000 | 0.5440 | 0.2938 |
| [threshold-directopt-memorytile-v3-select-32768-profiles.json](threshold-directopt-memorytile-v3-select-32768-profiles.json) | 32768 | 1.6630 | 0.0951 | 0.1506 | 0.0908 | 0.0507 |
| [threshold-directopt-memorytile-v3-select-65536-profiles.json](threshold-directopt-memorytile-v3-select-65536-profiles.json) | 65536 | 4.5530 | 0.4064 | 0.7457 | 0.5188 | 0.1787 |
| [threshold-directopt-n16-v3-select-32768-profiles.json](threshold-directopt-n16-v3-select-32768-profiles.json) | 32768 | 1.7658 | 0.0963 | 0.1557 | 0.1092 | 0.0963 |
| [threshold-directopt-n16-v3-select-65536-profiles.json](threshold-directopt-n16-v3-select-65536-profiles.json) | 65536 | 4.8599 | 0.4241 | 0.7572 | 0.6233 | 0.2980 |
| [threshold-directopt-normw1-v3-select-32768-profiles.json](threshold-directopt-normw1-v3-select-32768-profiles.json) | 32768 | 1.7994 | 0.0964 | 0.1962 | 0.1023 | 0.0962 |
| [threshold-directopt-normw1-v3-select-65536-profiles.json](threshold-directopt-normw1-v3-select-65536-profiles.json) | 65536 | 5.0021 | 0.4226 | 0.9906 | 0.5467 | 0.2959 |
| [threshold-directopt-normw2-v3-select-32768-profiles.json](threshold-directopt-normw2-v3-select-32768-profiles.json) | 32768 | 1.7697 | 0.0965 | 0.1627 | 0.1024 | 0.0965 |
| [threshold-directopt-normw2-v3-select-65536-profiles.json](threshold-directopt-normw2-v3-select-65536-profiles.json) | 65536 | 4.7591 | 0.4233 | 0.7368 | 0.5505 | 0.2967 |
| [threshold-directopt-normw4-v3-select-32768-profiles.json](threshold-directopt-normw4-v3-select-32768-profiles.json) | 32768 | 1.7631 | 0.0966 | 0.1559 | 0.1030 | 0.0968 |
| [threshold-directopt-normw4-v3-select-65536-profiles.json](threshold-directopt-normw4-v3-select-65536-profiles.json) | 65536 | 4.7914 | 0.4236 | 0.7724 | 0.5473 | 0.2972 |
| [threshold-directopt-pack4-v3-select-32768-profiles.json](threshold-directopt-pack4-v3-select-32768-profiles.json) | 32768 | 1.8331 | 0.0965 | 0.2273 | 0.1025 | 0.0969 |
| [threshold-directopt-pack4-v3-select-65536-profiles.json](threshold-directopt-pack4-v3-select-65536-profiles.json) | 65536 | 5.1536 | 0.4226 | 1.1491 | 0.5456 | 0.2956 |
| [threshold-directopt-packedmax-v3-select-32768-profiles.json](threshold-directopt-packedmax-v3-select-32768-profiles.json) | 32768 | 1.6874 | 0.0955 | 0.1507 | 0.0906 | 0.0507 |
| [threshold-directopt-packedmax-v3-select-65536-profiles.json](threshold-directopt-packedmax-v3-select-65536-profiles.json) | 65536 | 4.5878 | 0.4279 | 0.7403 | 0.5184 | 0.1767 |
| [threshold-directopt-persistent-v3-select-32768-profiles.json](threshold-directopt-persistent-v3-select-32768-profiles.json) | 32768 | 1.8288 | 0.0966 | 0.1559 | 0.1026 | 0.1647 |
| [threshold-directopt-persistent-v3-select-65536-profiles.json](threshold-directopt-persistent-v3-select-65536-profiles.json) | 65536 | 4.9693 | 0.4226 | 0.7541 | 0.5459 | 0.5061 |
| [threshold-directopt-persistent528-v3-select-32768-profiles.json](threshold-directopt-persistent528-v3-select-32768-profiles.json) | 32768 | 1.8505 | 0.0962 | 0.1556 | 0.1016 | 0.1903 |
| [threshold-directopt-persistent528-v3-select-65536-profiles.json](threshold-directopt-persistent528-v3-select-65536-profiles.json) | 65536 | 5.0481 | 0.4224 | 0.7540 | 0.5460 | 0.5847 |
| [threshold-directopt-popcount-v3-select-32768-profiles.json](threshold-directopt-popcount-v3-select-32768-profiles.json) | 32768 | 1.6965 | 0.0953 | 0.1574 | 0.0905 | 0.0505 |
| [threshold-directopt-popcount-v3-select-65536-profiles.json](threshold-directopt-popcount-v3-select-65536-profiles.json) | 65536 | 4.6144 | 0.4512 | 0.7498 | 0.5158 | 0.1772 |
| [threshold-directopt-precision-v3-select-32768-profiles.json](threshold-directopt-precision-v3-select-32768-profiles.json) | 32768 | 1.6776 | 0.0521 | 0.1831 | 0.0905 | 0.0505 |
| [threshold-directopt-precision-v3-select-65536-profiles.json](threshold-directopt-precision-v3-select-65536-profiles.json) | 65536 | 4.4200 | 0.2062 | 0.7834 | 0.5106 | 0.1771 |
| [threshold-directopt-prod128-v3-select-32768-profiles.json](threshold-directopt-prod128-v3-select-32768-profiles.json) | 32768 | 1.8855 | 0.2259 | 0.1555 | 0.1020 | 0.0956 |
| [threshold-directopt-prod128-v3-select-65536-profiles.json](threshold-directopt-prod128-v3-select-65536-profiles.json) | 65536 | 4.9782 | 0.6438 | 0.7540 | 0.5460 | 0.2953 |
| [threshold-directopt-prod256-v3-select-32768-profiles.json](threshold-directopt-prod256-v3-select-32768-profiles.json) | 32768 | 1.9023 | 0.2461 | 0.1550 | 0.1006 | 0.0949 |
| [threshold-directopt-prod256-v3-select-65536-profiles.json](threshold-directopt-prod256-v3-select-65536-profiles.json) | 65536 | 5.0534 | 0.7071 | 0.7536 | 0.5475 | 0.2956 |
| [threshold-directopt-producer128s1-v3-select-32768-profiles.json](threshold-directopt-producer128s1-v3-select-32768-profiles.json) | 32768 | 1.6957 | 0.0980 | 0.1486 | 0.0911 | 0.0507 |
| [threshold-directopt-producer128s1-v3-select-65536-profiles.json](threshold-directopt-producer128s1-v3-select-65536-profiles.json) | 65536 | 4.5475 | 0.4296 | 0.7274 | 0.5153 | 0.1769 |
| [threshold-directopt-producer32-v3-select-32768-profiles.json](threshold-directopt-producer32-v3-select-32768-profiles.json) | 32768 | 1.6997 | 0.1071 | 0.1484 | 0.0907 | 0.0505 |
| [threshold-directopt-producer32-v3-select-65536-profiles.json](threshold-directopt-producer32-v3-select-65536-profiles.json) | 65536 | 4.6336 | 0.5066 | 0.7271 | 0.5203 | 0.1768 |
| [threshold-directopt-producer64-v3-select-32768-profiles.json](threshold-directopt-producer64-v3-select-32768-profiles.json) | 32768 | 1.6906 | 0.0981 | 0.1485 | 0.0904 | 0.0506 |
| [threshold-directopt-producer64-v3-select-65536-profiles.json](threshold-directopt-producer64-v3-select-65536-profiles.json) | 65536 | 4.6887 | 0.5252 | 0.7276 | 0.5107 | 0.1771 |
| [threshold-directopt-qkblas-v3-select-32768-profiles.json](threshold-directopt-qkblas-v3-select-32768-profiles.json) | 32768 | 1.7726 | 0.0585 | 0.1997 | 0.1028 | 0.0968 |
| [threshold-directopt-qkblas-v3-select-65536-profiles.json](threshold-directopt-qkblas-v3-select-65536-profiles.json) | 65536 | 5.0224 | 0.3555 | 1.0610 | 0.5523 | 0.2969 |
| [threshold-directopt-qktriton-v3-select-32768-profiles.json](threshold-directopt-qktriton-v3-select-32768-profiles.json) | 32768 | 1.7954 | 0.0805 | 0.1997 | 0.1028 | 0.0968 |
| [threshold-directopt-qktriton-v3-select-65536-profiles.json](threshold-directopt-qktriton-v3-select-65536-profiles.json) | 65536 | 5.0823 | 0.4388 | 1.0576 | 0.5495 | 0.2956 |
| [threshold-directopt-retime-v3-select-32768-profiles.json](threshold-directopt-retime-v3-select-32768-profiles.json) | 32768 | 1.7600 | 0.0965 | 0.1559 | 0.1024 | 0.0965 |
| [threshold-directopt-retime-v3-select-65536-profiles.json](threshold-directopt-retime-v3-select-65536-profiles.json) | 65536 | 4.7817 | 0.4238 | 0.7572 | 0.5494 | 0.2974 |
| [threshold-directopt-rows2w4-v3-select-32768-profiles.json](threshold-directopt-rows2w4-v3-select-32768-profiles.json) | 32768 | 1.7607 | 0.0964 | 0.1571 | 0.1023 | 0.0965 |
| [threshold-directopt-rows2w4-v3-select-65536-profiles.json](threshold-directopt-rows2w4-v3-select-65536-profiles.json) | 65536 | 4.8052 | 0.4242 | 0.7732 | 0.5478 | 0.2980 |
| [threshold-directopt-rows4w4-v3-select-32768-profiles.json](threshold-directopt-rows4w4-v3-select-32768-profiles.json) | 32768 | 1.7826 | 0.0965 | 0.1771 | 0.1028 | 0.0967 |
| [threshold-directopt-rows4w4-v3-select-65536-profiles.json](threshold-directopt-rows4w4-v3-select-65536-profiles.json) | 65536 | 4.8357 | 0.4237 | 0.8123 | 0.5475 | 0.2974 |
| [threshold-directopt-scratch16-v3-select-32768-profiles.json](threshold-directopt-scratch16-v3-select-32768-profiles.json) | 32768 | 1.7295 | 0.0952 | 0.1503 | 0.0900 | 0.0954 |
| [threshold-directopt-scratch16-v3-select-65536-profiles.json](threshold-directopt-scratch16-v3-select-65536-profiles.json) | 65536 | 4.7239 | 0.4517 | 0.7403 | 0.5152 | 0.2926 |
| [threshold-directopt-scratch8-v3-select-32768-profiles.json](threshold-directopt-scratch8-v3-select-32768-profiles.json) | 32768 | 1.7408 | 0.0958 | 0.1529 | 0.0939 | 0.0956 |
| [threshold-directopt-scratch8-v3-select-65536-profiles.json](threshold-directopt-scratch8-v3-select-65536-profiles.json) | 65536 | 4.7342 | 0.4248 | 0.7461 | 0.5197 | 0.2950 |
| [threshold-directopt-split-v3-select-32768-profiles.json](threshold-directopt-split-v3-select-32768-profiles.json) | 32768 | 1.7889 | 0.1261 | 0.1554 | 0.1025 | 0.0964 |
| [threshold-directopt-split-v3-select-65536-profiles.json](threshold-directopt-split-v3-select-65536-profiles.json) | 65536 | 4.9620 | 0.6064 | 0.7538 | 0.5481 | 0.2966 |
| [threshold-directopt-splitreduce-v3-select-32768-profiles.json](threshold-directopt-splitreduce-v3-select-32768-profiles.json) | 32768 | 1.7681 | 0.0964 | 0.1556 | 0.1022 | 0.1071 |
| [threshold-directopt-splitreduce-v3-select-65536-profiles.json](threshold-directopt-splitreduce-v3-select-65536-profiles.json) | 65536 | 4.7835 | 0.4228 | 0.7550 | 0.5460 | 0.3226 |
| [threshold-directopt-state64-v3-select-32768-profiles.json](threshold-directopt-state64-v3-select-32768-profiles.json) | 32768 | 1.7508 | 0.0966 | 0.1565 | 0.1026 | 0.0966 |
| [threshold-directopt-state64-v3-select-65536-profiles.json](threshold-directopt-state64-v3-select-65536-profiles.json) | 65536 | 4.7019 | 0.4233 | 0.7527 | 0.5504 | 0.2968 |
| [threshold-directopt-state6464-v3-select-32768-profiles.json](threshold-directopt-state6464-v3-select-32768-profiles.json) | 32768 | 1.7570 | 0.0966 | 0.1564 | 0.1027 | 0.0966 |
| [threshold-directopt-state6464-v3-select-65536-profiles.json](threshold-directopt-state6464-v3-select-65536-profiles.json) | 65536 | 4.7510 | 0.4246 | 0.7536 | 0.5507 | 0.2980 |
| [threshold-directopt-stateacc16-v3-select-32768-profiles.json](threshold-directopt-stateacc16-v3-select-32768-profiles.json) | 32768 | 1.6873 | 0.0952 | 0.1503 | 0.0901 | 0.0504 |
| [threshold-directopt-stateacc16-v3-select-65536-profiles.json](threshold-directopt-stateacc16-v3-select-65536-profiles.json) | 65536 | 4.6090 | 0.4516 | 0.7419 | 0.5168 | 0.1768 |
| [threshold-directopt-stateacc32-v3-select-32768-profiles.json](threshold-directopt-stateacc32-v3-select-32768-profiles.json) | 32768 | 1.6904 | 0.0952 | 0.1504 | 0.0908 | 0.0505 |
| [threshold-directopt-stateacc32-v3-select-65536-profiles.json](threshold-directopt-stateacc32-v3-select-65536-profiles.json) | 65536 | 4.6303 | 0.4522 | 0.7420 | 0.5134 | 0.1773 |
| [threshold-directopt-statetiled-v3-select-32768-profiles.json](threshold-directopt-statetiled-v3-select-32768-profiles.json) | 32768 | 1.7499 | 0.0965 | 0.1564 | 0.1026 | 0.0963 |
| [threshold-directopt-statetiled-v3-select-65536-profiles.json](threshold-directopt-statetiled-v3-select-65536-profiles.json) | 65536 | 4.6978 | 0.4238 | 0.7534 | 0.5518 | 0.2974 |
| [threshold-directopt-stream-v3-select-32768-profiles.json](threshold-directopt-stream-v3-select-32768-profiles.json) | 32768 | 1.8068 | 0.0615 | 0.2427 | 0.1019 | 0.0958 |
| [threshold-directopt-stream-v3-select-65536-profiles.json](threshold-directopt-stream-v3-select-65536-profiles.json) | 65536 | 4.7991 | 0.2404 | 0.9847 | 0.5451 | 0.2938 |
| [threshold-directopt-stream128m-v3-select-32768-profiles.json](threshold-directopt-stream128m-v3-select-32768-profiles.json) | 32768 | 1.7692 | 0.0558 | 0.2093 | 0.1017 | 0.0959 |
| [threshold-directopt-stream128m-v3-select-65536-profiles.json](threshold-directopt-stream128m-v3-select-65536-profiles.json) | 65536 | 4.7188 | 0.2434 | 0.8865 | 0.5494 | 0.2963 |
| [threshold-directopt-stream128s1-v3-select-32768-profiles.json](threshold-directopt-stream128s1-v3-select-32768-profiles.json) | 32768 | 1.7773 | 0.0559 | 0.2095 | 0.1026 | 0.0966 |
| [threshold-directopt-stream128s1-v3-select-65536-profiles.json](threshold-directopt-stream128s1-v3-select-65536-profiles.json) | 65536 | 4.6580 | 0.2190 | 0.8477 | 0.5459 | 0.2956 |
| [threshold-directopt-stream32-v3-select-32768-profiles.json](threshold-directopt-stream32-v3-select-32768-profiles.json) | 32768 | 1.7833 | 0.0736 | 0.2041 | 0.1023 | 0.0964 |
| [threshold-directopt-stream32-v3-select-65536-profiles.json](threshold-directopt-stream32-v3-select-65536-profiles.json) | 65536 | 4.7760 | 0.3122 | 0.8636 | 0.5482 | 0.2963 |
| [threshold-directopt-stream64-v3-select-32768-profiles.json](threshold-directopt-stream64-v3-select-32768-profiles.json) | 32768 | 1.7537 | 0.0633 | 0.1864 | 0.1021 | 0.0960 |
| [threshold-directopt-stream64-v3-select-65536-profiles.json](threshold-directopt-stream64-v3-select-65536-profiles.json) | 65536 | 4.6131 | 0.2784 | 0.7422 | 0.5517 | 0.2962 |
| [threshold-directopt-stream64s1-v3-select-32768-profiles.json](threshold-directopt-stream64s1-v3-select-32768-profiles.json) | 32768 | 1.7344 | 0.0619 | 0.1657 | 0.1023 | 0.0965 |
| [threshold-directopt-stream64s1-v3-select-65536-profiles.json](threshold-directopt-stream64s1-v3-select-65536-profiles.json) | 65536 | 4.6427 | 0.2719 | 0.7652 | 0.5512 | 0.2967 |
| [threshold-directopt-stream8w-v3-select-32768-profiles.json](threshold-directopt-stream8w-v3-select-32768-profiles.json) | 32768 | 1.8559 | 0.0639 | 0.2940 | 0.1015 | 0.0954 |
| [threshold-directopt-stream8w-v3-select-65536-profiles.json](threshold-directopt-stream8w-v3-select-65536-profiles.json) | 65536 | 5.1794 | 0.2877 | 1.3129 | 0.5468 | 0.2941 |
| [threshold-directopt-streamdense-v3-select-32768-profiles.json](threshold-directopt-streamdense-v3-select-32768-profiles.json) | 32768 | 1.8119 | 0.0618 | 0.2449 | 0.1019 | 0.0960 |
| [threshold-directopt-streamdense-v3-select-65536-profiles.json](threshold-directopt-streamdense-v3-select-65536-profiles.json) | 65536 | 4.9077 | 0.2412 | 1.0861 | 0.5454 | 0.2948 |
| [threshold-directopt-streamm128w8-v3-select-32768-profiles.json](threshold-directopt-streamm128w8-v3-select-32768-profiles.json) | 32768 | 1.7659 | 0.0637 | 0.1994 | 0.1021 | 0.0962 |
| [threshold-directopt-streamm128w8-v3-select-65536-profiles.json](threshold-directopt-streamm128w8-v3-select-65536-profiles.json) | 65536 | 4.6514 | 0.2672 | 0.8061 | 0.5519 | 0.2966 |
| [threshold-directopt-streamn32-v3-select-32768-profiles.json](threshold-directopt-streamn32-v3-select-32768-profiles.json) | 32768 | 1.7338 | 0.0607 | 0.1619 | 0.1027 | 0.0968 |
| [threshold-directopt-streamn32-v3-select-65536-profiles.json](threshold-directopt-streamn32-v3-select-65536-profiles.json) | 65536 | 4.5741 | 0.2432 | 0.7408 | 0.5475 | 0.2960 |
| [threshold-directopt-streamsplit-v3-select-32768-profiles.json](threshold-directopt-streamsplit-v3-select-32768-profiles.json) | 32768 | 1.7500 | 0.0615 | 0.1839 | 0.1021 | 0.0962 |
| [threshold-directopt-streamsplit-v3-select-65536-profiles.json](threshold-directopt-streamsplit-v3-select-65536-profiles.json) | 65536 | 4.6858 | 0.2704 | 0.8231 | 0.5481 | 0.2962 |
| [threshold-directopt-streamwide-v3-select-32768-profiles.json](threshold-directopt-streamwide-v3-select-32768-profiles.json) | 32768 | 1.7281 | 0.0598 | 0.1825 | 0.0902 | 0.0956 |
| [threshold-directopt-streamwide-v3-select-65536-profiles.json](threshold-directopt-streamwide-v3-select-65536-profiles.json) | 65536 | 4.6847 | 0.2628 | 0.7264 | 0.5171 | 0.2940 |
| [threshold-directopt-tc16-v3-select-32768-profiles.json](threshold-directopt-tc16-v3-select-32768-profiles.json) | 32768 | 1.6876 | 0.0957 | 0.1487 | 0.0906 | 0.0472 |
| [threshold-directopt-tc16-v3-select-65536-profiles.json](threshold-directopt-tc16-v3-select-65536-profiles.json) | 65536 | 4.6374 | 0.4518 | 0.7273 | 0.5137 | 0.2111 |
| [threshold-directopt-tc3x4-v3-select-32768-profiles.json](threshold-directopt-tc3x4-v3-select-32768-profiles.json) | 32768 | 1.7208 | 0.0966 | 0.1566 | 0.1027 | 0.0516 |
| [threshold-directopt-tc3x4-v3-select-65536-profiles.json](threshold-directopt-tc3x4-v3-select-65536-profiles.json) | 65536 | 4.6374 | 0.4233 | 0.7541 | 0.5500 | 0.1816 |
| [threshold-directopt-tc3x8-v3-select-32768-profiles.json](threshold-directopt-tc3x8-v3-select-32768-profiles.json) | 32768 | 1.7285 | 0.0968 | 0.1565 | 0.1032 | 0.0583 |
| [threshold-directopt-tc3x8-v3-select-65536-profiles.json](threshold-directopt-tc3x8-v3-select-65536-profiles.json) | 65536 | 4.6807 | 0.4240 | 0.7533 | 0.5484 | 0.2032 |
| [threshold-directopt-tcreduce-v3-select-32768-profiles.json](threshold-directopt-tcreduce-v3-select-32768-profiles.json) | 32768 | 1.7248 | 0.0967 | 0.1566 | 0.1028 | 0.0542 |
| [threshold-directopt-tcreduce-v3-select-65536-profiles.json](threshold-directopt-tcreduce-v3-select-65536-profiles.json) | 65536 | 4.6702 | 0.4235 | 0.7527 | 0.5488 | 0.1924 |
| [threshold-directopt-words16-v3-select-32768-profiles.json](threshold-directopt-words16-v3-select-32768-profiles.json) | 32768 | 1.6956 | 0.0955 | 0.1552 | 0.0907 | 0.0506 |
| [threshold-directopt-words16-v3-select-65536-profiles.json](threshold-directopt-words16-v3-select-65536-profiles.json) | 65536 | 4.6131 | 0.4514 | 0.7378 | 0.5165 | 0.1769 |
| [threshold-directopt-words8-v3-select-32768-profiles.json](threshold-directopt-words8-v3-select-32768-profiles.json) | 32768 | 1.6849 | 0.0953 | 0.1482 | 0.0902 | 0.0505 |
| [threshold-directopt-words8-v3-select-65536-profiles.json](threshold-directopt-words8-v3-select-65536-profiles.json) | 65536 | 4.6290 | 0.4533 | 0.7296 | 0.5169 | 0.1778 |

</details>

4 failed and 6 cancelled trials are excluded from all accepted timing and quality results. Their logs and statuses remain in provenance and SUMMARY.json. The compact-output prototypes included compilation failures; quality jobs for the slower matched-subtraction prototype were cancelled after timing showed a regression.

Exact commands, immutable source hashes, source objects, raw outputs, and individual samples are retained in this directory and [SUMMARY.json](SUMMARY.json).
