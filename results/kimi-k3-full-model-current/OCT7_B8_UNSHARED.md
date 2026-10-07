# Superseded B8 owner configuration

These are valid trained, synchronized B8 measurements of the six-head,
unshared-owner-prefill-construction configuration. The decode score-workspace
fix and fused union were active; all 1,025 graph replays and four global-256
updates passed. They are preserved, not relabeled as failed or silently
replaced by a minimum. The new shared-scratch/twelve-head canonical sweep
supersedes them in [CURRENT_TIMINGS.md](CURRENT_TIMINGS.md).

| Context | Prefill, seconds/batch | Decode, ms/step |
|--:|--:|--:|
| 16K | 16.218 | 31.426 |
| 32K | 40.551 | 31.182 |
| 64K | 94.297 | 31.507 |
| 128K | 206.788 | 31.962 |
| 256K | 481.708 | 32.553 |

The engine reserved 256K capacity and only 1 GiB native cache. It retained
per-layer prefill construction score fields and experienced significant memory
pressure. Rank 0's allocator counters incremented by 48 reclamations in
64 checks over the 64K point's warmup/measurement combined. This counter is
not an exclusive duration attribution.

Raw data: [oct7-superseded-lod-b8-short-g6-unshared.json](oct7-superseded-lod-b8-short-g6-unshared.json).
