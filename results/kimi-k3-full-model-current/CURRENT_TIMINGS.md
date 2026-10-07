# Current K3 matched speed sweep

These measurements use the current [live-context dense splits and IPC-safe graph allocator](LIVE_SPLITS_ALLOCATOR.md). Eager caches retain expandable allocation; graph communication uses registered ordinary allocations. The [pre-fix sweep](OCT7_PRE_FIX_TIMINGS.md) is archived and is not mixed into these cells.

Full trained K3; TP8/DCP8/EP8, eight MI325X GPUs, frozen real ProLong prompts and identical teacher-forced continuations. Both arms use the approved G8 direct-state-I/O KDA prefill baseline and the same packed INT4 MoE weights. Attention caches remain BF16. Dense decode uses the improved Gluon kernel. LoD B1 uses DCP8; B8 uses one request's attention per GPU.

One exact-shape untimed warmup and one measured pass, 1,026 output tokens / 1,025 decode steps, four global-256 catch-ups per LoD request/layer. No prefix hits, preemptions, profiling events, or warmup times in these cells. A dash means no completed fresh measurement—not a reused older result.

B1 long-context LoD retains the exact token-sharded archive; its validated 16K allocator control also uses that storage layout. B8 512K uses the documented compact-directory/sharded-bank memory configuration. B8 1020K has not completed generation: previous capacity failures are not timings.

The first new B1 LoD 32K/64K passes overlapped CPU regression tests on the timing node. Their raw records remain intact, but only quiet replacement passes are used in the table.

## Prefill (seconds per batch)

| Batch | Context | Dense | Two-tier LoD | Dense / LoD |
|:--|--:|--:|--:|--:|
| B1 | 16K | 2.006 | 2.001 | 1.002× |
| B1 | 32K | 4.106 | 4.130 | 0.994× |
| B1 | 64K | 8.651 | 8.423 | 1.027× |
| B1 | 128K | 18.988 | 17.175 | 1.106× |
| B1 | 256K | 44.885 | 35.275 | 1.272× |
| B1 | 512K | 117.490 | 83.278 | 1.411× |
| B1 | 1020K | 341.966 | 178.188 | 1.919× |
| B8 | 16K | 16.104 | 16.226 | 0.992× |
| B8 | 32K | — | 33.856 | — |
| B8 | 64K | — | — | — |
| B8 | 128K | — | 150.572 | — |
| B8 | 256K | — | — | — |
| B8 | 512K | — | — | — |
| B8 | 1020K | — | — | — |

## Decode (milliseconds per batch step)

| Batch | Context | Dense | Two-tier LoD | Dense / LoD |
|:--|--:|--:|--:|--:|
| B1 | 16K | 21.311 | 21.646 | 0.985× |
| B1 | 32K | 21.459 | 21.644 | 0.991× |
| B1 | 64K | 21.721 | 21.647 | 1.003× |
| B1 | 128K | 22.227 | 21.689 | 1.025× |
| B1 | 256K | 23.380 | 21.756 | 1.075× |
| B1 | 512K | 24.692 | 21.827 | 1.131× |
| B1 | 1020K | 27.388 | 21.986 | 1.246× |
| B8 | 16K | 31.743 | 30.134 | 1.053× |
| B8 | 32K | — | 30.434 | — |
| B8 | 64K | — | — | — |
| B8 | 128K | — | 31.339 | — |
| B8 | 256K | — | — | — |
| B8 | 512K | — | — | — |
| B8 | 1020K | — | — | — |

## Sweep status

- `oct7-fixed-full-b1-long.json`: complete; {'length': 1044480, 'phase': 'point_complete', 'timestamp_unix': 1791384121.719901}
- `oct7-fixed-full-b1-short.json`: complete; {'length': 262144, 'phase': 'point_complete', 'timestamp_unix': 1791383002.2503943}
- `oct7-fixed-lod-b1-long.json`: complete; {'length': 1044480, 'phase': 'point_complete', 'timestamp_unix': 1791383568.2486668}
- `oct7-fixed-lod-b1-short-remaining-64k-retry2.json`: complete; {'length': 65536, 'phase': 'point_complete', 'timestamp_unix': 1791383882.3625453}
- `oct7-fixed-lod-b1-short.json`: complete; {'length': 262144, 'phase': 'point_complete', 'timestamp_unix': 1791383074.202787}
- `oct7-fixed-lod-b8-short.json`: in_progress; {'length': 65536, 'phase': 'warmup', 'timestamp_unix': 1791384107.7645981}
- `oct7-graph-allocator-lod-b1-long-fit.json`: complete; {'length': 524288, 'phase': 'point_complete', 'timestamp_unix': 1791381975.557606}
- `oct7-graph-allocator-lod-b8.json`: complete; {'length': 131072, 'phase': 'point_complete', 'timestamp_unix': 1791381951.7831194}
- `oct7-live-splits-floor-full-b1.json`: complete; {'length': 131072, 'phase': 'point_complete', 'timestamp_unix': 1791381562.9180846}
- `oct7-live-splits-full-b8-small-reservation.json`: complete; {'length': 16384, 'phase': 'point_complete', 'timestamp_unix': 1791382181.4139628}

## Raw sources

- [oct7-fixed-full-b1-long.json](oct7-fixed-full-b1-long.json)
- [oct7-fixed-full-b1-short.json](oct7-fixed-full-b1-short.json)
- [oct7-fixed-lod-b1-long.json](oct7-fixed-lod-b1-long.json)
- [oct7-fixed-lod-b1-short-remaining-64k-retry2.json](oct7-fixed-lod-b1-short-remaining-64k-retry2.json)
- [oct7-fixed-lod-b1-short.json](oct7-fixed-lod-b1-short.json)
- [oct7-fixed-lod-b8-short.json](oct7-fixed-lod-b8-short.json)
- [oct7-graph-allocator-lod-b1-long-fit.json](oct7-graph-allocator-lod-b1-long-fit.json)
- [oct7-graph-allocator-lod-b8.json](oct7-graph-allocator-lod-b8.json)
- [oct7-live-splits-floor-full-b1.json](oct7-live-splits-floor-full-b1.json)
- [oct7-live-splits-full-b8-small-reservation.json](oct7-live-splits-full-b8-small-reservation.json)
