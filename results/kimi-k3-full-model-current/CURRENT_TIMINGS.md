# Current K3 matched speed sweep

Full trained K3; TP8/DCP8/EP8, eight MI325X GPUs, frozen real ProLong prompts and identical teacher-forced continuations. Both arms use the approved G8 direct-state-I/O KDA prefill baseline and the same packed INT4 MoE weights. Attention caches remain BF16. Dense decode uses the improved Gluon kernel. LoD B1 uses DCP8; B8 uses one request's attention per GPU.

One exact-shape untimed warmup and one measured pass, 1,026 output tokens / 1,025 decode steps, four global-256 catch-ups per LoD request/layer. No prefix hits, preemptions, profiling events, or warmup times in these cells. A dash means no completed fresh measurement—not a reused older result.

## Prefill (seconds per batch)

| Batch | Context | Dense | Two-tier LoD | Dense / LoD |
|:--|--:|--:|--:|--:|
| B1 | 16K | 2.005 | 2.016 | 0.995× |
| B1 | 32K | 4.108 | 4.140 | 0.992× |
| B1 | 64K | 8.641 | 8.402 | 1.029× |
| B1 | 128K | 19.009 | 17.184 | 1.106× |
| B1 | 256K | 44.929 | 36.038 | 1.247× |
| B1 | 512K | 117.511 | — | — |
| B1 | 1020K | — | — | — |
| B8 | 16K | — | — | — |
| B8 | 32K | — | — | — |
| B8 | 64K | — | — | — |
| B8 | 128K | — | — | — |
| B8 | 256K | — | — | — |
| B8 | 512K | — | — | — |
| B8 | 1020K | — | — | — |

## Decode (milliseconds per batch step)

| Batch | Context | Dense | Two-tier LoD | Dense / LoD |
|:--|--:|--:|--:|--:|
| B1 | 16K | 22.238 | 22.237 | 1.000× |
| B1 | 32K | 22.305 | 22.240 | 1.003× |
| B1 | 64K | 22.634 | 22.243 | 1.018× |
| B1 | 128K | 22.998 | 22.302 | 1.031× |
| B1 | 256K | 23.849 | 22.370 | 1.066× |
| B1 | 512K | 25.352 | — | — |
| B1 | 1020K | — | — | — |
| B8 | 16K | — | — | — |
| B8 | 32K | — | — | — |
| B8 | 64K | — | — | — |
| B8 | 128K | — | — | — |
| B8 | 256K | — | — | — |
| B8 | 512K | — | — | — |
| B8 | 1020K | — | — | — |

## Sweep status

- `oct7-current-full-b1-power2.json`: in_progress; {'length': 1044480, 'phase': 'measurement', 'timestamp_unix': 1791342615.3977683, 'repeat_index': 0, 'repeats': 1}
- `oct7-current-lod-b1-long.json`: in_progress; {'length': 524288, 'phase': 'warmup', 'timestamp_unix': 1791342702.9517884}
- `oct7-current-lod-b1-short.json`: complete; {'length': 262144, 'phase': 'point_complete', 'timestamp_unix': 1791342608.5192482}

## Raw sources

- [oct7-current-full-b1-power2.json](oct7-current-full-b1-power2.json)
- [oct7-current-lod-b1-long.json](oct7-current-lod-b1-long.json)
- [oct7-current-lod-b1-short.json](oct7-current-lod-b1-short.json)
