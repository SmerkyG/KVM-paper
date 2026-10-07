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
| B1 | 512K | 117.511 | 83.521 | 1.407× |
| B1 | 1020K | 342.108 | 178.284 | 1.919× |
| B8 | 16K | 16.038 | 16.218 | 0.989× |
| B8 | 32K | 32.955 | 40.551 | 0.813× |
| B8 | 64K | 69.263 | — | — |
| B8 | 128K | 153.288 | — | — |
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
| B1 | 512K | 25.352 | 22.431 | 1.130× |
| B1 | 1020K | 28.049 | 22.611 | 1.240× |
| B8 | 16K | 32.948 | 31.426 | 1.048× |
| B8 | 32K | 33.975 | 31.182 | 1.090× |
| B8 | 64K | 35.588 | — | — |
| B8 | 128K | 38.618 | — | — |
| B8 | 256K | — | — | — |
| B8 | 512K | — | — | — |
| B8 | 1020K | — | — | — |

## Sweep status

- `oct7-current-full-b1-power2.json`: complete; {'length': 1044480, 'phase': 'point_complete', 'timestamp_unix': 1791342986.3664126}
- `oct7-current-full-b8-long.json`: in_progress; {'length': 262144, 'phase': 'warmup', 'timestamp_unix': 1791344013.7848468}
- `oct7-current-full-b8-short.json`: complete; {'length': 131072, 'phase': 'point_complete', 'timestamp_unix': 1791343919.5279937}
- `oct7-current-lod-b1-long.json`: complete; {'length': 1044480, 'phase': 'point_complete', 'timestamp_unix': 1791343395.1450899}
- `oct7-current-lod-b1-short.json`: complete; {'length': 262144, 'phase': 'point_complete', 'timestamp_unix': 1791342608.5192482}
- `oct7-current-lod-b8-long-512k.json`: failed; {'length': 524288, 'phase': 'warmup', 'timestamp_unix': 1791343656.9135764}
- `oct7-current-lod-b8-short-remaining-256k.json`: in_progress; {'length': 32768, 'phase': 'point_complete', 'timestamp_unix': 1791344153.751404}
- `oct7-current-lod-b8-short.json`: failed; {'length': 16384, 'phase': 'measurement', 'timestamp_unix': 1791343538.2167103, 'repeat_index': 0, 'repeats': 1}

## Raw sources

- [oct7-current-full-b1-power2.json](oct7-current-full-b1-power2.json)
- [oct7-current-full-b8-long.json](oct7-current-full-b8-long.json)
- [oct7-current-full-b8-short.json](oct7-current-full-b8-short.json)
- [oct7-current-lod-b1-long.json](oct7-current-lod-b1-long.json)
- [oct7-current-lod-b1-short.json](oct7-current-lod-b1-short.json)
- [oct7-current-lod-b8-long-512k.json](oct7-current-lod-b8-long-512k.json)
- [oct7-current-lod-b8-short-remaining-256k.json](oct7-current-lod-b8-short-remaining-256k.json)
- [oct7-current-lod-b8-short.json](oct7-current-lod-b8-short.json)
