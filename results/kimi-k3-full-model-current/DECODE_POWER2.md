# Kimi K3 four-update decode panel

Full model, TP8/DCP8/EP8, real ProLong traces, seed 0, one full-shape warmup
and one measured pass. 1,026 output tokens produce 1,025 timed decode steps.
Each LoD layer/rank must record exactly four catch-ups per request.
Times are end-to-end ms per batched decode step, including state updates.

| Context | B1 dense | B1 LoD | Dense / LoD | B8 dense | B8 LoD | Dense / LoD |
|--:|--:|--:|--:|--:|--:|--:|
| 16K | — | — | — | — | — | — |
| 32K | — | — | — | — | — | — |
| 64K | — | — | — | — | — | — |
| 128K | — | — | — | — | — | — |
| 256K | — | — | — | — | — | — |
| 512K | — | — | — | — | — | — |

A dash is not an estimate. B8 LoD at 256K+ and B1 LoD at 512K
previously failed warmup on VRAM. B1 dense 512K/1020K controls with
1,024 timed steps remain documented separately in README.md.

## Sources

