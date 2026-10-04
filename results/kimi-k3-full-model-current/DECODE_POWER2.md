# Kimi K3 four-update decode panel

Full model, TP8/DCP8/EP8, real ProLong traces, seed 0, one full-shape warmup
and one measured pass. 1,026 output tokens produce 1,025 timed decode steps.
Each LoD layer/rank must record exactly four catch-ups per request.
Times are end-to-end ms per batched decode step, including state updates.

| Context | B1 dense | B1 LoD | Dense / LoD | B8 dense | B8 LoD | Dense / LoD |
|--:|--:|--:|--:|--:|--:|--:|
| 16K | — | 20.700 | — | — | — | — |
| 32K | — | 20.696 | — | — | — | — |
| 64K | — | 20.688 | — | — | — | — |
| 128K | — | 20.830 | — | — | — | — |
| 256K | — | — | — | — | — | — |
| 512K | 24.686 | — | — | — | — | — |
| 1020K | 27.378 | — | — | — | — | — |

A dash is not an estimate. B8 LoD at 256K+ and B1 LoD at 512K
previously failed warmup on VRAM. B1 dense 512K/1020K values reuse
the already validated 1,024-step controls from README.md; dense has
no LoD catch-ups to amortize. They are not paired with a LoD result.
A partial source contributes only its completed, individually validated
points; a later warmup failure does not create another timing point.

## Sources

- [oct4-lod-b1-decode-power2-four-updates.partial.json](oct4-lod-b1-decode-power2-four-updates.partial.json)
- [oct4-full-b1-512k1020k-decode1025.json](oct4-full-b1-512k1020k-decode1025.json)
