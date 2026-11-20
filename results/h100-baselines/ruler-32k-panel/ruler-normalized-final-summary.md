# Normalized-query threshold routing: Llama 32K multi-key 3

Paired 32 frozen prompts with Llama-3.1-8B-Instruct, BF16 weights/KV, original LoD sparse prefill and FP32 PyTorch decode. The threshold opens a region when its score probability is greater than c times the average over nonempty remote regions. A normalized-query probability is a routing proxy; it is not the region’s native attention mass. Unopened regions remain summaries. Three-tier opens up to two native-query-scored pages per selected region and retains one residual summary for the other entries. Two-tier opens all entries in selected regions. The fixed-eight control selects eight regions using normalized-query scores. The 1,024-entry cap applies; local and sink are always exact. These are quality diagnostics, not optimized speed measurements.

| Method | First-response accuracy | Open regions / head | Exact remote entries / head | Exact pages / head | Selected routing-score probability |
|---|---:|---:|---:|---:|---:|
| Two-tier, normalized c=4 | 100.00% (32/32) | 137.11 | 3544.78 | — | 51.15% |
| Two-tier, normalized c=8 | 100.00% (32/32) | 50.72 | 1559.23 | — | 34.53% |
| Two-tier, normalized c=16 | 100.00% (32/32) | 16.52 | 617.53 | — | 21.48% |
| Two-tier, normalized c=24 | 87.50% (28/32) | 8.31 | 348.67 | — | 15.84% |
| Two-tier, normalized c=32 | 71.88% (23/32) | 5.06 | 230.17 | — | 12.60% |
| Three-tier, normalized c=8 | 93.75% (30/32) | 50.72 | 892.18 | 77.11 | 34.47% |
| Three-tier, normalized c=16 | 87.50% (28/32) | 16.48 | 305.17 | 25.81 | 21.31% |
| Three-tier, fixed 8 normalized | 53.12% (17/32) | 8.00 | 151.14 | 12.68 | 14.13% |
| Two-tier, native c=4 | 100.00% (32/32) | 120.66 | 2439.18 | — | 65.69% |
| Two-tier, native c=8 | 96.88% (31/32) | 53.96 | 1187.74 | — | 52.43% |
| Three-tier, native c=4 | 87.50% (28/32) | 120.86 | 1680.65 | 165.35 | 65.67% |

Two-tier, normalized c=4: failed rows []; 815 decode steps; open-region range 4–245; average oversized candidates skipped 0.000000.
Two-tier, normalized c=8: failed rows []; 815 decode steps; open-region range 0–114; average oversized candidates skipped 0.000000.
Two-tier, normalized c=16: failed rows []; 816 decode steps; open-region range 0–65; average oversized candidates skipped 0.000000.
Two-tier, normalized c=24: failed rows [126, 164, 302, 447]; 809 decode steps; open-region range 0–42; average oversized candidates skipped 0.000000.
Two-tier, normalized c=32: failed rows [78, 126, 156, 164, 193, 246, 302, 306, 447]; 822 decode steps; open-region range 0–33; average oversized candidates skipped 0.000000.
Three-tier, normalized c=8: failed rows [59, 316]; 816 decode steps; open-region range 0–115; average oversized candidates skipped 0.000000.
Three-tier, normalized c=16: failed rows [59, 316, 447, 455]; 816 decode steps; open-region range 0–65; average oversized candidates skipped 0.000000.
Three-tier, fixed 8 normalized: failed rows [59, 68, 78, 126, 139, 170, 193, 235, 236, 302, 306, 316, 436, 447, 455]; 812 decode steps; open-region range 8–8; average oversized candidates skipped 0.000000.
Two-tier, native c=4: failed rows []; 816 decode steps; open-region range 1–271; average oversized candidates skipped 0.000000.
Two-tier, native c=8: failed rows [68]; 814 decode steps; open-region range 0–140; average oversized candidates skipped 0.000000.
Three-tier, native c=4: failed rows [59, 202, 302, 316]; 816 decode steps; open-region range 1–271; average oversized candidates skipped 0.000000.

Mean counts weight each run by its own decode steps. All runs passed nine independent first-step selection/frontier checks. Three-tier additionally checked the selected pages and recomputed residuals from the unopened leaves.

[Per-layer and per-head statistics](ruler-normalized-final-heads.csv).

[Two-tier c=4 outputs](ruler-two-normalized-mean4-mk3.json), [c=8](ruler-two-normalized-mean8-mk3.json), [c=16](ruler-two-normalized-mean16-mk3.json), [c=24](ruler-two-normalized-mean24-mk3.json), [c=32](ruler-two-normalized-mean32-mk3.json), [three-tier c=8 outputs](ruler-three-normalized-mean8-mk3.json), and [three-tier c=16 outputs](ruler-three-normalized-mean16-after24-mk3.json). Each run has a corresponding `run.json` metadata file in this directory; [c=24 metadata](1794142241945432784-ruler-two-normalized-mean24-run.json) records its command and source snapshot.
