# LOD D/GQA kernel diagnostic (batch 8, gfx942)

This diagnostic separates the production two-tier exact-leaf and fused decode
kernels from whole-model vLLM timing. Synthetic cases use identical posting
lists and top-eight routes while varying only head dimension and GQA geometry.

## Exact-leaf prefill kernel

The production kernel already has the proposed sparse structure: one program
owns an expert-major query block, gathers arbitrary leaf/page indices, and
loops over `tl.dot([M,D], [D,N])` tiles. `N=32` processes two independently
indexed 16-token pages per iteration; it does not require contiguous leaves.

| Geometry | M16/N32 kernel | M16/N16 kernel | Result |
|---|---:|---:|---:|
| D128, KV4, G6 | 0.306 ms | 0.428 ms | N16 is 40% slower |
| D512, KV2, G8 | 0.985 ms | 1.669 ms | N16 is 69% slower |

N16 repeats query/program setup and gives the MFMA/PV path less work per
gather. It therefore loses despite being the literal 16-column formulation.

For actual D128 geometries, M64/N64/4 warps improves the isolated production
path versus the legacy M16/N32/2-warp default:

| Geometry | Legacy wall time | Tuned wall time | Improvement |
|---|---:|---:|---:|
| Muse (D128, KV2, G16) | 0.902 ms | 0.782 ms | 15.3% |
| OLMo (D128, KV8, G5) | 1.136 ms | 0.938 ms | 21.1% |
| Phi TP5 (D128, KV2, G4) | 0.497 ms | 0.462 ms | 7.5% |

For D512, M16/N32/2 warps remains the best prefill tile. Smaller M, N16,
larger tiles, output-accumulator splitting, QK D-axis splitting, and extra
`waves_per_eu` all regressed. The safe D512 optimization is instead in decode:
using scalar route accumulation rather than `tl.dot` improves the isolated
fused kernel from 0.304 to 0.278 ms (9.3%) with bit-identical BF16 output.

## Root cause of the cross-family whole-model slowdown

The CUSTOM LOD backend inherited generic `RocmAttentionBackend`, while full
attention used `ROCM_AITER_UNIFIED_ATTN`. Layers that are not LOD-eligible were
therefore delegated to generic Triton paged attention, not to the same AITER
path as the full-attention control:

- Muse: 52 attention layers, 13 LOD layers, 39 generic delegated layers.
- OLMo: 64 attention layers, 16 LOD layers, 48 generic delegated layers.
- Gemma: 30 attention layers, 5 LOD layers, 25 generic delegated layers.
- Qwen3.8: non-LOD layers are GDN and do not enter this attention backend.
- Phi: all 40 attention layers are LOD-eligible, so it did not have this
  delegated-layer amplification.

A GPU phase profile confirmed the mismatch: on the old Muse 8K run, measured
LOD sparse work was only about 455 ms in an 8.18 s prefill, and the fused LOD
decode kernel cost was far too small to explain the model-level decode gap.

The ROCm CUSTOM backend now inherits AITER unified attention and uses its
packed flash-style K/V cache layout. Eligible global layers still use LOD;
sliding-window or otherwise ineligible layers now use the same AITER family as
the full-attention baseline.

## End-to-end result after corrected native delegation

Cells compare the previous LOD result, corrected LOD, and the existing full
attention baseline. Prefill is aggregate prompt tokens/s; decode is ms per
batch-8 step.

| Model/context | Previous LOD prefill | Corrected LOD prefill | Full prefill | Previous LOD decode | Corrected LOD decode | Full decode |
|---|---:|---:|---:|---:|---:|---:|
| Muse 8K | 8,976 | 10,367 | 11,277 | 48.9 | 22.9 | 19.2 |
| Muse 16K | 6,533 | 10,132 | 10,955 | 79.5 | 23.7 | 18.7 |
| OLMo 8K | 6,733 | 7,881 | 9,129 | 64.4 | 36.4 | 25.9 |
| OLMo 16K | 4,585 | 7,287 | 8,828 | 104.1 | 38.6 | 26.7 |
| Gemma 8K | 22,826 | 34,344 | 36,041 | 38.1 | 9.75 | 8.6 |
| Gemma 16K | 16,703 | 31,029 | 28,686 | 68.0 | 10.3 | 8.7 |

Quality smoke tests remain correct: Muse is 8/8 at both 8K and 16K, and Gemma
is 8/8 at 16K. The synthetic pool parity test also passes. The D512 route
change improves Gemma 8K decode by about 1% end to end (9.86 to 9.75 ms); most
of the much larger family-level gain comes from fixing native delegation.

## Irregular GQA and remaining D128 costs

OLMo exposed a second, independent geometry bug. Its GQA factor is five, so
the coarse kernel's old power-of-two test forced one program per query head.
Padding the five rows to eight within each KV-head program, and masking the
three padding rows, reduces the representative B8/Q512/state2048 coarse call
from 15.087 to 2.858 ms (5.3x). Output differs by at most 3.05e-5 and LSE is
identical. The same GQA=5 and GQA=6 parity tests pass. At 16K, OLMo's aggregate
coarse phase falls from 2.355 s to 0.530 s and prefill rises from about 7.1K
to 8.0K tok/s.

The analogous padding change was also tried in the route-only top-k kernel,
but it did not improve whole-model timing and was removed. With no value
accumulator in that pass, the smaller head-major programs provide more useful
occupancy than one padded 64-row group.

The corrected phase profiles also disprove the hypothesis that D128 exact
leaf matmul alone explains the residual gap. Examples:

| Model/run | Route | Coarse | Exact leaf total | Actual exact kernel | State update |
|---|---:|---:|---:|---:|---:|
| Muse 16K | 512 ms | 257 ms | 274 ms | 124 ms | 109 ms |
| OLMo 16K, padded GQA | 575 ms | 530 ms | 460 ms | 218 ms | 330 ms |
| Phi 8K, TP5 | 227 ms | 364 ms | 206 ms | 57 ms | 154 ms |

For Phi, dispatch alone costs 133 ms versus 57 ms in the exact-leaf kernel.
Thus the remaining D128 penalty is distributed over routing, coarse
attention, state maintenance, and several dispatch/reduction launches. A
TP=1 Phi diagnostic was even worse relative to full attention, so its result
is not explained by TP=5 fragmentation.

Two plausible simplifications were rejected empirically:

- Computing BF16 state QK inside the fused route/coarse kernel avoids the
  routing-logit tensor but recomputes QK on the stable second scan. It reduces
  Muse 32K prefill from about 10.0K to 8.9K tok/s and OLMo from 7.6K to 5.5K.
- Increasing the internal prefill chunk/local window to 16K reduces Muse and
  OLMo throughput. Phi improves from 10.3K to 11.4K with an 8K window, but
  remains far below its 21.3K full-attention control.
- A 64-column D128 decode tile wins on the synthetic posting-list benchmark
  but loses at 64K end to end (OLMo 52.47 to 56.10 ms; Muse 28.37 to 28.62
  ms), where the longest real list determines tail work. The retained rule is
  N32.

## 32K/64K crossover

These are the corrected batch-8 measurements with a 16K vLLM scheduler
budget. Prefill cells are aggregate prompt tok/s; decode cells are ms per
batch step. The retained LOD timings use the three-repeat long runs and the
final D128 N32 decode rule.

| Model | 32K prefill Full / LOD | 64K prefill Full / LOD | 32K decode Full / LOD | 64K decode Full / LOD |
|---|---:|---:|---:|---:|
| Gemma-4-26B-A4B | 20,453 / 28,994 | 13,091 / 26,251 | 10.27 / 11.55 | 11.69 / 13.41 |
| Qwen3.8-27B-FP8 | 6,298 / 7,184 | 4,690 / 6,487 | 44.54 / 41.20 | 52.02 / 45.88 |
| Muse-Glimmer-30B | 10,824 / 9,983 | 10,013 / 9,479 | 18.94 / 26.09 | 19.18 / 28.37 |
| OLMo-3-1125-32B | 8,348 / 7,617 | 7,642 / 7,165 | 27.91 / 44.51 | 30.36 / 52.47 |
| Phi-4 TP5 | 21,331 / 11,112 | 18,574 / 10,017 | 8.51 / 14.77 | 9.76 / 16.30 |

Qwen is faster in both phases at both lengths. Gemma now has the strongest
prefill crossover (1.42x at 32K and 2.01x at 64K), but its decode remains
12-15% slower. Muse and OLMo approach parity in prefill at 64K (0.95x and
0.94x) but do not cross; their decode remains materially slower. Phi does not
approach crossover by 64K.

## Files

- `diagnostic.json`, `tune128.json`, and `tune512*.json`: prefill kernel data.
- `decode_diagnostic.json` and `decode_tune*.json`: fused decode data.
- `muse_phase_8k.json`: old-path GPU phase profile.
- `*_aiter_delegate_*.json`: corrected end-to-end vLLM measurements.
- `coarse_gqa_pad.json` and `verify_irregular_gqa_pad.json`: irregular-GQA
  timing and numerical parity.
- `*_d128long32_final*.json`: consistent 32K/64K D128 measurements.
