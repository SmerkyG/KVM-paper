# H100 Llama-3.1-8B-Instruct baseline comparison

**Validity warning:** Earlier LoD results linked from this report used a
state-update kernel later found capable of dropping value-summary updates.
Rerun those LoD measurements with the synchronization fix before comparison.

Work in progress. The benchmark uses the same pinned checkpoint and saved
ProLong prompt-token manifest as [the LoD panel](H100_LLAMA_RESULTS.md).
Batch size is one; the RetroInfer and QUEST panels generate 1,025 output
tokens with end-of-sequence ignored, using two warmups and three measured
iterations. Squeezed Attention's
released evaluation path is excluded from the performance comparison.
The later, matched whole-step-graphed comparison inside RetroInfer is in
[RETRO_LOD.md](RETRO_LOD.md); the RetroInfer rows below are the earlier
ungraphed author-runtime measurements.

| Method | Context | Prefill | Decode | State |
|---|---:|---:|---:|---|
| vLLM dense | 32,768 | 1.211 s | 7.979 ms/token | complete |
| LoD three-tier BF16 | 32,768 | 1.497 s | 7.999 ms/token | complete |
| LoD three-tier INT4 KV | 32,768 | 1.688 s | 7.964 ms/token | complete |
| QUEST-style vLLM BF16, 25% token budget | 32,768 | 1.216 s | 10.534 ms/token | preliminary; quality pending |
| QUEST-style vLLM BF16, full budget | 32,768 | 1.216 s | 18.770 ms/token | diagnostic |
| RetroInfer GPU-only | 32,768 | 1.960 s | 12.836 ms/token | complete |
| RetroInfer framework, full attention | 32,768 | 1.648 s | 9.593 ms/token | complete |
| vLLM dense | 65,536 | 3.552 s | 9.395 ms/token | complete |
| LoD three-tier BF16 | 65,536 | 4.003 s | 8.268 ms/token | complete |
| LoD three-tier INT4 KV | 65,536 | 4.334 s | 8.159 ms/token | complete |
| QUEST-style vLLM BF16, 25% token budget | 65,536 | 3.529 s | 13.321 ms/token | preliminary; quality pending |
| RetroInfer GPU-only | 65,536 | 5.912 s | 11.673 ms/token | complete |
| RetroInfer framework, full attention | 65,536 | 5.282 s | 11.024 ms/token | complete |
| QUEST FP16, 25% token budget | 32,768 | 1.932 s | 11.778 ms/token | complete |
| QUEST FP16, full budget | 32,768 | 1.963 s | 14.301 ms/token | complete |
| QUEST FP16, 25% token budget | 65,536 | 6.150 s | 14.998 ms/token | complete |
| QUEST FP16, full budget | 65,536 | 6.279 s | 20.155 ms/token | complete |

The completed RetroInfer results are slower than its own dense path at 32K
and 64K.
Its GPU-only implementation does not implement CUDA graph capture, even when
the configuration flag is set; vLLM uses graphs. Its cache allocation is
outside the measured prefill, and online index construction is included. The
QUEST CUDA build was also running on the host during the completed RetroInfer
measurements; a future repeat can assess CPU contention. The repeat was
deferred to prioritize the common-runtime comparison. These are speed results,
not quality comparisons.

## Interpreting the comparisons

The standalone frameworks have substantially different dense decode times.
Consequently their speedup ratios use different denominators and do not
establish a ranking of attention algorithms. The most comparable standalone
figures are each implementation's absolute latency and its own dense baseline,
with the framework and precision differences explicitly disclosed.

For a controlled algorithm comparison, integrate each available method into
the same vLLM model execution path, with the same model checkpoint, weight/KV
precision, prompts, generated-token count, batch size, context capacity, and
CUDA-graph policy. Each method may retain the storage format and metadata it
requires; charge its index creation, cache updates, selection and attention
kernels to that method. Include the native dense backend and each method's
full-budget mode, and report absolute latency as well as speedup against the
single native dense run. Profile attention, selection, cache maintenance and
other model work separately to explain the total, rather than subtracting
standalone framework timings or assuming all overhead is outside attention.
In addition to normal generation, use a shared forced token sequence for a
controlled replay so generated-token differences do not change the workload.
Monitor GPU clocks and report repeated-run variation.

Report prefill, decode, full-request time, peak GPU/CPU memory and quality at
the same setting. Include cold and reused-prefix cases where a method requires
offline indexing. For the current published QUEST kernel, an FP16 LoD and
dense control can provide a precision-matched comparison; a native BF16 Quest
port is needed for a BF16-controlled comparison. Squeezed Attention's public
evaluation path calls released Triton kernels, but contains substantial
per-token Python and PyTorch work; it does not by itself reproduce the paper's
optimized generation setup.

A benchmark-only QUEST-style BF16 reimplementation is being integrated with
vLLM's grouped KV cache. Its page-bound and attention outputs pass reference
checks in the Triton interpreter, including partial pages and noncontiguous
cache layout. GPU validation, including graph replay, and a vLLM smoke run
passed. The 32K and 64K timing runs have completed. These
preliminary timings do not establish an algorithm ranking without quality
checks and further optimization. A preliminary 32K passkey check passed all
three tested depths; it is not a broad quality evaluation. This is an independent
reimplementation and must be identified separately from the authors' released
kernels. The timing jobs use normal generation rather than fixed-token replay;
matching generated-token count alone does not prevent output divergence.
At 32K, the prototype's 25% budget appears 1.78× faster than its own
full-budget path, but is slower than the common native dense reference.
The full-budget run produced the same 1,025 generated token IDs as native
dense on the saved prompt; this is a useful sanity check, not a comprehensive
numerical or quality validation.

The released QUEST CUDA kernels dispatch FP16 only, so the initial QUEST
measurements use FP16 weights and KV, while LoD and RetroInfer use BF16
weights and BF16 KV (or INT4 LoD KV where stated). An experimental BF16
kernel build failed: the released radix selector and page-update kernels do
not compile for BF16. A BF16 result would require further kernel work. The
released QUEST fast path also required a Llama 3.1 rotary-encoding adapter and
expansion of grouped-query KV heads. The latter increases its cache traffic
and memory use; this is not an optimized native GQA implementation. The
rotary adapter passed the BF16 reference check through position 66,560, with
absolute tolerance 0.04. QUEST's first two layers remain dense, as in its code.

Squeezed Attention's released evaluation code includes and invokes Triton
centroid-lookup and sparse-attention kernels. Its per-token path also recomputes
static cluster counts with a Python loop, constructs masks and indices, and
copies or reshapes KV data. The released lookup is a two-pass, unsplit
implementation; the paper describes single-pass and split-KV generation
optimizations. The authors' README lists integration of efficient generation
kernels into the evaluation pipeline as future work. The 32K, 16-decode-step
pilot measured 1,359 ms/token in this evaluation path, with clustering timed
separately; it did not profile these individual costs. It should not be
presented as the speed of the paper's optimized system. A common-runtime port
could reuse the released kernels, move invariant preprocessing out of decode,
and address graph capture and GQA before a matched correctness and speed test.

Scripts: [comparison harness](h100_comparison.py), [Squeezed harness](h100_squeezed.py),
and [RoPE check](h100_comparison_rope_check.py). Source revisions, local
compatibility changes, dependency inventories, and completed JSON outputs
are in `results/h100-baselines/`. Runs are submitted through the H100
`cluster-run` adapter and logged under `~/.local/state/lod-cluster-run/` on
the H100 host.

## Double-P: illustrative estimate, not a measurement

The Double-P paper's Figure 8 reports end-to-end decoding latency relative
to its own dense baseline. At batch one, its plotted relative speedups for
RetroInfer and Double-P are approximately 1.04× and 1.28× at 32K, and 1.39×
and 1.62× at 64K. Applying the ratio of those ratios to our latest matched
GPU-only RetroInfer runs produces illustrative Double-P estimates of
10.68 ms/token at 32K (13.139 × 1.04 / 1.28) and 11.29 ms/token at 64K
(13.162 × 1.39 / 1.62). These are not measured Double-P results and should
not be mixed into a measured-result table or used for a definitive ranking.

The paper's plotted RetroInfer time is roughly 24 ms/token at both lengths,
substantially above our measured GPU-only timing. Its dense baseline is also much
slower than ours. The paper uses an H100 PCIe, Intel Xeon Gold 5416S, CUDA
12.8, PyTorch 2.8 and NIAH/RULER; our H100 reports 80 GB HBM3 and a 700 W
default power limit, with an Intel Xeon Platinum 8480C, CUDA 12.4, PyTorch
2.5 and ProLong prompts. The precise model checkpoint and all execution
details in the paper have not been verified. The underlying relative
speedups may not transfer between these systems, especially when their dense
and non-attention costs differ. Source: [Double-P, Figure 8](https://arxiv.org/pdf/2602.05191).
Earlier runs with RetroInfer's original Triton environment produced different
anchors (12.836 and 11.673 ms/token) and estimates (10.43 and 10.02 ms/token);
that variation reinforces that the projections are tentative.
The later whole-step-graphed RetroInfer results are 9.766 and 10.583 ms/token
at 32K and 64K. Applying the same paper ratios to those anchors yields
illustrative estimates of 7.94 and 9.08 ms/token respectively. The paper's
graph policy has not been confirmed, making the transferability of those
estimates particularly uncertain. The matched graphed measurements are
documented in [RETRO_LOD.md](RETRO_LOD.md).
