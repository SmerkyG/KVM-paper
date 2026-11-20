# H100 two-tier average-mass threshold routing

Experimental opt-in batch-one Llama-3.1-8B-Instruct two-tier BF16 decode in the RetroInfer runtime. For each query head, region routing uses the normalized query and opens every region with estimated routing-proxy probability greater than 16 times the mean over nonempty remote regions. Regions larger than 1,024 entries remain summarized, as in the reference. Unopened regions remain summaries; selected entries, local and sink are exact. Selection has no fixed eight-region cap or top-k sort. It computes a reduction for the average-mass threshold, compacts candidates deterministically, and distributes exact pages across workers.

Both comparison paths use whole-model CUDA graphs, BF16 model and cache, the same teacher-forced ProLong prompts, batch one, and one H100 80 GB HBM3. Each repetition prefills, captures the graph, runs 512 untimed decode steps, and measures the next 1,024 steps; context grows through the window. Cache catch-up and any periodic updates during the timed window are included. Initial allocation, prefill and graph capture are excluded from decode timing. One warmup repetition precedes two timed repetitions; values below are their medians.

| Initial context | Fixed eight ms/token | Threshold c=16 ms/token | Decode latency change | Fixed-eight prefill | Threshold prefill |
|---:|---:|---:|---:|---:|---:|
| 32K | 8.332 | 7.869 | -5.55% | 1.653 s | 1.662 s |
| 64K | 8.714 | 8.571 | -1.65% | 4.196 s | 4.204 s |

32K timed repetitions: fixed eight [8.3278, 8.3354] ms/token; threshold [7.8632, 7.8749] ms/token. Prompt-token SHA-256: `120725db8565d8c9016bde24247bf17cb7ae51b3ce4acca38a8f54e710e2ce68`.

64K timed repetitions: fixed eight [8.7134, 8.7154] ms/token; threshold [8.5716, 8.5697] ms/token. Prompt-token SHA-256: `8160ddf14d64449dfbbf0c4462cfe49a78ced64189d58c8cd602d141b3bc0d2f`.

## Correctness and limitations

The CUDA-graph implementation scored 32/32 on the frozen 32K RULER multi-key-3 cohort and reproduced every generated token from the FP32 reference using the same prompts and two-tier sparse prefill. This is one RULER task, not a full-panel quality result. In the reference, these samples averaged 16.52 open regions and 617.53 exact remote entries per head; those counts are not measurements of the timing prompts.
On one real 32K prefill/decode step, all 32 layers matched the FP32 reference region selections; maximum attention component difference was 0.003906. The independent synthetic frontier suite included an oversized region, empty selection, and more than eight open regions; maximum output component difference was 0.000482.
The 64K graph/eager comparison checked 25 sampled steps, including steps around cache catch-up boundaries; maximum logit difference was 0.000000.
The H100 path is experimental and limited to batch-one, two-tier BF16 Llama GQA geometry. It is enabled only in the RetroInfer adapter by `LOD_H100_THRESHOLD_MULTIPLIER=16`; it is not a serving default and does not implement three-tier or INT4. The timing comparison is not quality-matched: fixed eight and threshold routing open different regions and numbers of KV entries.

[32K fixed eight](h100-threshold-top8-32k-final.json), [32K threshold](h100-threshold-c16-32k-final.json), [64K fixed eight](h100-threshold-top8-64k-final.json), [64K threshold](h100-threshold-c16-64k-final.json), [GPU quality](h100-threshold-gqa-ruler-quality.json), [real-query check](h100-threshold-gqa-real-check.json), [synthetic edge checks](h100-threshold-gqa-edge-check.json), [64K graph check](h100-threshold-64k-maint-graph-check.json).

[Kernel and adapter](../../../benchmarks/h100_threshold_decode.py), [32K baseline run record](1794143415918660890-threshold-top8-32k-final-run.json), [32K threshold run record](1794143494773913040-threshold-c16-32k-final-run.json), [64K baseline run record](1794143422828533777-threshold-top8-64k-final-run.json), [64K threshold run record](1794143501903745394-threshold-c16-64k-final-run.json), [64K graph check record](1794143555765722388-threshold-64k-maint-graph-check-run.json), and [quality run record](1794143357718384477-threshold-gqa-ruler-quality-run.json).
