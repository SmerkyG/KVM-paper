# H100 two-tier average-mass threshold routing

Experimental opt-in batch-one Llama-3.1-8B-Instruct two-tier BF16 decode in the RetroInfer runtime. For each query head, region routing uses the normalized query and opens every region with estimated routing-proxy probability greater than 16 times the mean over nonempty remote regions. Regions larger than 1,024 entries remain summarized, as in the reference. Unopened regions remain summaries; selected entries, local and sink are exact. Selection has no fixed eight-region cap or top-k sort. It computes a reduction for the average-mass threshold, compacts candidates deterministically, and distributes exact pages across workers.

Both comparison paths use whole-model CUDA graphs, BF16 model and cache, the same teacher-forced ProLong prompts, batch one, and one H100 80 GB HBM3. Each repetition prefills, captures the graph, runs 512 untimed decode steps, and measures the next 1,024 steps; context grows through the window. Cache catch-up and any periodic updates during the timed window are included. Initial allocation, prefill and graph capture are excluded from decode timing. One warmup repetition precedes two timed repetitions; values below are their medians.

| Initial context | Fixed eight ms/token | Threshold c=16 ms/token | Decode latency change | Fixed-eight prefill | Threshold prefill |
|---:|---:|---:|---:|---:|---:|
| 32K | 8.332 | 7.837 | -5.94% | 1.653 s | 1.665 s |
| 64K | 8.714 | 8.350 | -4.19% | 4.196 s | 4.200 s |

32K timed repetitions: fixed eight [8.3278, 8.3354] ms/token; threshold [7.8361, 7.8369] ms/token. Prompt-token SHA-256: `120725db8565d8c9016bde24247bf17cb7ae51b3ce4acca38a8f54e710e2ce68`.

64K timed repetitions: fixed eight [8.7134, 8.7154] ms/token; threshold [8.351, 8.3483] ms/token. Prompt-token SHA-256: `8160ddf14d64449dfbbf0c4462cfe49a78ced64189d58c8cd602d141b3bc0d2f`.

## Correctness and limitations

An isolated graph-captured probe on one real 32K layer measured fixed eight at 47.60 µs and threshold at 48.65 µs per call, with the same local-length rollback in both. The isolated probe uses one fixed query and omits the model and cache maintenance; it is not a full-model speed comparison.
The CUDA-graph implementation scored 32/32 on the frozen 32K RULER multi-key-3 cohort and reproduced every generated token from the FP32 reference using the same prompts and two-tier sparse prefill. This is one RULER task, not a full-panel quality result. In the reference, these samples averaged 16.52 open regions and 617.53 exact remote entries per head; those counts are not measurements of the timing prompts.
On one real 32K prefill/decode step, all 32 layers matched the FP32 reference region selections; maximum attention component difference was 0.003906. The independent synthetic frontier suite included an oversized region, empty selection, and more than eight open regions; maximum output component difference was 0.000482.
The 64K graph/eager comparison checked 25 sampled steps, including steps around cache catch-up boundaries; maximum logit difference was 0.000000.
The H100 path uses 64 exact-attention worker splits by default. It is experimental and limited to batch-one, two-tier BF16 Llama GQA geometry. It is enabled only in the RetroInfer adapter by `LOD_H100_THRESHOLD_MULTIPLIER=16`; it is not a serving default and does not implement three-tier or INT4. The timing comparison is not quality-matched: fixed eight and threshold routing open different regions and numbers of KV entries.

[32K fixed eight](h100-threshold-top8-32k-final.json), [32K threshold](h100-threshold-c16-64split-32k.json), [64K fixed eight](h100-threshold-top8-64k-final.json), [64K threshold](h100-threshold-c16-64split-64k.json), [GPU quality](h100-threshold-64split-ruler-quality.json), [real-query check](h100-threshold-64split-final-real.json), [synthetic edge checks](h100-threshold-64split-final-synthetic.json), [64K graph check](h100-threshold-64split-64k-graph-check.json).

[Experimental kernel](../../../benchmarks/h100_threshold_decode.py), [RetroInfer adapter](../../../benchmarks/retro_lod_adapter.py), [32K baseline run record](1794143415918660890-threshold-top8-32k-final-run.json), [32K threshold run record](1794143866230651252-threshold-c16-64split-32k-run.json), [64K baseline run record](1794143422828533777-threshold-top8-64k-final-run.json), [64K threshold run record](1794143873354370133-threshold-c16-64split-64k-run.json), [64K graph check record](1794143881369314419-threshold-64split-64k-graph-check-run.json), and [quality run record](1794143858636736956-threshold-64split-ruler-quality-run.json).
