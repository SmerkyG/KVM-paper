# K2 dense baselines for the ablation figures

Native dense attention uses `ROCM_AITER_UNIFIED_ATTN`. Timings use K2 Horizon FP8, BF16 KV, batch one, TP1, and 16K scheduler chunks on MI325X. Both dense jobs ran the exact same 64K ProLong prompt as the LoD timing runs. Prefill measures time to first token. Decode covers 1,024 timed steps including any cache work. Greedy continuation tokens can differ.

The region threshold and BSWA dense runs match their original timing protocols, with one warmup and three or two measured repeats, respectively. Each dense run used a GPU from the corresponding LoD study. The LoD points retain their original absolute medians across the study GPUs. Control points use the median of the paired LoD control timings.

| Chart | Dense TTFT (s) | Dense decode (ms/token) | Dense RULER |
|---|---:|---:|---:|
| Region mass threshold | 17.175 | 41.477 | 93.23% |
| BSWA | 17.135 | 41.191 | 93.23% |

The RULER baseline reuses dense scores on the same 32 prompts for each of the 13 tasks. The new ProLong dense loss uses the same eight 64K documents as the block sweep.

Dense ProLong loss was 0.495866 nats/token. The matched 16K LoD control scored 0.495732, so the dotted line is +0.000134 nats/token on the delta axis.

[Commands](jobs.json) · [Measured results](REPORT.json) · [RULER controls](../quality32-final-v1/REPORT.md)
