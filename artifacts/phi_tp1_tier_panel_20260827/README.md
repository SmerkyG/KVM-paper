# Phi-4 TP1 64K tier comparison

This is a matched batch-eight speed comparison of native full attention,
two-tier LOD, and recursive three-tier LOD on one MI300X. Every row uses eight
distinct real ProLong prompts of 65,472 tokens, chat formatting, BF16 LOD
state, 64 decode tokens, three measured repetitions after warmup, and
`max_num_batched_tokens=4096`.

The 4,096-token scheduler is required here: at TP1, Phi exposes all 40 query
heads and 10 KV heads to each LOD worker, and the current selector workspace
does not fit the 16,384-token aggregate used by the canonical multi-model
panel. The same 4,096-token aggregate is used for full attention, so the TP1
comparisons below are matched.

| implementation | prefill, all 8 prompts | full / implementation | decode, B8 step | full / implementation |
|---|---:|---:|---:|---:|
| native full attention | **56.095 s** | 1.000x | 14.796 ms | 1.000x |
| two-tier, prefill top-3 / decode top-8 | 228.145 s | 0.246x | **12.826 ms** | **1.154x** |
| three-tier, prefill top-3 / decode top-8 | 189.468 s | 0.296x | 14.179 ms | 1.043x |

Three-tier prefill is 16.95% faster than two-tier, but is still 3.38x slower
than native full attention. Two-tier decode is the winner at TP1, reducing
latency by 13.31% versus full attention; three-tier reduces it by 4.17%.

For context, the prior canonical TP5 records were 28.119 / 43.216 / 34.788
seconds for full / two-tier / three-tier prefill and 9.970 / 11.198 / 9.998
milliseconds for decode. Those records used a 16,384-token aggregate and are
therefore contextual, not scheduler-matched, comparisons. Moving from TP5 to
TP1 multiplies LOD prefill time by roughly the fivefold increase in per-rank
query heads (5.28x for two-tier and 5.45x for three-tier), whereas native full
prefill grows only 2.00x. The custom LOD prefill kernels and selector workspace
are consequently the TP1 bottleneck. Decode scales much better and benefits
from eliminating TP communication.

Phi-4 has a native 16,384-token position limit and no configured RoPE scaling.
At 64K, vLLM warns that positions can produce non-finite values. These results
are useful as kernel-speed measurements, but not as a Phi quality result; the
historical full-attention and LOD 64K NIAH-S3 controls were both 0/64. The run
also hardened grouped routing so non-finite scores cannot leak an integer
sentinel into leaf-page addressing.

Raw measurements:

- `phi_full_tp1_64k_b8_mbt4_daemon_r3.json`
- `phi_two_tp1_64k_b8_mbt4_fixed_daemon_r3.json`
- `phi_three_tp1_64k_b8_mbt4_daemon_r3.json`
