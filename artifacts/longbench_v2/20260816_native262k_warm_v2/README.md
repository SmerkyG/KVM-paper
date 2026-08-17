# Qwen3.5-35B-A3B LongBench-v2: full attention vs LOD

Run date: 2026-08-16

Both modes evaluated the same 503 LongBench-v2 examples with eight one-GPU
servers, eight requests per server, length sorting within each shard, thinking
disabled, and guided answer-choice decoding. The server used the model's native
262,144-position limit. Raw prompts were capped at 262,016 tokens to reserve
space for chat-template tokens. Consequently, 403 examples were untruncated and
100 examples longer than the model's native context were truncated.

Before measurement, every server processed an unmeasured batch of eight
262,016-token requests and an unmeasured shortest-length batch. The evaluator's
`run_wall_seconds` begins after tokenization and both warm-ups.

| Metric | Full attention | LOD INT4 | LOD speedup |
|---|---:|---:|---:|
| Eight-GPU wall (slowest shard) | 1,354.50 s | 545.48 s | 2.48x |
| Aggregate GPU time | 9,118.30 s | 3,619.99 s | 2.52x |
| Accuracy, all 503 | 251/503 (49.90%) | 141/503 (28.03%) | — |
| Accuracy, untruncated 403 | 200/403 (49.63%) | 118/403 (29.28%) | — |
| Accuracy, truncated 100 | 51/100 (51.00%) | 23/100 (23.00%) | — |

The timed run is therefore 59.7% shorter in wall time, but the current LOD
configuration has a severe quality regression and is not an acceptable
replacement for full attention yet.

For the untruncated subset, reconstructing batch completion time from the
per-request records gives a conservative slowest-shard estimate of 656.13 s
for full attention and 305.42 s for LOD (2.15x). A boundary batch can contain
both untruncated and truncated requests, so this subset timing is less exact
than the complete-run timing above.

`comparison.json` contains the paired accuracy breakdown and prediction
agreement. `full/` and `lod/` contain the raw per-example records.
