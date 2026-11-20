# Scope of the BF16 prefill cache handoff bug

The bug requires an aligned, single-request continuation on the vLLM cross-layer BF16 path whose previous exact tail was produced by a generic prefill path. The generic path can keep that tail only in the recent K/V buffer. The cross-layer update subsequently reads from the chronological page archive. The verified 512-token case met these conditions. The repair copies the recent tail before the batched read.

| Paper results | Exposure found |
|---|---|
| K2 512-token prefill ablation | Confirmed in the original row. The repaired rerun replaced its loss, timing, and memory measurements. |
| Ordinary batch-one Qwen and K2 with a 16K prefill block | The initial 16K block takes the staged path and writes the complete BF16 archive, including its tail. The subsequent aligned continuations preserve it. Short prompts that fit in the first block have no handoff. The repaired K2 16K loss control repeated the old result. |
| Batch-one K2 region threshold, centroid, BSWA and page ablations | Fixed the first exact prefill block at 16K. BSWA block sizes in this table are decode settings. |
| K2 and Qwen batch-eight AMD serving and concurrent quality | The representative aligned and mixed-input probes below found no affected handoff. The older concurrent jobs lack per-step cache traces, so the probe covers the tested schedules. |
| Native dense, INT4 and H100 RetroInfer-harness panels | Native dense uses vLLM’s dense cache. INT4 stages the authoritative recent tail directly. The H100 comparison harness builds and updates its cache outside this vLLM handoff. |

We instrumented the original BF16 archive read before the repair, using the release two-tier source at 94ec960 and both frozen top-two RULER three-tier serving snapshots. Counts are layer-by-continuation checks. The aligned case uses eight 32K prompts. The mixed long-input case uses eight lengths from 12K through 32K.

| Model and mode | Aligned checks | Aligned mismatches | Mixed checks |
|---|---:|---:|---:|
| k2-two-tier | 512 | 0 | 0 |
| qwen-two-tier | 128 | 0 | 0 |
| k2-three-tier-bf16 | 512 | 0 | 0 |
| qwen-three-tier-bf16 | 128 | 0 | 0 |

The mixed long-input jobs completed without selecting the vulnerable aligned path. An initial stress case that included a 500-token prompt completed its aligned arm and then raised a separate one-token AITER assertion in the mixed arm. The mixed long-input checks are reported from the subsequent completed jobs.

[512 diagnosis](../k2-prefill-block-512-fixed-v1/CACHE_HANDOFF_AUDIT.md) · [512 corrected results](../k2-prefill-block-512-fixed-v1/REPORT.md) · [Aligned and initial mixed commands](jobs.json) · [Completed mixed commands](mixed-jobs.json) · [Records](REPORT.json)
