# Llama-3.1-8B-Instruct: H100 measurements

**Validity warning:** These earlier LoD runs used a state-update kernel later
found capable of dropping value-summary updates. Their LoD quality and timing
results need to be rerun with the synchronization fix before use in comparisons.

At the 128K-window point, BF16 and INT4 deliver 1.39× and 1.43× native decode speed, respectively. Complete-request throughput improves by 1.27× and 1.23×. Prefill is slower at 8K, 32K, and 64K, and faster in the 128K-window case in this panel.

One NVIDIA H100 80GB HBM3, tensor parallelism one, BF16 model weights. Native vLLM uses FlashAttention-3. LoD uses eight routed regions and two pages per region during three-tier decode.

Each timing is the median of three measurements after two warmups, with 1,025 forced output tokens. Prompts, runtime packages, scheduling, context capacity, and memory target are checked for each comparison. The 128K-window case uses 128,000 prompt tokens within a 131,072-token model limit.

## Decode

Milliseconds per batch step, including periodic LoD catch-up. The batch-eight metric follows the existing ProLong cohort timing; requests may enter and leave decode at different times.

| Batch | Prompt tokens | Native | Three-tier BF16 | Three-tier INT4 | BF16 speedup | INT4 speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8,192 | 6.880 | 7.858 | 7.870 | 0.88× | 0.87× |
| 1 | 32,768 | 7.979 | 7.999 | 7.964 | 1.00× | 1.00× |
| 1 | 65,536 | 9.395 | 8.268 | 8.159 | 1.14× | 1.15× |
| 1 | 128,000 | 12.121 | 8.722 | 8.489 | 1.39× | 1.43× |
| 8 | 32,768 | 17.805 | 12.275 | 12.258 | 1.45× | 1.45× |

## Prefill

Seconds, measured separately from model startup.

| Batch | Prompt tokens | Native | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 1 | 8,192 | 0.211 | 0.259 | 0.362 |
| 1 | 32,768 | 1.211 | 1.497 | 1.688 |
| 1 | 65,536 | 3.552 | 4.003 | 4.334 |
| 1 | 128,000 | 11.011 | 9.596 | 10.303 |
| 8 | 32,768 | 10.363 | 13.373 | 14.161 |

## Complete request cohort

Seconds, measured separately from model startup.

| Batch | Prompt tokens | Native | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 1 | 8,192 | 7.257 | 8.309 | 8.423 |
| 1 | 32,768 | 9.384 | 9.699 | 9.843 |
| 1 | 65,536 | 13.178 | 12.467 | 12.693 |
| 1 | 128,000 | 23.431 | 18.517 | 18.995 |
| 8 | 32,768 | 28.606 | 25.944 | 26.719 |

Two-tier BF16 also completes the 32K batch-one panel: 1.377 s prefill and 8.346 ms per decode step.

## Quality and serving checks

These are retrieval sanity checks. They do not establish full RULER quality parity.

Validation passed 76 regression checks and 31 focused H100 checks. The [timing audit](../results/h100-llama31/audit.json) found no reported JIT compilation inside measured calls.

| LoD mode | Sequential 8K/32K/64K retrieval | Concurrent mixed-length requests |
|---|---:|---:|
| three-tier-bf16 | 9/9 | 8/8 |
| three-tier-int4 | 9/9 | 8/8 |

Native vLLM passes all nine sequential retrieval checks. The actual OpenAI HTTP server passes two short questions and a 19,557-token retrieval request in three-tier INT4.

Greedy output hashes vary across some repeated LoD speed runs. The report measures fixed-length performance; it does not claim identical generated continuations across runs or equivalence to dense attention.

## Reproduction and artifacts

Use [the H100 guide](H100_LLAMA.md) for setup and commands. Source snapshots, per-file hashes, commands, hardware identity, and logs are retained in [the provenance directory](../results/h100-llama31/provenance/). The complete installed environment is in [runtime-freeze.txt](../results/h100-llama31/runtime-freeze.txt).

- [native-speed-b1-cached](../results/h100-llama31/raw/native-speed-b1-cached.json)
- [h100-final-three-tier-bf16-b1](../results/h100-llama31/raw/h100-final-three-tier-bf16-b1.json)
- [h100-final-three-tier-int4-b1](../results/h100-llama31/raw/h100-final-three-tier-int4-b1.json)
- [llama-native-b8](../results/h100-llama31/raw/llama-native-b8.json)
- [h100-final-three-tier-bf16-b8](../results/h100-llama31/raw/h100-final-three-tier-bf16-b8.json)
- [h100-final-three-tier-int4-b8](../results/h100-llama31/raw/h100-final-three-tier-int4-b8.json)
- [h100-window-128k-full](../results/h100-llama31/raw/h100-window-128k-full.json)
- [h100-window-128k-three-tier-bf16](../results/h100-llama31/raw/h100-window-128k-three-tier-bf16.json)
- [h100-window-128k-three-tier-int4](../results/h100-llama31/raw/h100-window-128k-three-tier-int4.json)
- [h100-final-three-tier-bf16-retrieval-b1](../results/h100-llama31/raw/h100-final-three-tier-bf16-retrieval-b1.json)
- [h100-final-three-tier-bf16-retrieval-b8](../results/h100-llama31/raw/h100-final-three-tier-bf16-retrieval-b8.json)
- [h100-final-three-tier-int4-retrieval-b1](../results/h100-llama31/raw/h100-final-three-tier-int4-retrieval-b1.json)
- [h100-final-three-tier-int4-retrieval-b8](../results/h100-llama31/raw/h100-final-three-tier-int4-retrieval-b8.json)
- [native-retrieval](../results/h100-llama31/raw/native-retrieval.json)
- [final-http-smoke](../results/h100-llama31/raw/final-http-smoke.json)
- [h100-final-two-tier-b1](../results/h100-llama31/raw/h100-final-two-tier-b1.json)
