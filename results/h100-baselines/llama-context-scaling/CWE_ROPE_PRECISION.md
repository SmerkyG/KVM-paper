# CWE context length and rotary table precision

## Where the score changes

| Context | Prompts | Dense CWE | Squeezed CWE | Zero-hit prompts, dense / Squeezed |
|---|---:|---:|---:|---:|
| 16K | 32 | 93.44% | 94.06% | 0 / 0 |
| 32K original | 32 | 37.50% | 54.69% | 15 / 9 |
| 32K held out | 128 | 43.67% | 56.09% | 47 / 32 |
| 64K | 32 | 3.75% | 4.69% | 28 / 28 |

On the held-out 32K cohort, 132 of Squeezed's 159 extra target-word hits come from the 47 prompts where dense scores zero. Squeezed recovers at least one target word on 17 of those prompts. Dense has at least five words absent from the input lists in 54 numbered responses, compared with 35 for Squeezed. This diagnostic counts responses with at least eight parseable numbered entries.

Median list sizes are 3,160 words at 16K, 6,160 at 32K, and 12,220 at 64K. The median repetition count for each target word is 30 in all three splits. The splits use distinct word lists and distractors. [16K](CWE_16K_32.md) · [32K held out](CWE_32K_128.md) · [64K](DENSE_PREFILL.md#64k-ruler).

## Dense RoPE table precision at 32K

We replayed all 32 original CWE prompts on Llama-3.1-8B-Instruct, with dense causal prefill and decode. The native runtime stores inverse frequencies and sine and cosine tables in float32. The second arm computes the same Llama 3.1 inverse frequencies, phases and trigonometric values in float64, then stores the results in the float32 table consumed by the same rotary kernel. Model weights and KV remain BF16 in both arms. The maximum absolute sine or cosine difference through the first 32,768 positions is 0.002334. [Llama 3.1 frequency formula](https://raw.githubusercontent.com/huggingface/transformers/v4.51.3/src/transformers/modeling_rope_utils.py).

| Table construction | Mean CWE score | Zero-hit prompts |
|---|---:|---:|
| Native float32 | 37.50% | 15 / 32 |
| Float64, stored as float32 | 37.19% | 15 / 32 |

Float64 construction changes the score by -0.31 percentage points (paired 95% bootstrap interval -2.19 to +1.25, 100,000 resamples). The very same 15 prompts score zero in both arms. First tokens match on 32 of 32 prompts and first-response token sequences match on 16. This isolates precision in building the rotary table.

## Reproduction

- native, shard 0. [Results](llama32k-cwe-rope-native-shard0.json) · [Command](provenance/1794813282842566832-llama32k-cwe-rope-native-shard0/run.json) · [Log](provenance/1794813282842566832-llama32k-cwe-rope-native-shard0/output.log).
- native, shard 1. [Results](llama32k-cwe-rope-native-shard1.json) · [Command](provenance/1794813282835684673-llama32k-cwe-rope-native-shard1/run.json) · [Log](provenance/1794813282835684673-llama32k-cwe-rope-native-shard1/output.log).
- native, shard 2. [Results](llama32k-cwe-rope-native-shard2.json) · [Command](provenance/1794813282852797110-llama32k-cwe-rope-native-shard2/run.json) · [Log](provenance/1794813282852797110-llama32k-cwe-rope-native-shard2/output.log).
- native, shard 3. [Results](llama32k-cwe-rope-native-shard3.json) · [Command](provenance/1794813282847751291-llama32k-cwe-rope-native-shard3/run.json) · [Log](provenance/1794813282847751291-llama32k-cwe-rope-native-shard3/output.log).
- float64, shard 0. [Results](llama32k-cwe-rope-float64-shard0.json) · [Command](provenance/1794813316691982254-llama32k-cwe-rope-float64-shard0/run.json) · [Log](provenance/1794813316691982254-llama32k-cwe-rope-float64-shard0/output.log).
- float64, shard 1. [Results](llama32k-cwe-rope-float64-shard1.json) · [Command](provenance/1794813317627075925-llama32k-cwe-rope-float64-shard1/run.json) · [Log](provenance/1794813317627075925-llama32k-cwe-rope-float64-shard1/output.log).
- float64, shard 2. [Results](llama32k-cwe-rope-float64-shard2.json) · [Command](provenance/1794813316227782913-llama32k-cwe-rope-float64-shard2/run.json) · [Log](provenance/1794813316227782913-llama32k-cwe-rope-float64-shard2/output.log).
- float64, shard 3. [Results](llama32k-cwe-rope-float64-shard3.json) · [Command](provenance/1794813317023330031-llama32k-cwe-rope-float64-shard3/run.json) · [Log](provenance/1794813317023330031-llama32k-cwe-rope-float64-shard3/output.log).
