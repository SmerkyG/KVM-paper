# H100 dense and threshold prefill attention components

BF16 Llama-3.1-8B-Instruct, batch one, the same fixed prompt for each context length, 32 model layers, one warmup and two measured repetitions. GPU event totals exclude weight projections and FFNs. Dense measures the complete causal FlashAttention call for all prompt queries; the selected-entry kernel measures only opened remote KV entries for later queries. It excludes LoD’s exact first chunk, local and sink attention, routing, index construction, and merging. These components therefore do not compute the same amount of work.

| Context | Dense attention GPU time | LoD selected-entry GPU time | Selected / dense |
|---|---:|---:|---:|
| 32K | [0.872 s](prefill-attention-dense-32k-gpu-events.json) | [0.503 s](prefill-attention-entry16-32k-gpu-events.json) | 57.7% |
| 64K | [3.409 s](prefill-attention-dense-64k-gpu-events.json) | [1.829 s](prefill-attention-entry16-64k-gpu-events.json) | 53.6% |

The LoD diagnostic synchronizes between stages; its wall-clock component times include synchronization and launch overhead. They cannot be added directly to the GPU event times or used as end-to-end measurements. The selected kernel event is recorded around its attention launch, after index expansion. The uninstrumented [prefill comparison](../prefill-next/REPORT.md) records full-model latency separately.

32K synchronized LoD stages, summed across layers: route logits 0.150 s, select and coarse 0.467 s, compact 0.083 s, expand indices 0.233 s, exact leaves 0.531 s, merge 0.102 s.
64K synchronized LoD stages, summed across layers: route logits 0.630 s, select and coarse 2.001 s, compact 0.368 s, expand indices 1.094 s, exact leaves 1.911 s, merge 0.305 s.

The LoD setting uses entry-16, GPU-packed offsets, reused routing data, 128-region selection tiles and unchanged 16,384-token exact prefill chunks. The strict bound covers logical exact remote entries per query head, not the complete execution.
