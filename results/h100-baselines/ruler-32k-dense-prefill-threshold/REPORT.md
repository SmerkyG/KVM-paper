# H100 Llama 32K: dense vs. sparse prefill with threshold decode

BF16 weights/KV; batch one; 32 matched frozen examples per task. Both are two-tier LoD decode. Region uses c=16 times mean region mass; entry uses the strict 16√n per-head bound. Dense-prefill runs retain LoD cache construction and match the complete source snapshot of their respective sparse-prefill baseline. Each first token is checked against the dense model. This diagnostic is not a speed test.

An asterisk marks a complete 32-example task from a run still in progress; it remains provisional until that run succeeds. Remaining queued historical-snapshot controls were superseded by the complete same-revision prefill matrix.

| Task | Region sparse | Region dense | Change | Entry sparse | Entry dense | Change |
|---|---:|---:|---:|---:|---:|---:|
| niah_single_1 | — | — | — | — | — | — |
| niah_single_2 | — | — | — | — | — | — |
| niah_single_3 | — | — | — | — | — | — |
| niah_multikey_1 | — | — | — | — | — | — |
| niah_multikey_2 | 100.00% | 100.00% | +0.00 | 68.75% | 84.38% | +15.62 |
| niah_multikey_3 | 100.00% | 100.00% | +0.00 | 71.88% | 65.62% | -6.25 |
| niah_multivalue | — | — | — | — | — | — |
| niah_multiquery | — | — | — | — | — | — |
| vt | — | — | — | — | — | — |
| fwe | — | — | — | — | — | — |
| cwe | 29.69% | 26.88% | -2.81 | 25.00% | 35.31% | +10.31 |
| qa_1 | 68.75% | 81.25% | +12.50 | 65.62% | 78.12% | +12.50 |
| qa_2 | 46.88% | 50.00% | +3.12 | 46.88% | 50.00% | +3.12 |

First-response scoring stops at the first end-of-turn or end-of-text token. A 32-example task has limited precision. Original run commands, source hashes and logs are in the provenance directory.
