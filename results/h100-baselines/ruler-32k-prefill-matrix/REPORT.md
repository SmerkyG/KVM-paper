# H100 Llama 32K: matched prefill comparison

BF16 model and KV, batch one, 32 frozen examples per task. All four prefill modes use the same source snapshot and the same two-tier decode rule within each table. Dense prefill retains LoD cache construction; its first token is checked against the original dense model. The 16K exact prefill size is unchanged. The threshold methods are experimental.

First-response scores stop at the first end-of-turn or end-of-text token. An asterisk marks a task completed inside a job still running, pending the job’s successful completion. A dash means not yet complete.

## Region-threshold decode

| Task | Full prefill | Top-8 prefill | Region c=16 prefill | Entry-16 prefill |
|---|---:|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 96.88% | 100.00% | 100.00% | 100.00% |
| niah_multikey_1 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_2 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_3 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multivalue | 96.88% | 99.22% | 99.22% | 99.22% |
| niah_multiquery | 95.31% | 98.44% | 95.31% | 95.31% |
| vt | 100.00% | 100.00% | 100.00% | 100.00% |
| fwe | 88.54% | 88.54% | 88.54% | 88.54% |
| cwe | 26.88% | 29.38% | 30.00% | 31.25% |
| qa_1 | 81.25% | 68.75% | 81.25% | 75.00% |
| qa_2 | 50.00% | 46.88% | 50.00% | 50.00% |

dense unweighted 13-task average: 87.36%.
top8 unweighted 13-task average: 87.02%.
region unweighted 13-task average: 88.02%.
entry unweighted 13-task average: 87.64%.

## Entry-threshold decode

| Task | Full prefill | Top-8 prefill | Region c=16 prefill | Entry-16 prefill |
|---|---:|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 96.88% | 96.88% | 96.88% | 100.00% |
| niah_multikey_1 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_2 | 84.38% | 68.75% | 75.00% | 75.00% |
| niah_multikey_3 | 65.62% | 71.88% | 75.00% | 71.88% |
| niah_multivalue | 97.66% | 99.22% | 99.22% | 99.22% |
| niah_multiquery | 97.66% | 97.66% | 98.44% | 98.44% |
| vt | 100.00% | 98.75% | 98.75% | 100.00% |
| fwe | 90.62% | 89.58% | 90.62% | 90.62% |
| cwe | 35.31% | 24.69% | 32.19% | 37.19% |
| qa_1 | 78.12% | 65.62% | 75.00% | 68.75% |
| qa_2 | 50.00% | 46.88% | 50.00% | 50.00% |

dense unweighted 13-task average: 84.33%.
top8 unweighted 13-task average: 81.53%.
region unweighted 13-task average: 83.93%.
entry unweighted 13-task average: 83.93%.

All comparisons use matched inputs and first-response scoring. Complete-source digests and cluster-run commands are in the provenance directory. The dense-prefill diagnostic also constructs the LoD cache; it is not a prefill speed test.
