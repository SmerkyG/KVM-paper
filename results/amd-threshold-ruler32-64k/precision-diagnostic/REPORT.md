# Precision diagnostic on changed Qwen examples

These 10 cases were selected because their scores changed in the main panel by diagnostic submission time. This is a selected diagnostic, not a quality estimate or a substitute for the 13-task follow-up. The same frozen inference source was run with only the exact-entry probability-residual correction enabled.

| Task | Example | Old c=16 | Optimized | Precision correction |
|---|---:|---:|---:|---:|
| niah_single_3 | 10 | 1.000 | 0.000 | 1.000 |
| niah_multikey_1 | 27 | 1.000 | 0.000 | 1.000 |
| niah_multikey_2 | 21 | 0.000 | 1.000 | 1.000 |
| niah_multivalue | 20 | 1.000 | 0.750 | 0.750 |
| ruler_fwe | 25 | 0.667 | 1.000 | 1.000 |
| ruler_qa_hotpot | 20 | 1.000 | 0.000 | 1.000 |
| ruler_qa_hotpot | 29 | 1.000 | 0.000 | 1.000 |
| ruler_qa_squad | 1 | 0.750 | 1.000 | 1.000 |
| ruler_qa_squad | 24 | 0.000 | 0.667 | 0.000 |
| ruler_qa_squad | 30 | 0.667 | 1.000 | 0.667 |

The correction restored four of the five losses included here, including the two near-copy retrieval errors and two HotpotQA errors. The missing multivalue answer was not restored. Some earlier gains also disappeared. This supports investigating probability-rounding effects but does not isolate every numerical difference or establish unchanged quality.

[Exact command and selected cohort](job.json), [raw predictions](qwen-precise.json), [complete 13-task follow-up](../precision-panel/REPORT.md).
