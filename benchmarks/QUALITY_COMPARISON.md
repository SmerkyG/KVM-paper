# Limited quality comparisons for H100 attention methods

The [ClusterKV comparison](../results/h100-baselines/clusterkv/REPORT.md)
uses the same frozen 416-prompt Llama-3.1-8B-Instruct cohort at 32K. See its
per-task first-response scores alongside dense, RetroInfer and two-tier LoD.

The [dense-prefill LoD run](../results/h100-baselines/clusterkv/REPORT.md#32k-ruler-quality)
scores **87.46%** with two-tier c=16 materialized threshold decode in BF16.
The matched sparse-prefill LoD run scores **86.96%**, GPU-only RetroInfer
with dense prefill scores **88.53%**, and dense BF16 scores **88.88%**.
The first generated token matches dense BF16 on all 416 prompts.

Status: **the 32K RULER panel is complete for dense and LoD BF16**, with 32
examples per task across all 13 tasks. See the
[paired 13-task results](../results/h100-baselines/ruler-32k-panel/ruler-32k-full13-summary.md).
The INT4 run was stopped to prioritize the full
13-task dense and BF16 comparison. The five-example
smoke runs for dense, LoD BF16 and LoD INT4 completed and passed one
graph-versus-eager decode check per task. Those smoke results are not a
quality estimate. The 32-per-task runs below cover all 13 RULER tasks but are
exploratory and do not evaluate the full dataset.

The H100 cluster-run IDs are
`1794083495924400286-ruler-32k-lod-bf16-full`,
`1794083501773105495-ruler-32k-dense-full`, and the stopped
`1794083521539844461-ruler-32k-lod-int4-full`. Their incremental JSON files
are under `/home/ubuntu/comparison/results/`, with matching names ending in
`.json`; check `stage` and recorded counts before interpreting them. The
[runner](retro_ruler_run.py) saves each prediction, token hash and score.
[The comparison script](retro_ruler_compare.py) refuses incomplete or
unpaired inputs before producing a summary. The former five-task summary job
was cancelled along with the INT4 run; its incomplete results must not be used
as a full INT4 score.
The tokenizer did not expose a revision in the result metadata; the H100's
offline Hugging Face cache `main` reference was observed as
`0e9e39f249a16976918f6564b8830bc894c89659`. The comparison also checks the
actual tokenized-prompt hashes across runs.

The [expanded 13-task cohort](../results/h100-baselines/quality-ruler-32k-frozen-13x32.json.gz)
has 32 examples per task. Its preparation
completed as cluster-run `1794084944098606007-ruler-32k-full13-freeze`.
The eight missing tasks ran under
`1794084969628211172-ruler-32k-full13-lod-bf16-missing8` and
`1794084976022477104-ruler-32k-full13-dense-missing8`; the verified full-panel
summary completed as `1794085009425920949-ruler-32k-full13-summary`. The
runner checks that the original five tasks in the expanded cohort match the
frozen original before running the new tasks. The summary is also on the H100
at `/home/ubuntu/comparison/results/ruler-32k-full13-summary.md`.

### Two-tier follow-up on selected Llama tasks

The [paired two-tier follow-up](../results/h100-baselines/ruler-32k-panel/ruler-32k-two-tier-bf16-selected-summary.md)
runs BF16 two-tier LoD on eight tasks with relatively low scores or gaps,
using the same 32 examples and tokenized inputs as dense and three-tier.
Using first-response scores, multi-key 2 improves from 75.00% to 96.88%
and multi-key 3 from 50.00% to 78.12%; dense scores 100% on both. CWE is
23.12% versus 27.19% for three-tier and 37.50% dense. QA 1 and QA 2 are
unchanged versus three-tier on these examples. These selected tasks do not
constitute a full-panel score; bootstrap intervals and full fixed-length
scores are in the linked summary. The production profile limits opened
regions to 1,024 leaves; larger regions stay coarse. The two-tier check,
run and summary were `1794088037196872991-ruler-32k-two-tier-bf16-check`,
`1794088071507944844-ruler-32k-two-tier-bf16-selected` and
`1794088078074635130-ruler-32k-two-tier-bf16-summary`. The per-example
[two-tier outputs](../results/h100-baselines/ruler-32k-panel/ruler-32k-two-tier-bf16-selected.json)
include graph/eager checks.

**Full-panel threshold and RetroInfer follow-up:** The [13-task comparison](../results/h100-baselines/ruler-32k-panel/ruler-llama-threshold-vs-retroinfer-full13-summary.md)
uses 32 matched examples per task. First-response averages are 88.88% dense,
87.04% two-tier LoD with normalized-query c=16 threshold routing, and 88.53%
RetroInfer. RetroInfer uses dense prefill while LoD uses sparse prefill; each
task score is based on only 32 examples. The full report also includes
three-tier top-2, per-task results, and fixed-length scores.

**Latest optimized threshold prefill and decode:** The [fresh combined run](../results/h100-baselines/threshold-next/REPORT.md)
scores **87.52%** across the same 13 tasks and 32 examples/task, versus 88.53%
RetroInfer and 88.88% dense. This validates the current materialized decoder
with closed-summary threshold prefill; the earlier 88.02% threshold-prefill
result is not substituted for it. The report also includes the four-task Qwen
comparison: adding threshold prefill raises CWE from 71.56% to 78.44%, versus
79.38% dense, while multikey-2, multikey-3 and QA-1 remain at 90.63%, 100% and
59.38%. The Qwen panel does not imply a complete RULER average.

**Earlier fixed-eight RetroInfer follow-up:** The same eight tasks and examples have a
[paired dense / LoD / RetroInfer comparison](../results/h100-baselines/ruler-32k-panel/ruler-llama-retroinfer-32k-summary.md).
On first-response scoring RetroInfer scores 100.00% on multi-key 3, compared
with 100.00% dense, 78.12% two-tier and 50.00% three-tier; RetroInfer minus
two-tier is +21.88 points (paired 95% interval +9.38 to +37.50). On CWE it
scores 36.25%, versus 37.50% dense and 23.12% two-tier; on QA 1 it scores
81.25%, versus 81.25% dense and 68.75% two-tier. These are only 32 examples
per task. RetroInfer uses dense prefill and its released GPU cache settings;
LoD uses sparse prefill, so the comparison does not isolate the cause of the
LoD gap. The [per-example RetroInfer outputs](../results/h100-baselines/ruler-32k-panel/ruler-llama-retroinfer-32k-selected.json)
record the cache settings and graph checks. The check, selected-task run and
summary were `1794090424903250548-ruler-llama-retroinfer-32k-check`,
`1794090492743783925-ruler-llama-retroinfer-32k-selected`, and
`1794090499531475031-ruler-llama-retroinfer-32k-summary`.

**Harness and prefill diagnostics:** On one long input per task, the
[Hugging Face BF16 SDPA check](../results/h100-baselines/ruler-32k-panel/ruler-llama-dense-hf-long-check.json)
matches the recorded dense runtime's first five token choices under teacher
forcing. This is a limited independent check, not proof that all logits or
all model behavior match. Its run is
`1794090645601525555-ruler-llama-dense-hf-long-check`.

In a [32-example multi-key 3 prefill ablation](../results/h100-baselines/ruler-32k-panel/ruler-llama-lod-dense-prefill-mk3-summary.md),
three-tier LoD builds its usual cache while using dense attention outputs
throughout prefill, then uses its usual LoD decode. The first generated token
matches dense for all 32 examples. First-response quality changes from 50.00%
with sparse prefill to 56.25% with dense prefill (paired 95% interval −9.38 to
+21.88 points); dense and RetroInfer each score 100%. The remaining gap
implicates the LoD decode/cache path or its approximation, but does not
distinguish an implementation defect from expected approximation error.
The [outputs](../results/h100-baselines/ruler-32k-panel/ruler-llama-lod-dense-prefill-mk3.json)
are retained. Runs were `1794090576059404725-ruler-llama-lod-dense-prefill-check`,
`1794090596411738989-ruler-llama-lod-dense-prefill-mk3`, and
`1794091788372860305-ruler-llama-lod-dense-prefill-summary`.

The complementary [sparse-prefill / dense-decode ablation](../results/h100-baselines/ruler-32k-panel/ruler-llama-prefill-decode-factorial-summary.md)
retains every original KV produced by sparse prefill, verifies every layer's
stored K and V, and uses the native dense decode graph. All 32 first generated
tokens match the original LoD sparse-prefill run. On the same multi-key 3
examples, first-response accuracy is **100.00%**, compared with 50.00% for
LoD prefill/decode, 56.25% for dense prefill/LoD decode and 100.00% for
dense prefill/decode. The paired interval for sparse-prefill/dense-decode
minus sparse-prefill/LoD-decode is +34.38 to +65.62 points. This identifies
decode/cache approximation as the dominant contributor on this small cohort;
it does not establish whether the loss is expected or an implementation bug.
The [individual outputs](../results/h100-baselines/ruler-32k-panel/ruler-llama-sparse-prefill-dense-decode-mk3.json)
are retained. Runs were `1794109429305584489-ruler-llama-sparse-prefill-dense-decode-check`,
`1794109436216099973-ruler-llama-sparse-prefill-dense-decode-mk3`, and
`1794109475177897594-ruler-llama-prefill-decode-factorial-summary`.

A [PyTorch reference decode diagnostic](../results/h100-baselines/ruler-32k-panel/ruler-torch-decode-three-failures.md)
replayed three incorrect Llama multi-key 3 examples using the optimized sparse
prefill and its reconstructed region/page cache. The pure PyTorch attention
reference also failed all three; one reproduced the identical first-response
token sequence, and two produced different errors. The optimized cache update
remains shared, so this checks the attention calculation rather than the
whole cache pipeline. This selected set is not an accuracy estimate.

The [combined native-query routing / four-sink ablation](../results/h100-baselines/ruler-32k-panel/ruler-llama-native-route-four-sink-mk3-summary.md)
changed both sparse prefill and three-tier BF16 top-2 decode on the same 32
multi-key 3 examples. First-response accuracy dropped from 50.00% to 25.00%
(paired change −25.00 points, bootstrap 95% interval −43.75 to −9.38); one
example improved and nine worsened. All first tokens after prefill matched.
The test combines two changes, so it cannot attribute the effect to either.
The default settings were not changed.

A [frontier audit of four shared two-tier and three-tier failures](../results/h100-baselines/ruler-32k-panel/ruler-frontier-audit-summary.md)
finds heads whose exact attention concentrates on an answer token while its
region ranks below the eight opened regions and its centroid greatly
underestimates the true region mass. For those heads, an oracle that opens the
answer token's region brings the two-tier attention output substantially
closer to dense. The oracle uses answer and dense-attention information and
has not been tested as an end-to-end decoding policy.

On the same four failures, a [fixed-query QUEST-style region-bound diagnostic](../results/h100-baselines/ruler-32k-panel/ruler-quest-region-bound-summary.md)
recomputed ideal per-channel min/max bounds over LoD's semantic regions. A
straight top-eight bound ranking captured less true attention mass than the
current mean-based router on these selected examples, because many bounds were
loose. This does not measure end-to-end quality or QUEST's own page layout.

A [two-tier 36-region follow-up](../results/h100-baselines/ruler-32k-panel/ruler-two-tier-36-route-mk3-summary.md)
uses the same 32 Llama multi-key 3 inputs. A PyTorch decode path with 36 open
regions scored 100.00% versus 84.38% for its paired eight-region control;
RetroInfer and dense also scored 100.00%. The previously optimized eight-region
run scored 78.12%. The PyTorch path uses different arithmetic and was checked
against the readable attention reference on one first decode step; the 36
regions are selected separately per query head, unlike RetroInfer's 36 shared
clusters per GQA group. This is not a speed measurement.

A [70% adaptive two-tier diagnostic](../results/h100-baselines/ruler-32k-panel/ruler-two-tier-p70-geometry-summary.md)
on those 32 Llama examples scored 56.25% using native-query centroid-mass
ranking, opening 194.63 regions per query head on average. A control using 70%
of LoD's normalized-query routing-score distribution scored 100.00% and
opened 364.26 regions on average; this distribution is a routing proxy, not
native centroid attention mass. Rankings and opening counts differ, so the
comparison does not isolate their effects. [Layer-by-head counts](../results/h100-baselines/ruler-32k-panel/ruler-two-tier-p70-geometry-layer-head.csv)
are saved. Both are FP32 PyTorch decode diagnostics, not speed measurements.

### Qwen3-8B two-tier follow-up

Status: **dense and two-tier runs complete**, with multi-key 2,
multi-key 3, CWE and QA 1 selected for their Llama gaps. This is
`Qwen/Qwen3-8B`, not Qwen3.8. The 32K task inputs were generated with the
RetroInfer RULER generators and Qwen3's tokenizer and chat template (thinking
disabled), because re-tokenizing the Llama cohort produced 42–45K inputs on
some tasks. The Qwen cohort is therefore different from the Llama cohort;
both Qwen methods share its exact examples. It is preserved
[here](../results/h100-baselines/ruler-32k-panel/qwen3-ruler-32k-frozen.json.gz).
All generated examples, including the output-token allowance, fit within
32,768 tokens. Runs use native RoPE, not YaRN.

On first-response scoring, dense versus two-tier is 93.75% versus 87.50%
for multi-key 2; 100.00% versus 93.75% for multi-key 3; 79.38% versus
73.75% for CWE; and 62.50% versus 56.25% for QA 1. The paired bootstrap
95% interval for the CWE difference is −11.25 to −1.25 percentage points;
the other differences' intervals include zero. Each task uses 32 examples.
These four tasks are not a full-panel average. The
[summary](../results/h100-baselines/ruler-32k-panel/qwen3-ruler-two-tier-selected-summary.md)
includes first-response and fixed-length scores and paired intervals. The
[dense outputs](../results/h100-baselines/ruler-32k-panel/qwen3-ruler-dense-selected.json)
and [two-tier outputs](../results/h100-baselines/ruler-32k-panel/qwen3-ruler-two-tier-selected.json)
contain individual predictions and graph checks.

The custom RetroInfer Qwen3 adapter's dense output matched Hugging Face's
first three tokens on one actual long input per selected task, after a short
prefill and decode check. These checks are limited and do not establish
identical logits for all inputs. LoD graph/eager checks are also required.
Run IDs: `24252-qwen3-ruler-32k-prepare` on the cluster, followed on the H100
by `1794088311276183212-qwen3-retro-hf-reference-check`,
`1794088452751224574-qwen3-ruler-dense-check`,
`1794088459947413933-qwen3-ruler-two-tier-check`,
`1794088901448546608-qwen3-ruler-long-hf-check`,
`1794088908443923178-qwen3-ruler-dense-selected-verified`, and
`1794088915293173952-qwen3-ruler-two-tier-selected-verified`. The summary
completed as `1794088921798651315-qwen3-ruler-two-tier-summary-verified`.
The [long-input reference check](../results/h100-baselines/ruler-32k-panel/qwen3-ruler-long-hf-check.json)
is preserved alongside the results.

**RetroInfer follow-up:** The [paired Qwen comparison](../results/h100-baselines/ruler-32k-panel/ruler-qwen3-retroinfer-32k-summary.md)
adds RetroInfer on these same 128 examples. Its first-response scores are
93.75% for multi-key 2, 100.00% for multi-key 3, 78.44% for CWE and 65.62%
for QA 1. Two-tier scores 87.50%, 93.75%, 73.75% and 56.25%, respectively;
the paired 95% intervals for their differences all include zero. Dense
scores are 93.75%, 100.00%, 79.38% and 62.50%. RetroInfer uses dense
prefill and LoD sparse prefill. The Qwen3 model adaptation is ours and uses
the same geometry-matched RetroInfer cache settings as Llama, not an
author-released Qwen3 integration. [Per-example outputs](../results/h100-baselines/ruler-32k-panel/ruler-qwen3-retroinfer-32k-selected.json)
are retained. The pilot, run and summary are
`1794090431061683450-ruler-qwen3-retroinfer-32k-check`,
`1794090508556493242-ruler-qwen3-retroinfer-32k-selected`, and
`1794090515092755420-ruler-qwen3-retroinfer-32k-summary`.

Across all 13 tasks, fixed-length scoring gives 89.36% dense and 82.86% LoD
BF16. After restricting scoring to the first response, the means are 88.88%
and 80.58%, respectively. The fixed-length procedure follows RetroInfer's
released `ignore_eos=True` convention, but may award credit for text generated
after an end-of-turn token. In particular, LoD multi-key 2 falls from 93.75%
to 75.00% under first-turn scoring. The
[full end-of-turn audit](../results/h100-baselines/ruler-32k-panel/full13-eos-audit.json)
records all 416 paired examples and both scores.

### Completed paired BF16 results

The BF16 model uses dense attention or LoD three-tier top-two BF16 KV. All
160 prompt hashes, input token hashes and reference answers agree. The input
token counts range from 28,561 to 32,747. Each method's first checked decode
step per task produced identical graph and eager logits.

| Task | Dense | LoD BF16 | LoD minus dense |
|---|---:|---:|---:|
| Multi-key | 100.00% | 53.12% | −46.88 pp |
| Multi-value | 99.22% | 99.22% | 0.00 pp |
| Variable tracking | 100.00% | 95.62% | −4.38 pp |
| Frequent-word extraction | 90.62% | 89.58% | −1.04 pp |
| QA | 81.25% | 71.88% | −9.38 pp |
| **Five-task mean** | **94.22%** | **81.89%** | **−12.33 pp** |

Each task uses 32 examples. Scores follow RULER's fractional matching for
multi-value, tracking and word extraction. Small differences are uncertain;
the paired percentile-bootstrap 95% interval for the multi-key difference is
−62.50 to −28.12 percentage points. Dense and LoD predictions are retained
in the [dense](../results/h100-baselines/ruler-32k-panel/ruler-32k-dense-full.json)
and [LoD BF16](../results/h100-baselines/ruler-32k-panel/ruler-32k-lod-bf16-full.json)
result files.

**End-of-turn audit:** The runner follows RetroInfer's `ignore_eos=True`
generation policy and scores the entire fixed-length response, including any
text after the first end-of-turn token. Rescoring only the first turn leaves
dense FWE (90.62%) and QA 1 (81.25%) unchanged, so this does not explain the
large difference from Double-P's dense FWE score. It changes dense multi-value
from 99.22% to 96.88%, LoD multi-key from 53.12% to 50.00%, and LoD QA 1
from 71.88% to 68.75%. All other scores in these five completed tasks stay
the same. The fixed-length numbers above remain as originally run; see the
[per-example end-of-turn audit](../results/h100-baselines/ruler-32k-panel/eos-audit.json).
Double-P's end-of-turn policy has not been established. The full-panel audit
above includes the eight added tasks.

**Dense baseline discrepancy:** Double-P reports 50.40% dense on 32K FWE
and 75.50% on QA 1; our scores are 90.62% and 81.25%, respectively. On our
32 examples, percentile-bootstrap 95% intervals for those task scores are
85.42–95.83% for FWE and 65.62–93.75% for QA 1. These intervals describe
sampling uncertainty for our cohort, not the difference between the two
studies. All three FWE reference words are among the most frequent coded
words in each of our 32 prompts. First-turn rescoring does not change either
dense score. The prompt token counts vary from 28,561 to 32,747 on FWE and
29,166 to 32,715 on QA 1. See the [cohort audit](../results/h100-baselines/ruler-32k-panel/dense-cohort-audit.json).
Our examples are from a third-party dataset with a Llama 3 chat template;
Double-P does not establish that its exact checkpoint variant, prompts,
generation policy or example selection match ours. The FWE mismatch remains
unexplained and should be investigated before treating the paper results as
directly comparable.

**Possible FWE/CWE column reversal in Double-P:** Its 32K table labels the
dense scores FWE 50.40 and CWE 95.17. For Llama-3.1-8B-Instruct at 32K,
[DefensiveKV Table 6](https://openreview.net/pdf/0148a9917eff6921b63d8cfff558e22e4a69e1e2.pdf)
reports dense CWE 45.22 and FWE 94.13, and
[SEAL Table 2](https://aclanthology.org/2025.acl-long.1405.pdf) reports dense
CWE 2.6 and FWE 93.0. The latter also finds that dense CWE declines sharply
as context length increases; Double-P's column labeled FWE follows this
pattern instead. A column reversal would make the Double-P numbers much more
consistent with these other measurements, but is **not verified**: the paper
does not establish its exact checkpoint, samples, or per-example outputs.
Do not silently relabel its published results.

As an additional conditional check, if Double-P used 200 examples and the
standard metric with three FWE references and ten CWE references per example,
scores increment by 1/6 and 0.05 percentage points, respectively. Its
32K Llama dense score labeled FWE, 50.40, cannot be rounded from the former
grid but fits the latter; the score labeled CWE, 95.17, fits the former.
Its reported sample count and exact scoring configuration are not established,
so this is corroborating evidence, not proof of a swap.

For our 32 dense CWE examples, a [reference audit](../results/h100-baselines/ruler-32k-panel/cwe-dense-audit.json)
finds all ten reference words are strictly the most frequent words in their
target lists. Fourteen responses begin by naming “the”; 15 receive zero
credit. The dense score is 37.50%, while LoD BF16 scores 27.19%. This is
consistent with a failure to extract from the list in many cases, rather than
incorrect reference words; it does not exclude other issues.

A paired CWE prompt-format diagnostic on those same 32 lists changed the
dense score from 37.50% to 49.69% when the Llama 3 chat wrapper was replaced
with RetroInfer-style `[INST]` turns and the system message removed. The
paired bootstrap 95% interval for the change is −2.50 to +26.88 percentage
points. First-turn and fixed-length scores agree in both runs. This shows
possible sensitivity to the combined formatting changes, but the interval is
wide and Double-P's actual prompt remains unknown. See the
[diagnostic summary](../results/h100-baselines/ruler-32k-panel/ruler-32k-cwe-prompt-ablation-summary.md)
and [recorded outputs](../results/h100-baselines/ruler-32k-panel/ruler-32k-cwe-prompt-wrapper.json).
The generation and summary runs were `1794087640956231496-ruler-32k-cwe-prompt-wrapper`
and `1794087744860960909-ruler-32k-cwe-prompt-compare`.

One concrete formatting difference is that RetroInfer's released
`llama-3.1-8b` RULER recipe uses the `meta-chat` template
`[INST] {task_template} [/INST]` plus the task's answer prefix. The frozen
third-party dataset instead has Llama 3 system/user/assistant chat headers.
Double-P does not specify its exact prompt format, so this is not established
as the cause of the discrepancy. The dense prompt-wrapper ablation completed:
run
`1794086365421246621-ruler-32k-dense-fwe-qa1-retroinfer-prompt` reuses the
same 32 FWE and 32 QA 1 contexts and answers, replacing only the wrapper and
removing the original system message. FWE remained at 90.62%; QA 1 changed
from 81.25% to 75.00%. Each task contains only 32 paired examples, and the
wrapper is not known to be the one used by Double-P. The
[ablation summary](../results/h100-baselines/ruler-32k-panel/ruler-32k-prompt-ablation-summary.md)
reports both full fixed-length and first-turn scores. The first summary job
failed because raw tokenizer decoding did not apply the same punctuation
cleanup as the recorded outputs; the corrected summary completed as
`1794087217760672812-ruler-32k-prompt-ablation-summary-fixed`.

The five-task mean weights the selected task scores equally; it is **not**
the 13-task 32K mean in Double-P Table 2. That table reports 90.00% for
dense and 89.91% for Double-P at 32K; its separate `Avg.` column averages
across 16K, 32K and 64K. For context, averaging just these five task scores
from Double-P Table 5 gives 84.62% for the paper's dense baseline and 85.53%
for Double-P. Those are unpaired paper-cohort figures, not measurements on
our examples. In particular, the paper's dense frequent-word score is 50.40%
versus 90.62% here, so the datasets or setups differ substantially and the
means should not be used as a direct LoD-versus-Double-P ranking.

Dense recovered all 32 multi-key answers; LoD recovered 17. On the 15 examples
that LoD missed, its first generated token was the same as dense's; the
first output-token divergence occurred at zero-based positions 2 through 20.
This observation does not isolate whether the errors originate in prefill,
decode attention, or an interaction between them. INT4 is incomplete and its
run has been stopped.

## Llama: a paired RULER panel

Use the same Llama-3.1-8B-Instruct model and frozen requests for dense,
RetroInfer, LoD three-tier top-two BF16 and INT4 KV, and QUEST. This follows
the 32K RULER track in [Double-P, Table 5](https://arxiv.org/html/2602.05191)
while keeping the amount of evaluation modest. The selected tasks cover
distinct failure modes:

| Task | What it probes | Published dense / RetroInfer / QUEST / Double-P (%) |
|---|---|---:|
| `niah_multikey_3` | Difficult multi-key retrieval | 100.00 / 98.50 / 89.00 / 98.50 |
| `niah_multivalue` | Multiple values for a key | 97.38 / 95.38 / 92.38 / 97.12 |
| `vt` | Variable tracking | 99.80 / 99.20 / 99.00 / 99.20 |
| `fwe` | Frequent-word extraction | 50.40 / 37.20 / 44.50 / 60.35 |
| `qa_1` | Document question answering | 75.50 / 76.00 / 73.00 / 72.50 |

The numbers above are **literature references**, not measurements on our
examples. The paper names LLaMA-3.1-8B but does not establish that its model
variant, examples, seed or prompt formatting match ours. Double-P has not
been run locally. Do not put its numbers in a table implying paired results
or use the paper's 13-task mean as a five-task score.

The [preparation script](retro_ruler_prepare.py) froze 32 randomly selected
examples per task, independently sampled without replacement with seed 42,
from the third-party [self-long RULER dataset](https://huggingface.co/datasets/self-long/RULER-llama3-1M),
pinned at revision `5da49e4000224d3289ffb14fc32cbd20211765ed`. It records
source-file and individual-prompt hashes, source rows, references and
prompts. A [compressed copy of the frozen cohort](../results/h100-baselines/quality-ruler-32k-frozen-32.json.gz)
is in the workspace; the original is on the H100 at
`/home/ubuntu/comparison/results/quality-ruler-32k-frozen-32.json`. Its
preparation was recorded by cluster-run as
`1794074614517009501-quality-ruler-llama-32k-freeze-32`. These are 32K-labelled
examples, not necessarily exactly 32,768 tokens; record actual token counts.
They are not known to be the paper's examples. No method was used to select
the samples.

Run each method on every exact same request, preserving the dataset's existing
chat formatting and answer prefix. Record the tokenizer and model revisions,
token IDs, generated outputs, all method settings and per-example scores.
Use greedy generation and the reference RULER scorer: for NIAH, variable
tracking and frequent-word extraction it gives the fraction of reference
strings found in the response; QA accepts any reference. Matches ignore case.
The released RULER generation limits are 128 tokens for NIAH, 30 for variable
tracking, 50 for frequent-word extraction and 32 for QA. Keep generation
policy identical and report it; the paper does not establish all details.

For every task report the method score, matched dense score, paired difference
from dense and a paired bootstrap interval over examples. Save individual
predictions so failures can be inspected. With 32 examples per task, intervals
will be wide and small gaps cannot be reliably ranked; for a binary task,
even 32 successes cannot rule out a meaningful failure rate. Treat this as a
screening panel and expand preselected tasks or run all 13 before making a
precise paper claim. Do not use a five-task average as if it were RULER's
standard aggregate.

### Method and speed alignment

- Use corrected LoD state updates, and the same three-tier top-two settings
  as the H100 timing comparison. Run BF16 and INT4 KV separately.
- Run RetroInfer with the released settings already used in our timing
  comparison; record its growing recent buffer and rebuild policy.
- QUEST's released kernels in our shared-KV adapter require FP16. Include
  both FP16 dense (its precision-matched control) and BF16 dense (the control
  for the other methods). Also report the two dense scores alongside one
  another. The current graph adapter rejects prompts whose token lengths are
  not divisible by 16. Support and validate partial pages first, or use a
  validated author path that accepts them; do not pad, truncate or discard
  examples merely to pass this check.
- For a paper-oriented QUEST point use 16-token pages and the paper's 25%
  budget. For a speed-linked point also evaluate the 2,048-token budget used
  in our previous timing table. These are different configurations; time the
  25% point before making a quality–speed comparison involving it.
- Report sink and local policies, dense layers, prefill behavior and actual
  exact-token budgets for each implementation. Double-P states four sink and
  64 local tokens; our existing implementations differ, so this is a
  comparison of the measured configurations, not an exact reproduction.
  Quality is batch-one; attach timing only where method, model precision,
  budget and context match, and identify any batch-size difference.

The existing three-passkey test is a sanity check, not this panel; old LoD
quality results from before the state-update correction are not a substitute.
The older Qwen and K2 RULER results use different models and cannot be pooled
with these Llama scores.

## Separate Qwen3 comparison with Multipole and Squeezed

For a later paper-aligned track, use Qwen3-8B and the LongBench V2 **Short**
(<32K) split, with dense, LoD, Multipole and the Multipole authors' Squeezed
baseline. [Multipole, Table 1](https://proceedings.neurips.cc/paper_files/paper/2025/file/359f442022ef31bcd9e41959576107ef-Paper-Conference.pdf)
reports accuracy for 128 and 512-token budgets; its 512-token scores are
41.5% dense, 40.2% Squeezed and 40.4% Multipole. Its settings include YaRN,
temperature 0.6, top-p 0.95, top-k 20 for Qwen3 and three trials. The paper's
Squeezed baseline incorporates Multipole's online clustering and uses one
centroid per eight tokens; Multipole uses one per 16. Label this adapted
Squeezed baseline explicitly, and report local/sink and token budgets rather
than assuming that the methods have identical costs.

The [Qwen3 long-context quality and decode report](../results/h100-baselines/qwen-long/REPORT.md)
now measures complete-model graph decode from shared optimized LoD prefill
activations. It pairs sampled greedy RULER quality with decode timings for
the released calibrated Squeezed port at 32K/64K and the Multipole authors’
1:8 Squeezed variant near 128K. Decode windows include ongoing KV writes
and index updates. The report also checks full dense and LoD quality at
64K and near 128K and records independent Transformers dense controls.

The paper-aligned LongBench V2 split, stochastic generation, and three-trial
evaluation remain pending.
