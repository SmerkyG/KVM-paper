# Paired LongBench v2 answer changes

This audit pairs the 96 untruncated Qwen3.8 examples by frozen ID and input-token hash. It checks the generated option against the benchmark key in each arm. Pure sparse scored six more correct answers than standard LoD. Nine selected options changed, including six wrong-to-right changes, three wrong-to-wrong changes, and zero right-to-wrong changes.

## Task domain

| Group | Examples | Standard correct | Pure sparse correct | Wrong to right | Right to wrong | Wrong to wrong |
|---|---:|---:|---:|---:|---:|---:|
| Code Repository Understanding | 5 | 3 | 3 | 0 | 0 | 0 |
| Long In-context Learning | 11 | 6 | 6 | 0 | 0 | 0 |
| Long Structured Data Understanding | 3 | 0 | 0 | 0 | 0 | 1 |
| Long-dialogue History Understanding | 12 | 6 | 8 | 2 | 0 | 1 |
| Multi-Document QA | 26 | 17 | 19 | 2 | 0 | 0 |
| Single-Document QA | 39 | 25 | 27 | 2 | 0 | 1 |

## Official length label

| Group | Examples | Standard correct | Pure sparse correct | Wrong to right | Right to wrong | Wrong to wrong |
|---|---:|---:|---:|---:|---:|---:|
| medium | 39 | 20 | 22 | 2 | 0 | 3 |
| short | 57 | 37 | 41 | 4 | 0 | 0 |

## Rendered input length

| Group | Examples | Standard correct | Pure sparse correct | Wrong to right | Right to wrong | Wrong to wrong |
|---|---:|---:|---:|---:|---:|---:|
| <32K | 38 | 22 | 25 | 3 | 0 | 0 |
| 32K to <64K | 22 | 15 | 17 | 2 | 0 | 0 |
| 64K to <96K | 16 | 10 | 10 | 0 | 0 | 0 |
| 96K to <128K | 20 | 10 | 11 | 1 | 0 | 3 |

Length and domain breakdowns are descriptive and were constructed after viewing the aggregate result. Five gains fall below 64K, where 60 of the 96 prompts fall.

## Changed choices

| Frozen ID | Subdomain | Input tokens | Key | Standard | Pure sparse | Unweighted | Dense | Question and evidence |
|---|---|---:|---|---|---|---|---|---|
| `66f2e0bc821e116aacb2bf37` | Legal | 10,927 | D | C | D | C | C | Two legal documents. Pure sparse selected Ginsburg’s position on sex-neutral leave. The other arms selected the egg-freezing proposition. This prompt fits in the first exact 16K block. |
| `66ebde0f5a08c7b9b35e1256` | Academic | 21,801 | D | A | D | D | A | Asked how two image-conditioning streams work together. Pure sparse and unweighted selected the choice describing direct image injection into noise, consistent with the paper’s description of latent concatenation. |
| `671b1480bb02136c067d4f14` | Agent history QA | 29,667 | A | D | A | A | A | Asked which player won most often. The per-player result messages give player 0 seven wins and player 6 five. |
| `671b2c67bb02136c067d50f7` | Agent history QA | 45,187 | D | B | D | D | B | Asked for the second-highest contributor. Player 9 contributed 71 tokens, behind player 5’s 76. Standard LoD and dense selected the highest contributor. |
| `66ece28e821e116aacb1db1f` | Literary | 55,787 | D | A | D | D | D | Asked why the lake weeps in the prologue. The passage explicitly identifies the lake’s own reflected beauty. Standard LoD chose Narcissus’s beauty. |
| `66f016e6821e116aacb25497` | Legal | 115,963 | D | B | D | B | B | Asked which environmental compliance scenarios carry risk. Pure sparse selected all three. The other arms selected only the marine renewable-energy scenario. |
| `670fbb78bb02136c067d2cd3` | Detective | 99,062 | B | C | A | C | C | Detective inference about the killer. |
| `66f2b5ec821e116aacb2b1ec` | Table QA | 110,379 | C | B | D | D | B | Album-rating question involving a within-artist minimum and a second qualifying album. |
| `6719c1febb02136c067d44db` | Dialogue history QA | 119,222 | D | A | C | D | D | Asked which dialogue request the assistant could not satisfy. Dense and unweighted recovered the keyed answer. |

Unweighting the summaries recovered four of the six gains. The two remaining gains were the legal questions. Dense recovered two of the six, and it chose the same incorrect option as standard LoD on the other four.

The 10,927-token legal prompt used exact prefill in both LoD arms. Its choice changed during decode. Standard and pure sparse opened 208.6 and 207.9 remote entries per decode query head, respectively. The counters measure the amount of exact work. They cannot identify which evidence was opened or how summaries changed the option logits.

At 128K RULER, the standard-versus-sparse gain is concentrated in Multikey-3, with smaller gains in Multikey-2 and SQuAD. That is consistent with closed summaries carrying useful broad context on the synthetic retrieval task. In LongBench, the options often require comparing related evidence or checking a specific detail. A larger population-biased summary contribution can plausibly shift the option scores toward a distractor. The saved outputs do not identify the responsible heads, regions, or tokens.

[LongBench results](REPORT.md) · [Frozen selection](longbench-selection.json) · [128K RULER results](../reviewer2-summary-ablation-v2/REPORT.md)
