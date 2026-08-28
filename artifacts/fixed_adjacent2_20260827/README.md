# Fixed adjacent-2 versus adjacent-4 (Qwen3.5-0.8B)

This repeats the adjacent-pooling diagnostic with fixed non-overlapping pairs
of remote-token K/V instead of groups of four. All other settings match
`artifacts/fixed_adjacent4_20260827`: BF16, 512 exact local tokens, exact leaf
archive, coarse mean plus `log(count)`, chat-formatted NIAH with thinking
disabled, and a sufficiently large state target to prevent any KVM routing or
cross-group merge.

## NIAH-S3

| Context | Policy | T/2 | T/4 |
|---:|---|---:|---:|
| 8K | coarse only | 0/8 | 0/8 |
| 8K | top-4 groups | 64/64 | 64/64 |
| 8K | top-8 groups | 64/64 | 64/64 |
| 8K | mass threshold 1/128, cap 16 | 8/8 | 8/8 |
| 64K | top-4 groups | 8/8 | 8/8 |

With pairs, top-4 opens eight exact remote tokens; with groups of four, it
opens sixteen. Both retain perfect retrieval on these panels. Coarse-only
generation still fails completely, even though the pair means are less blurry.

## ProLong loss

Eight identical deterministic 8K documents and 65,528 prediction tokens.

| Method | CE | PPL | PPL vs full |
|---|---:|---:|---:|
| Full attention | 3.252045 | 25.8431 | baseline |
| Similarity-routed KVM LOD, top-8 | 3.254731 | 25.9126 | +0.27% |
| Fixed T/2, top-8 | 3.258194 | 26.0025 | +0.62% |
| Fixed T/2, top-4 | 3.263383 | 26.1378 | +1.14% |
| Fixed T/2, mass 1/128, cap 16 | 3.275737 | 26.4627 | +2.40% |
| Fixed T/4, top-8 | 3.272414 | 26.3749 | +2.06% |
| Fixed T/4, top-4 | 3.283240 | 26.6620 | +3.17% |
| Fixed T/4, mass 1/128, cap 16 | 3.301290 | 27.1476 | +5.05% |

Pair pooling removes about 70% of the top-8 perplexity overhead introduced by
pooling groups of four. It remains about 2.3 times the CE delta of
similarity-routed KVM (`+0.00615` versus `+0.00269`). Seven of the eight paired
documents have higher loss than full attention; the eighth improves by only
0.00088 CE.

At equal exact-leaf work, T/2 top-8 and T/4 top-4 each open sixteen remote
tokens, but T/2 is substantially better (26.0025 versus 26.6620 PPL). The
benefit therefore comes from sharper coarse summaries rather than merely from
opening more leaves.

## Tradeoff

T/2 is a much better quality point than T/4, but it doubles coarse-attention
work and coarse-state storage. It also remains linear-state and
quadratic-prefill: the
coarse field is half of the remote context rather than `O(sqrt(T))`.
