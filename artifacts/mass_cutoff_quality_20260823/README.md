# Mass-cutoff two-level LoD quality panel

Qwen/Qwen3.5-0.8B, Hugging Face inference, BF16 leaves and page summaries,
16sqrt(T) state schedule, 4096-token state-update groups, expert-major paged
leaves, mass threshold 1/16, at most 16 opened routes, and previous-chunk state
partition LSE with represented-token growth correction.

## RULER NIAH

Each cell uses 64 examples. `Overall` is the total over NIAH-1, NIAH-2, and
NIAH-3.

| Context | NIAH-1 | NIAH-2 | NIAH-3 | Overall |
|---:|---:|---:|---:|---:|
| 8k | 64/64 | 64/64 | 64/64 | 192/192 |
| 16k | 64/64 | 64/64 | 64/64 | 192/192 |
| 32k | 64/64 | 64/64 | 64/64 | 192/192 |
| 64k | 64/64 | 64/64 | 64/64 | 192/192 |

Total: 768/768 exact matches.

Cluster run: `10198-masscutoff-niah-full-panel`.

## ProLong cross entropy

Thirty-two deterministically selected 8192-token documents, evaluated
sample-for-sample in all modes. Each sample predicts 8191 tokens.

| Mode | Mean CE | Perplexity | CE delta vs full | PPL ratio vs full |
|---|---:|---:|---:|---:|
| Full attention | 2.578282 | 13.174489 | 0 | 1.000000 |
| Top-8 two-level LoD | 2.580792 | 13.207591 | +0.002509 | 1.002513 |
| Mass-cutoff two-level LoD | 2.587190 | 13.292367 | +0.008908 | 1.008947 |

The paired mass-cutoff minus full CE delta has sample standard deviation
0.007782, standard error 0.001376, and normal-approximation 95% interval
[0.006211, 0.011604]. Mass cutoff is +0.006398 CE versus matched top-8 LoD,
or a 1.006419 perplexity ratio.

Cluster runs: `10199-masscutoff-ce8k-paired32` and
`10200-top8-ce8k-paired32-current`.
