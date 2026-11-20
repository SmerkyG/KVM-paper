# Llama 3.1 64K RULER retrieval: per-entry versus per-region

Four selected retrieval tasks, 16 matched examples per task; BF16 model and KV, greedy decoding. LoD uses the same top-eight sparse prefill for both rules; dense uses dense prefill and decode. These selected tasks and small sample do not represent a full RULER average. Region c=16 retains the existing 1,024-entry cap; entry multiple B uses the B sqrt(n) remote-entry bound and has no fixed per-region cap.

| Task | Dense | Per-region | Entry 16 | Entry 16 minus region, pp (paired 95% interval) |
|---|---:|---:|---:|---:|
| niah_single_3 | 100.00 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| niah_multikey_3 | 100.00 | 75.00 | 37.50 | -37.50 [-62.50, -12.50] |
| niah_multivalue | 98.44 | 98.44 | 98.44 | +0.00 [+0.00, +0.00] |
| niah_multiquery | 100.00 | 93.75 | 93.75 | +0.00 [-4.69, +4.69] |

Scores stop at the first end-of-turn or end-of-text token. Intervals are paired bootstrap intervals; each answer can move a single-answer task score by 6.25 points.

## Last-step remote work and proxy coverage

The measurements sample the final decode step for every example, layer, and query head. Proxy mass is normalized over remote regions and is not exact attention mass. Exact local and sink entries are excluded.

| Task | Rule | Mean entries/head | Mean regions/head | Mean proxy mass opened | Max bound fraction | Empty heads |
|---|---|---:|---:|---:|---:|---:|
| niah_single_3 | Per-region | 1042.5 | 24.60 | 32.47% | 2.533 | 0.21% |
| niah_single_3 | Entry 16 | 317.8 | 77.67 | 34.18% | 0.598 | 0.15% |
| niah_multikey_3 | Per-region | 1281.5 | 27.61 | 37.37% | 3.929 | 0.16% |
| niah_multikey_3 | Entry 16 | 338.7 | 63.30 | 29.92% | 0.396 | 0.04% |
| niah_multivalue | Per-region | 1003.3 | 24.53 | 33.59% | 2.304 | 0.19% |
| niah_multivalue | Entry 16 | 313.7 | 75.35 | 34.42% | 0.598 | 0.15% |
| niah_multiquery | Per-region | 1108.0 | 25.11 | 31.81% | 2.341 | 0.15% |
| niah_multiquery | Entry 16 | 317.3 | 76.18 | 32.57% | 0.593 | 0.13% |

The maximum bound fraction uses 16 sqrt(n) for the region control and each entry rule’s work bound.

Frozen cohort SHA-256: `052da0ca32b648f9eec8a5bb42ab10129a92a91961bbdfce97eccc9602e60dc9`.
Results: [Dense](dense-ruler-64k-retrieval-4x16.json) · [Per-region](threshold-region-ruler-64k-4x16.json) · [Entry 16](threshold-entry-ruler-64k-4x16.json).
