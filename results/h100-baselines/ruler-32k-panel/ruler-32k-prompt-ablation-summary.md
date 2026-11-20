# RULER dense prompt-wrapper diagnostic

The same FWE and QA 1 contexts and reference answers are run with the original Llama 3 chat wrapper and with the `[INST] ... [/INST]` wrapper in RetroInfer’s released RULER recipe. The system message is absent from the latter. This tests the wrapper change, not an exact reproduction of Double-P, whose prompt template is not established.

| Task | Original, full output | RetroInfer wrapper, full output | Difference, pp (paired 95% interval) | Original, first turn | RetroInfer wrapper, first turn |
|---|---:|---:|---:|---:|---:|
| fwe | 90.63 | 90.62 | -0.00 [-3.12, +3.12] | 90.63 | 90.62 |
| qa_1 | 81.25 | 75.00 | -6.25 [-15.62, +0.00] | 81.25 | 75.00 |

Generation follows the released RetroInfer fixed-length `ignore_eos=True` policy. First-turn scores exclude text after the first Llama end-of-turn or end-of-text token. Each task has 32 examples; intervals are paired bootstrap intervals over those examples.
