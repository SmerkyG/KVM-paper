# Wave 2: count readout and Qwen portability

All runs use the original KVM dense causal mask and SDPA call. The frozen mixer
is taken from the original K0 rerun, with changes restricted to FP32 assignment
counts, the corrected count reduction, and count division. Those changes are
inactive for birth-radius readout. No kernel or attention backend changes are
made in this wave.

| Experiment | Model and readout | Peak LR | Cluster job |
| --- | --- | --- | --- |
| Q1 | Qwen3 + KVM-sqrt, birth radius | `4e-3` | `24821` (native attention, verified by `24820`) |
| F0 | GPTA-2 + KVM-256, birth radius | `2e-3` | `24805` |
| F1 | GPTA-2 + KVM-256, corrected count | `2e-3` | `24806` |
| A3 | GPTA-2 + KVM-sqrt, corrected count | `2e-3` | `24807` |

F0/F1 isolate readout at fixed capacity. Compare A3 against the original K0
rerun to test that same change with growing capacity. Q1 matches the selected
Qwen full-attention run's RMSNorm, SwiGLU, six query and three KV heads, half
RoPE, initial shared weights, and training settings. It adds 30,864 parameters
for KVM, about 0.02% of the Qwen control's total.

The frozen manifests are under `experiments/wave2-20261110`. Each experiment
trains for 6,000 steps on eight GPUs with seed 0, identical ProLong data order,
and 8,192-token context. Training logs online to `smerky/speedtrain_prolong`.
Each job automatically runs the existing evaluation suite with local logging.

Q1 jobs `24804` and `24809` aborted during compilation warmup on different
nodes, before optimizer steps or checkpoint creation. The eager one-layer
forward/backward passed in `24808`; its compiled backward faulted. Compiling
the loss alone passed (`24811`), while compiling the whole decoder block alone
reproduced the fault (`24810`). A synchronized trace (`24813`) localized the
abort to Triton autotuning of a fused value and merge-gate gradient kernel.
The original state-read and state-update operations passed in isolation with
both three and six KV heads (`24814`).

Separating attention and feed-forward compilation passed the one-layer
forward/backward on all eight GPUs (`24812`). Q1's decoder wrapper now applies
that split. The KVM mixer file, SDPA call and mask, weights and parameter
names, optimizer configuration, and data order are preserved. The candidate
is frozen in `experiments/wave2-Q1-original-fixed-20261110`; full-shape job
`24815` checks three forward/backward passes at 12 layers and 8 x 8192 tokens
per GPU, without optimizer steps. CPU comparison of the old and new wrappers
found bitwise-identical initial weights, logits, loss, and gradients, with
unchanged optimizer parameter labels. At full depth, `24815` still faults in
backward, and the corresponding training retry `24816` became nonfinite at
step 2 before another GPU fault. The split is therefore not a verified fix.
The full-depth diagnosis in `experiments/q1-fullshape-diagnosis-20261110`
reproduced the fault (`24817`). Disabling attention's pointwise autotuning
allowed one backward pass, but the second hit a memory aperture violation in
the same family of fused value/merge-gate gradient kernels (`24818`). Native
attention with compiled MLP and loss passed three full-shape batches on all
eight GPUs (`24819`).

The verified candidate leaves only Q1's KVM attention outside `torch.compile`.
The original mixer, dense mask, SDPA call, data, and full-run configuration are
unchanged. The actual trainer completed 16 optimizer steps with finite losses
(`10.9951` to `8.0232`) and validation (`8.1655`) in `24820`. Its smoke checkpoint
reloaded without key errors, all 195 stored tensors were finite, and prefill
and cached decode both succeeded. This checkpoint is for verification only.
The frozen candidate and `smoke-verification.json` are in
`experiments/wave2-Q1-native-attention-20261110`. The user approved this
compilation change and full training was submitted as `24821` with the
original 6,000-step recipe and automatic local evaluation.

```bash
uv run --no-sync python scripts/wave2.py prepare \
  --output experiments/wave2-RUN \
  --kvm-reference experiments/kvm-sqrt-original-20261110 \
  --qwen-reference experiments/qwen-lr-sweep-20261110 \
  --eval-reference experiments/qwen-selected-eval-20261110
uv run --no-sync python scripts/wave2.py submit --output experiments/wave2-RUN
```
