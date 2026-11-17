# Wave 1: 120M experiments

Four independent eight-GPU jobs use the same ProLong data ordering, GPT-2
tokenizer, 8,192-token context, global batch of 524,288 tokens, seed 0, and
6,000 optimizer steps (3,145,728,000 tokens). Checkpoints are saved at steps
2,000 and 4,000 and the final model is exported in Hugging Face format.

| ID | Model | Total parameters |
| --- | --- | ---: |
| K0 | GPTAlpha-2, KVM `16 sqrt(N)`, least-redundant append | 162,942,096 |
| E1 | K0 with uniform random append selection | 162,942,096 |
| B1 | K0 with chunked bipartite grouping | 162,942,096 |
| Q0 | Qwen3-style full attention, half RoPE, GQA | 162,223,872 |

The existing "120M" configuration label is retained; total parameter counts
include both untied embedding matrices. Q0 differs in total count by 0.44%.
Q0 uses 12 layers, width 768, six query heads and three KV heads of width 128,
and SwiGLU intermediate width 2,304. Qwen3 blocks, RMSNorm, QK RMSNorm, and
initialization come from Transformers 5.3.0. Its 64-channel RoPE uses exactly
the same frequency and rotation convention as GPTAlpha-2. Token shift and
value residuals are absent. The trainer already shifts labels, so the Qwen
adapter uses the same unshifted cross-entropy helper as the other runs.

The initial K0/E1/B1 runs shared the reference KVM update arithmetic and a
lower-right causal SDPA bias, enabling native Flash SDPA without materializing
a dense mask. Their frozen sources retain that implementation. Original-path
reruns described below use the original dense-mask attention instead.
Their initial parameters and state-size schedules are identical. E1 uses
separate seeded train/evaluation routing generators, independent of model
initialization and global random state. Seeds also distinguish layers and
DDP ranks. Exact training resumption is not implemented by the base trainer;
the intermediate optimizer snapshots are retained for inspection.

B1 keeps sinks outside the candidate pool. Existing compressed rows and the
overflow block form a pool, in that order. Within each 256-row matching chunk,
even source rows choose their most similar odd destination using normalized
keys. Candidate edges are ranked across chunks and exactly enough source rows
are merged to reach K0's scheduled budget. Surviving rows retain their order.
Old sums are not gated again; merged incoming contributions receive the merge
gate. Each surviving destination retains its original birth radius. Thus B1
compares the grouping procedure while retaining KVM accumulation/readout.

The historical count-readout branch had a wrong reduction axis. It is fixed
and tested with unequal token/slot counts and a protected sink. Wave 1 uses
birth-radius readout, which does not execute this branch.

## Validation and launch

Use the ROCm environment installed with `uv sync --python 3.12 --extra rocm72`.
All GPU commands must be sent through `cluster-run`.
PyArrow is pinned to 23.0.1 because the newer 26.0.1 wheel requires NumPy 2,
while the training environment intentionally retains NumPy 1.26.4.

```bash
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python scripts/wave1.py inspect
cluster-run --detach --num-gpus 8 --name wave1-preflight -- \
  env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 uv run --no-sync python \
  scripts/wave1.py smoke --device cuda --context 8192
uv run --no-sync python scripts/wave1.py prepare --output experiments/wave1-RUN
uv run --no-sync python scripts/wave1.py submit --output experiments/wave1-RUN
```

Preparation copies tracked and untracked, non-ignored sources into an immutable
source snapshot and records per-file SHA-256 hashes, resolved configurations,
parameter/token budgets, and Git provenance in `manifest.json`. Submission
records each cluster job ID. Workers verify the source hashes before launch,
use the shared uv environment, and request all eight GPUs on one node. They
use the cluster-allocated rendezvous port. Compiled kernels are cached on each
node's temporary filesystem. Dataset/tokenizer loading uses existing offline
caches. W&B defaults to online logging in `speedtrain_prolong`, using the
configured account. An explicit `WANDB_MODE` environment setting overrides
that default. The initial Wave 1 submissions used offline W&B logging; their
telemetry is retained locally.

The output folder contains `runs/ID/train.log`, intermediate checkpoints,
final Hugging Face files, and `telemetry/ID`. Source/configuration changes made
after submission do not affect these runs. Models with dynamic mixer imports
should be evaluated with `HF_WRAPPER_BASE_PATH` set to the saved source folder.

Review the first Wave 1 results before launching any Wave 2 or 350M+ training.

## Qwen evaluation

Reuse the existing GPTA-2 scores from paper Tables 2 and 4. The default
`--qwen` evaluation launcher runs Q0 alone; `--gpta` is an optional explicit rerun.
Use `--model K0=/absolute/path/to/K0` (or E1/B1) for the KVM controls.
The Q0 suite is the six zero-shot short tasks, NIAH-S1/S2/S3 at 4K/8K/16K/32K
(500 examples per task and length), all 13 RULER tasks at 4K, all four
LongBench few-shot tasks, and 32K TextbookChapters position-wise loss.
Keep the paper's prompts, generation settings, and score conventions.
Use the harness seeds `0,1234,1234,1234` for Python/NumPy/PyTorch/few-shot.
No model context-length override is introduced. Record sample outputs and
task configurations so any discrepancy can be investigated before rerunning
an existing baseline.

The pinned evaluation extra is installed in a separate environment:

```bash
UV_PROJECT_ENVIRONMENT=.venv-eval uv sync --python 3.12 --extra rocm72 --extra eval
UV_PROJECT_ENVIRONMENT=.venv-eval uv run --no-sync python scripts/eval_wave1.py \
  prepare --output experiments/qwen-eval-RUN --qwen /absolute/path/to/Q0
```

Preparation provisions and checks NLTK's sentence tokenizer in a private
`~/.cache/kvm-eval-nltk` directory. Workers receive this path through
`NLTK_DATA`; the shared home-level NLTK cache has permissions that newer
NLTK versions reject. A failed suite makes the driver exit with an error,
while completed suites remain available for resumption.

Submit the frozen `source/scripts/eval_wave1.py worker --output OUTPUT` with
`cluster-run --detach --num-gpus 8`, `uv run --no-sync`, and
`UV_PROJECT_ENVIRONMENT` pointing to the absolute `.venv-eval` directory.
The driver performs an eight-example likelihood/generation preflight, then
runs short tasks first. Evaluation progress, scores, task-level logs, and
sample predictions are saved locally under the evaluation output. W&B is
reserved for online training logs in `smerky/speedtrain_prolong`.
Use `prepare --resume-from PREVIOUS_OUTPUT` to reuse completed suites with
identical models and evaluation settings. These evaluations measure quality; their
wall-clock times are not optimized model latency benchmarks.

GPTA-2's published 120M short-task reference is LAMBADA perplexity 51.5,
LAMBADA accuracy 31.0%, ARC-Challenge normalized accuracy 24.7%, ARC-Easy
accuracy 49.2%, HellaSwag normalized accuracy 32.8%, PIQA accuracy 64.4%,
and WinoGrande accuracy 50.0% (six-accuracy mean 42.0%). Its LongBench
few-shot mean is 16.9 and RULER-4K is 34.0. Historical task/harness versions
should be retained when interpreting small differences.

## Qwen learning-rate sweep

Reuse Q0 as the `2e-3` point and train three additional full-budget runs:
`Q-lr-5e-4`, `Q-lr-1e-3`, and `Q-lr-4e-3`. All use the same initialization
seed, data order, 6,000 steps, 200-step warmup, linear decay, optimizer,
weight decay, batch size, partial RoPE, and Qwen architecture as Q0.
Only peak learning rate and the W&B display-name suffix change.
Select the peak LR by final DCLM validation loss at step 6,000, before
examining downstream scores for the new runs. Expand the grid one step
outward if its best point is a boundary.

```bash
uv run --no-sync python scripts/wave1.py prepare \
  --output experiments/qwen-lr-sweep-RUN \
  --experiments Q-lr-5e-4 Q-lr-1e-3 Q-lr-4e-3
WANDB_MODE=online uv run --no-sync python scripts/wave1.py submit \
  --output experiments/qwen-lr-sweep-RUN
```

Each job reserves all eight GPUs on one node and records online training
logs in `smerky/speedtrain_prolong`. Submission queues jobs when all nodes
are occupied. The prepared manifest records resolved settings and immutable
sources. For the initial sweep, `validation.json` additionally verifies the
configuration differences against the original Q0 manifest, unchanged model
and optimizer sources, and identical training/validation dataset fingerprints.
The Qwen HF export wrapper includes the previously verified RoPE import fix;
the training loop is unchanged.

The first four completed LR points selected `4e-3` (validation loss 3.4195)
over `2e-3` (3.4276). The boundary extension adds `8e-3`; a parallel `1e-2`
trial was requested afterward. Both use the same 6,000-step budget and
unchanged settings. `scripts/qwen_lr_pipeline.py` trains a trial, selects by
final validation loss, and evaluates the selected checkpoint in the separate
evaluation environment. The `1e-2` controller waits for the `8e-3` selection
and includes all six LR points. If the selected checkpoint matches the other
controller's choice, it forwards that evaluation's status and releases its
GPU allocation. Otherwise it evaluates the new winner. Training logs remain
online; evaluation logs remain local.

## Original-implementation reruns

The user requires approval before any further kernel or attention-backend
changes. The working experiment adapter now inherits dense-mask attention.
The first Wave 1 checkpoints and results remain recorded with their original
Flash-compatible sources; comparisons against the historical paper must
identify that implementation difference.

The following full-budget eight-GPU reruns use seed 0, unchanged data order,
and the same model, optimizer, learning-rate schedule, and 6,000 training steps:

| Experiment | Cluster job | Training output |
| --- | --- | --- |
| K0, original KVM | `24800-kvm-sqrt-original-train-eval` | `experiments/kvm-sqrt-original-20261110` |
| B1, bipartite | `24801-kvm-B1-original-train-eval` | `experiments/kvm-ablations-original-20261110` |
| E1, random append | `24803-kvm-E1-original-train-eval` | `experiments/kvm-E1-original-20261110` |

K0 restores `model/kvm_mixer.py` byte-for-byte from repository commit
`ece92749eb0aa9b909e7aac396d8c70cdb088527`. E1/B1 restore that file with only
an append-selection hook required for the random-append control. Their
attention method and base merge arithmetic are checked against the original
method ASTs. The random-selection and bipartite method bodies remain unchanged
from the first runs, and the Flash attention override is removed. Tiny-model
CPU forward/backward checks verify finite computations and dense boolean
attention masks. The manifests and `validation.json` files record these checks.

`scripts/train_eval_pipeline.py` runs training with online W&B in
`smerky/speedtrain_prolong`, then runs the same full evaluation protocol with
local logging. Each job has its own controller status and evaluation output.
The superseded queued job `24802` was cancelled before starting so E1 could
use an independent controller output directory; its replacement is `24803`.
