# console-nn model card

## Intended use

Classify console output into `success`, `fail`, `block`, `unknown`, label
16-token chunks, and flag whether the operation reached a terminal state.
Designed for developer/CI logs: shell, Python, package managers, builds,
systemd, Docker/Kubernetes/Terraform, Windows commands.

Not a security oracle. Do not use as the only gate for destructive operations
or privileged deployments.

## Architecture

FCN over 16-token chunks, ALBERT-style factorized embedding, dynamic chunk
count, masked mean+max document pooling, chunk/document/completion heads.
No recurrence, no attention, no fixed sequence length.

## Data

- 45,000 synthetic documents (36k/4.5k/4.5k). Exact duplicates are removed
  from the mixed splits, with test taking priority over validation and train.
- 201 real Linux sessions collected by `tools/collect_real.py` (safe local
  commands in a temp workspace), 147 curated Linux/Windows transcripts from
  `examples/corpora/curated`, plus 23 concrete example sessions.
- The 23 examples are held out in the real test split; curated transcripts are
  split 70/15/15 so they contribute to both training and evaluation.
- 20 complex held-out command families and 20+ error families.
- Train/val/test use separate seeds, context lengths and command pools.
- Real test sessions are held out from their own split and are also reportable
  with `console_nn eval --data data/real --split test`.

## Supported log languages

**English only.** The tokenizer vocabulary, chunk/document labels, synthetic
templates and keyword expectations are English. Command names and paths are
language-neutral, but status markers (`error`, `failed`, `permission denied`,
`not found`, `success`, ...) are English. Non-English output may be classified
from command/context alone, but the model is not trained or validated for it.
No translation or locale detection is performed.

## Covered domains

The generator covers 51 command families. They are grouped into seven families
for the domain-holdout evaluation (`tools/eval_domain_holdout.py`):

| group | domains |
|---|---|
| python | python, pip, pytest, conda |
| build_package | build, compile, package, package_update, maven, gradle, dotnet |
| system | shell, systemd, kernel, journal, filesystem, network, security |
| data_infra | archive, backup, cloud, data_process, database, docker, git, media, redis, postgres, mysql, nginx |
| windows_misc | windows_cmd, windows_update, powershell, real_errors, errors, commands, multi, multi_chain |
| language_ecosystems | node, npm, cargo, go, ruby, php |
| orchestration | kubectl, helm, terraform, ansible, github_actions, gitlab_ci, jenkins |

Coverage means these tools appear in training data; it is not a guarantee for
every subcommand, plugin, locale or vendor-specific message. Unseen tools are
handled only through shared English vocabulary and transfer.

## Metrics

See `results/<variant>/metrics.json` and the local training log. Metrics are
split into float and int8, including per-class precision/recall/F1, confusion
matrix, chunk accuracy, completion accuracy, agreement and probability delta.
`console_nn validate --strict` enforces the int8 gate.

Real/curated held-out test (`data/real`, 85 sessions): document accuracy
90.59%, document macro F1 77.57%, chunk accuracy 57.37%, completion accuracy
87.06%.
The split contains two `unknown` examples and mixes collected, curated and
example sessions.

Additional evaluations:

- **Domain holdout** (`results/cpu/domain_holdout.json`): one domain family
  removed from training, three seeds per fold, bootstrap 95% document CIs.
- **Baselines and ablations** (`results/cpu/baselines.json`,
  `results/cpu/ablations.json`): regex, hashed word 1..3-gram logistic
  regression, no digit tokenization, no factorized embedding.
- **Calibration** (`results/cpu/calibration.json`): document-head
  temperature fitted on validation NLL, ECE on synthetic and real validation,
  and an unknown risk-coverage curve.
- **Contrast pairs** (`results/cpu/contrast.json`): the same command with
  different outcomes must receive different, correct verdicts. The score is
  stored as `contrast_pair_acc` in `results/cpu/metrics.json`.

### Evaluation snapshot (CPU checkpoint)

- Domain holdout: macro document accuracy **77.34%** over 7 held-out families (68.09% to 88.86% per family), 4584 documents.
- Baselines: regex 67.39% synthetic / 84.71% real; hashed 1-3 gram logistic regression 96.79% / 88.23%; CPU FCN 98.95% / 90.59%.
- Same-budget ablation control: 98.31% synthetic.
- Calibration: temperature 1.0846; ECE synthetic val 0.00314 -> 0.00320; real val 0.10901 -> 0.10705 (n=43).
- int8 graph unchanged: 104 -> 104 ops.
- Contrast pairs: 500 same-command pairs, contrast_pair_acc 97.60%.

## int8 deployment

TFLite input: `[1, C, 16]` int32, dynamic `C`. int8 weights, float
activations. `console_nn benchmark` records CPU latency and size.
`console_nn doctor --strict` verifies hashes, tokenizer/config, dynamic shape,
one TFLite invocation and a smoke classification.

## Evaluation caveats

- The 85-session real test is too small to separate the FCN from hashed
  n-gram logistic regression: 77/85 vs 75/85, McNemar `p = 0.625`.
- Domain-holdout runs use a reduced budget (4 x 250); the production CPU model
  uses 10 x 600. The full-data budget control differs by about 0.6 pp, so the
  77.34% holdout macro is dominated by domain shift.
- Risk-coverage is measured on synthetic validation (4489 documents), not on
  the 43 real validation sessions.
- The pre-clean corpus had exact train/test and train/val duplicates. The
  current mixed corpus is deduplicated, curated commands do not cross splits,
  and the model was retrained; `results/cpu/split_audit.json` reports zero
  exact and zero command-level cross-split overlap.

## Failure modes

- Domain shift from synthetic templates to real production logs.
- `unknown` is the hardest class (completed run with no recognizable marker,
  or truncated capture).
- Chunk-level neutral/output lines are noisier than document verdicts.
- Very large inputs are bounded by RAM, not by a fixed token limit.
- int8 can change borderline document predictions; use the float model or
  lower confidence thresholds for high-risk gates.

## Monitoring

Log `status`, `confidence`, `completed`, `n_chunks`, `backend`, model manifest
hash and latency. Alert on low-confidence rate, unknown-rate drift and
document/chunk disagreement.
