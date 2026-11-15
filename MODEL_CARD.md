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

- 30,000 synthetic documents (24k/3k/3k).
- 80 real Linux sessions collected by `tools/collect_real.py` (safe local
  commands in a temp workspace), plus 23 concrete example sessions.
- The 23 examples are held out in the real test split.
- 20 complex held-out command families and 20+ error families.
- Train/val/test use separate seeds, context lengths and command pools.
- Real test sessions are held out from their own split and are also reportable
  with `console_nn eval --data data/real --split test`.

## Metrics

See `artifacts/<variant>/metrics.json` and `run.log`. Metrics are split into
float and int8, including per-class precision/recall/F1, confusion matrix,
chunk accuracy, completion accuracy, agreement and probability delta.
`console_nn validate --strict` enforces the int8 gate.

Real held-out test (`data/real`, 39 sessions): document accuracy 94.87%,
document macro F1 70.67%, chunk accuracy 58.22%, completion accuracy 84.62%.
The split contains no `unknown` examples.

## int8 deployment

TFLite input: `[1, C, 16]` int32, dynamic `C`. int8 weights, float
activations. `console_nn benchmark` records CPU latency and size.
`console_nn doctor --strict` verifies hashes, tokenizer/config, dynamic shape,
one TFLite invocation and a smoke classification.

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
