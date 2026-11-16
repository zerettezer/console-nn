#!/usr/bin/env python
"""Train CPU models with one domain family held out and report bootstrap CIs.

Example::

    python tools/eval_domain_holdout.py \
        --data data/mixed \
        --config configs/cpu.json \
        --out artifacts/cpu/domain_holdout.json

The default budget is deliberately small: five folds times three seeds is
fifteen training runs.  Increase ``--epochs`` and ``--steps`` when compute
allows; every run records the exact budget in the output JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from console_nn.domain_holdout import GROUP_NAMES, run_domain_holdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/mixed")
    parser.add_argument("--config", default="configs/cpu.json")
    parser.add_argument("--out", default="artifacts/cpu/domain_holdout.json")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=250,
                        dest="steps_per_epoch")
    parser.add_argument("--validation-steps", type=int, default=60)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--keep-runs", action="store_true")
    parser.add_argument("--groups", default=",".join(GROUP_NAMES),
                        help="comma-separated subset of groups")
    args = parser.parse_args()

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    unknown = [g for g in groups if g not in GROUP_NAMES]
    if unknown:
        parser.error(f"unknown groups: {unknown}")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    from console_nn.domain_holdout import run_domain_holdout as run

    # The runner always walks GROUP_NAMES; a custom subset is applied by
    # temporarily narrowing the module-level order.
    if groups != GROUP_NAMES:
        import console_nn.domain_holdout as module

        module.GROUP_NAMES = groups

    payload = run(
        args.config,
        args.data,
        args.out,
        seeds=seeds,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
        bootstrap_samples=args.bootstrap,
        work_dir=args.work_dir,
        keep_runs=args.keep_runs,
    )
    print(json.dumps(payload.get("summary", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
