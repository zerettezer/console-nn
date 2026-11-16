"""Domain-holdout evaluation.

The synthetic train/val/test splits share a generator, so their scores say
little about a new CLI tool.  This module groups the 31 generator domains into
five families, trains the CPU model with one family removed, and measures the
model on documents from that unseen family only.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import encode_split, load_documents, make_dataset
from .evaluate import collect_predictions, load_run
from .synth import write_jsonl

DOMAIN_GROUPS: Dict[str, List[str]] = {
    "python": ["python", "pip", "pytest", "conda"],
    "build_package": [
        "build", "compile", "package", "package_update", "maven", "gradle",
        "dotnet",
    ],
    "system": [
        "shell", "systemd", "kernel", "journal", "filesystem", "network",
        "security",
    ],
    "data_infra": [
        "archive", "backup", "cloud", "data_process", "database", "docker",
        "git", "media", "redis", "postgres", "mysql", "nginx",
    ],
    "windows_misc": [
        "windows_cmd", "windows_update", "powershell", "real_errors",
        "errors", "commands", "multi", "multi_chain",
    ],
    "language_ecosystems": ["node", "npm", "cargo", "go", "ruby", "php"],
    "orchestration": [
        "kubectl", "helm", "terraform", "ansible", "github_actions",
        "gitlab_ci", "jenkins",
    ],
}

GROUP_NAMES = list(DOMAIN_GROUPS)
_DOMAIN_TO_GROUP = {
    domain: group
    for group, domains in DOMAIN_GROUPS.items()
    for domain in domains
}
# Real collector sessions use the plain domain name "system".
_DOMAIN_TO_GROUP.setdefault("system", "system")


def group_of(document: Dict) -> str:
    """Return the domain family for a document (unknown domains are ``other``)."""
    return _DOMAIN_TO_GROUP.get(str(document.get("domain", "")), "other")


def group_documents(documents: Sequence[Dict]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = {name: [] for name in GROUP_NAMES}
    for document in documents:
        name = group_of(document)
        if name in groups:
            groups[name].append(document)
    return groups


def prepare_fold(
    data_dir: str,
    out_dir: str,
    held_out: str,
) -> Dict[str, int]:
    """Write filtered train/val/test splits for one held-out group."""
    if held_out not in DOMAIN_GROUPS:
        raise ValueError(f"unknown group {held_out!r}")
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, int] = {}
    for split in ("train", "val", "test"):
        documents = load_documents(data_dir, split)
        if split == "test":
            selected = [d for d in documents if group_of(d) == held_out]
        else:
            selected = [d for d in documents if group_of(d) != held_out]
        write_jsonl(selected, str(target / f"{split}.jsonl"))
        stats[split] = len(selected)
    return stats


def _bootstrap_ci(values: np.ndarray, samples: int = 1000,
                  seed: int = 13) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(samples, values.size))
    means = values[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return (float(low), float(high))


def evaluate_documents(model, tokenizer, cfg, documents: Sequence[Dict]) -> Dict:
    encoded = encode_split(documents, tokenizer, cfg, stride=cfg.data.eval_stride)
    dataset = make_dataset(encoded, cfg, cfg.train.batch_size, shuffle=False)
    pred = collect_predictions(
        model, dataset, completed_head=cfg.model.heads.completed_head
    )
    doc_true = pred["doc_true"].astype(int)
    doc_pred = np.argmax(pred["doc_prob"], axis=1)
    doc_correct = (doc_pred == doc_true).astype(np.float64)
    completed_correct = None
    completed_accuracy = None
    if "completed_true" in pred and len(pred["completed_true"]):
        completed_true = pred["completed_true"].astype(int)
        completed_pred = (pred["completed_prob"] >= 0.5).astype(int)
        completed_correct = (completed_pred == completed_true).astype(np.float64)
        completed_accuracy = float(completed_correct.mean())
    return {
        "n": int(doc_true.size),
        "doc_accuracy": float(doc_correct.mean()),
        "completion_accuracy": completed_accuracy,
        "doc_correct": doc_correct,
        "completed_correct": completed_correct,
    }


def run_fold(
    config_path: str,
    data_dir: str,
    out_dir: str,
    held_out: str,
    seed: int,
    epochs: int,
    steps_per_epoch: int,
    validation_steps: int,
    bootstrap_samples: int = 1000,
) -> Dict:
    from .config import load_config
    from .train import train_model

    work = Path(out_dir)
    fold_data = work / f"data_{held_out}_seed{seed}"
    fold_run = work / f"run_{held_out}_seed{seed}"
    data_stats = prepare_fold(data_dir, str(fold_data), held_out)

    cfg = load_config(config_path)
    cfg.data.seed = int(seed)
    cfg.train.epochs = int(epochs)
    cfg.train.steps_per_epoch = int(steps_per_epoch)
    cfg.train.validation_steps = int(validation_steps)
    cfg.train.early_stopping_patience = 2
    cfg.train.reduce_lr_patience = 1
    cfg.train.verbose = 0
    summary = train_model(cfg, str(fold_data), str(fold_run), verbose=0)

    model, tokenizer, run_cfg = load_run(str(fold_run))
    documents = load_documents(str(fold_data), "test")
    measured = evaluate_documents(model, tokenizer, run_cfg, documents)
    doc_low, doc_high = _bootstrap_ci(
        measured["doc_correct"], samples=bootstrap_samples, seed=seed
    )
    comp_low = comp_high = None
    if measured["completed_correct"] is not None:
        comp_low, comp_high = _bootstrap_ci(
            measured["completed_correct"], samples=bootstrap_samples, seed=seed
        )

    result = {
        "group": held_out,
        "seed": int(seed),
        "data": data_stats,
        "train_seconds": summary.get("seconds"),
        "epochs": summary.get("epochs_run"),
        "documents": measured["n"],
        "doc_accuracy": round(measured["doc_accuracy"], 5),
        "doc_accuracy_ci95": [
            round(doc_low, 5), round(doc_high, 5)
        ],
        "completion_accuracy": (
            round(measured["completion_accuracy"], 5)
            if measured["completion_accuracy"] is not None else None
        ),
        "completion_accuracy_ci95": (
            [round(comp_low, 5), round(comp_high, 5)]
            if comp_low is not None else None
        ),
        "_doc_correct": measured["doc_correct"].tolist(),
        "_completed_correct": (
            measured["completed_correct"].tolist()
            if measured["completed_correct"] is not None else None
        ),
    }
    return result


def aggregate_folds(folds: Sequence[Dict], bootstrap_samples: int = 1000) -> Dict:
    """Combine seeds per group and build the final table."""
    by_group: Dict[str, List[Dict]] = {name: [] for name in GROUP_NAMES}
    for fold in folds:
        by_group.setdefault(fold["group"], []).append(fold)

    table = []
    for group in GROUP_NAMES:
        runs = by_group.get(group, [])
        if not runs:
            continue
        doc_accuracy = float(np.mean([r["doc_accuracy"] for r in runs]))
        comp_values = [
            r["completion_accuracy"] for r in runs
            if r["completion_accuracy"] is not None
        ]
        completion = float(np.mean(comp_values)) if comp_values else None
        doc_correct = np.mean(
            [np.asarray(r["_doc_correct"], dtype=np.float64) for r in runs],
            axis=0,
        )
        doc_low, doc_high = _bootstrap_ci(
            doc_correct, samples=bootstrap_samples, seed=17
        )
        comp_low = comp_high = None
        completion_correct = [
            np.asarray(r["_completed_correct"], dtype=np.float64)
            for r in runs
            if r["_completed_correct"] is not None
        ]
        if completion_correct:
            pooled = np.mean(completion_correct, axis=0)
            comp_low, comp_high = _bootstrap_ci(
                pooled, samples=bootstrap_samples, seed=19
            )
        table.append(
            {
                "group": group,
                "domains": DOMAIN_GROUPS[group],
                "seeds": [r["seed"] for r in runs],
                "documents": runs[0]["documents"],
                "doc_accuracy": round(doc_accuracy, 5),
                "doc_accuracy_ci95": [
                    round(doc_low, 5), round(doc_high, 5)
                ],
                "completion_accuracy": (
                    round(completion, 5) if completion is not None else None
                ),
                "completion_accuracy_ci95": (
                    [round(comp_low, 5), round(comp_high, 5)]
                    if comp_low is not None else None
                ),
                "seed_doc_accuracy_std": round(
                    float(np.std([r["doc_accuracy"] for r in runs])), 5
                ),
                "runs": [
                    {
                        key: value
                        for key, value in run.items()
                        if not key.startswith("_")
                    }
                    for run in runs
                ],
            }
        )

    covered = [row for row in table if row["documents"]]
    summary = {
        "groups": len(table),
        "macro_doc_accuracy": round(
            float(np.mean([r["doc_accuracy"] for r in covered])), 5
        )
        if covered else None,
        "weighted_doc_accuracy": round(
            float(
                np.average(
                    [r["doc_accuracy"] for r in covered],
                    weights=[r["documents"] for r in covered],
                )
            ),
            5,
        )
        if covered else None,
        "documents": int(sum(r["documents"] for r in covered)),
    }
    return {"table": table, "summary": summary}


def run_domain_holdout(
    config_path: str,
    data_dir: str,
    out_path: str,
    seeds: Sequence[int] = (1, 2, 3),
    epochs: int = 4,
    steps_per_epoch: int = 250,
    validation_steps: int = 60,
    bootstrap_samples: int = 1000,
    work_dir: Optional[str] = None,
    keep_runs: bool = False,
) -> Dict:
    from .train import set_seed

    out = Path(out_path)
    work = Path(work_dir) if work_dir else out.parent / "domain_holdout"
    work.mkdir(parents=True, exist_ok=True)
    folds: List[Dict] = []
    payload: Dict = {
        "config": config_path,
        "data": data_dir,
        "domain_groups": DOMAIN_GROUPS,
        "budget": {
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "validation_steps": validation_steps,
            "seeds": list(seeds),
            "bootstrap_samples": bootstrap_samples,
        },
        "folds": [],
    }

    for group in GROUP_NAMES:
        for seed in seeds:
            print(f"[holdout] group={group} seed={seed}", flush=True)
            set_seed(seed)
            fold = run_fold(
                config_path,
                data_dir,
                str(work),
                group,
                int(seed),
                epochs,
                steps_per_epoch,
                validation_steps,
                bootstrap_samples,
            )
            folds.append(fold)
            payload["folds"].append(
                {
                    key: value
                    for key, value in fold.items()
                    if not key.startswith("_")
                }
            )
            aggregated = aggregate_folds(folds, bootstrap_samples)
            payload.update(aggregated)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            if not keep_runs:
                shutil.rmtree(work / f"run_{group}_seed{seed}", ignore_errors=True)
                shutil.rmtree(work / f"data_{group}_seed{seed}", ignore_errors=True)

    return payload
