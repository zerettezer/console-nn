"""Contrast-pair evaluation: same command, different outcome labels."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .data import encode_split, load_documents, make_dataset
from .evaluate import collect_predictions, load_run, save_json


def build_contrast_pairs(
    documents: Sequence[Dict],
    max_pairs: int = 500,
) -> Tuple[List[Tuple[int, int]], List[Dict]]:
    """Pair documents that share a command but carry different labels.

    Returns pair indices and the selected documents in their original order.
    """
    groups: Dict[Tuple[str, str], Dict[str, List[int]]] = {}
    for index, document in enumerate(documents):
        command = str(document.get("command", "")).strip()
        if not command:
            continue
        key = (str(document.get("platform", "")), command)
        groups.setdefault(key, {}).setdefault(
            str(document.get("status", "")), []
        ).append(index)

    pairs: List[Tuple[int, int]] = []
    for key in sorted(groups):
        by_status = groups[key]
        if len(by_status) < 2:
            continue
        # One representative per outcome keeps the pairs balanced and avoids
        # over-weighting a command with many near-duplicate transcripts.
        representatives = {
            status: indexes[0] for status, indexes in sorted(by_status.items())
        }
        for left, right in itertools.combinations(sorted(representatives), 2):
            pairs.append((representatives[left], representatives[right]))
            if len(pairs) >= max_pairs:
                return pairs, list(documents)
    return pairs, list(documents)


def evaluate_contrast(
    run_dir: str,
    data_dir: str,
    split: str = "test",
    max_pairs: int = 500,
) -> Dict:
    model, tokenizer, cfg = load_run(run_dir)
    documents = load_documents(data_dir, split)
    pairs, ordered = build_contrast_pairs(documents, max_pairs=max_pairs)

    encoded = encode_split(ordered, tokenizer, cfg, stride=cfg.data.eval_stride)
    dataset = make_dataset(encoded, cfg, cfg.train.batch_size, shuffle=False)
    predictions = collect_predictions(
        model, dataset, completed_head=cfg.model.heads.completed_head
    )
    predicted = np.argmax(predictions["doc_prob"], axis=1)
    truth = predictions["doc_true"].astype(int)
    names = list(cfg.model.heads.class_names)

    details = []
    correct_pairs = 0
    changed_pairs = 0
    for left, right in pairs:
        left_ok = bool(predicted[left] == truth[left])
        right_ok = bool(predicted[right] == truth[right])
        correct = left_ok and right_ok
        changed = bool(predicted[left] != predicted[right])
        correct_pairs += int(correct)
        changed_pairs += int(changed)
        details.append(
            {
                "command": documents[left].get("command"),
                "platform": documents[left].get("platform"),
                "left": {
                    "id": documents[left].get("id"),
                    "true": names[truth[left]],
                    "predicted": names[predicted[left]],
                    "correct": left_ok,
                },
                "right": {
                    "id": documents[right].get("id"),
                    "true": names[truth[right]],
                    "predicted": names[predicted[right]],
                    "correct": right_ok,
                },
                "pair_correct": correct,
                "verdict_changed": changed,
            }
        )

    total = len(pairs)
    payload = {
        "run": str(run_dir),
        "data": data_dir,
        "split": split,
        "pairs": total,
        "contrast_pair_acc": round(correct_pairs / total, 5) if total else None,
        "verdict_change_rate": round(changed_pairs / total, 5) if total else None,
        "details": details,
    }
    return payload


def write_contrast(report: Dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_json(report, path)
