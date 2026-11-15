"""Evaluation helpers: metrics, confusion matrices and run inspection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .config import Config
from .data import load_documents, make_dataset, encode_split
from .model import build_model, compile_model
from .tokenizer import NgramTokenizer


def collect_predictions(model, dataset, completed_head: bool = True):
    """Return dict of numpy arrays with labels and probabilities."""
    chunk_true, chunk_prob, doc_true, doc_prob = [], [], [], []
    comp_true, comp_prob = [], []
    for batch in dataset:
        ids, y = batch
        pred = model(ids, training=False)
        y_chunk = y["chunk_logits"].numpy()
        mask = y_chunk > 0
        chunk_true.append(y_chunk[mask] - 1)
        probs = tf.nn.softmax(pred["chunk_logits"], axis=-1).numpy()
        chunk_prob.append(probs.reshape(-1, probs.shape[-1])[mask.reshape(-1)])
        doc_true.append(y["doc_logits"].numpy())
        doc_prob.append(tf.nn.softmax(pred["doc_logits"], axis=-1).numpy())
        if completed_head and "completed_logits" in pred:
            comp_true.append(y["completed_logits"].numpy())
            comp_prob.append(tf.sigmoid(pred["completed_logits"]).numpy())
    out = {
        "chunk_true": np.concatenate(chunk_true) if chunk_true else np.array([]),
        "chunk_prob": np.concatenate(chunk_prob, axis=0) if chunk_prob else np.empty((0, 4)),
        "doc_true": np.concatenate(doc_true) if doc_true else np.array([]),
        "doc_prob": np.concatenate(doc_prob, axis=0) if doc_prob else np.empty((0, 4)),
    }
    if comp_true:
        out["completed_true"] = np.concatenate(comp_true)
        out["completed_prob"] = np.concatenate(comp_prob).reshape(-1)
    return out


def classification_metrics(y_true, prob, names: Sequence[str]) -> Dict:
    y_true = np.asarray(y_true).astype(int)
    if y_true.size == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "n": 0}
    y_pred = np.argmax(prob, axis=1)
    labels = list(range(len(names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {}
    for i, name in enumerate(names):
        per_class[name] = {
            "precision": round(float(precision[i]), 5),
            "recall": round(float(recall[i]), 5),
            "f1": round(float(f1[i]), 5),
            "support": int(support[i]),
        }
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 5),
        "macro_f1": round(float(np.mean(f1)), 5),
        "weighted_f1": round(
            float(np.average(f1, weights=np.maximum(support, 1))), 5
        ),
        "n": int(y_true.size),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": list(names),
    }


def evaluate_arrays(pred: Dict, class_names: Sequence[str]) -> Dict:
    result = {
        "doc": classification_metrics(pred["doc_true"], pred["doc_prob"], class_names),
        "chunk": classification_metrics(
            pred["chunk_true"], pred["chunk_prob"], class_names
        ),
    }
    if "completed_true" in pred and len(pred["completed_true"]):
        y = pred["completed_true"].astype(int)
        p = pred["completed_prob"]
        result["completed"] = {
            "accuracy": round(float(np.mean((p >= 0.5).astype(int) == y)), 5),
            "n": int(len(y)),
        }
        try:
            result["completed"]["roc_auc"] = round(float(roc_auc_score(y, p)), 5)
        except ValueError:
            result["completed"]["roc_auc"] = None
    return result


def evaluate_dataset(model, dataset, class_names: Sequence[str], completed=True) -> Dict:
    pred = collect_predictions(model, dataset, completed_head=completed)
    return evaluate_arrays(pred, class_names)


def resolve_model_dir(run_dir: str) -> Path:
    """Accept either a run folder or its self-contained ``model/`` folder."""
    run = Path(run_dir)
    if (run / "config.json").exists():
        return run
    if (run / "model" / "config.json").exists():
        return run / "model"
    return run


def load_run(run_dir: str, weights: Optional[str] = None) -> Tuple[tf.keras.Model, NgramTokenizer, Config]:
    run = resolve_model_dir(run_dir)
    with open(run / "config.json", "r", encoding="utf-8") as fh:
        from .config import nested_update

        cfg = Config()
        nested_update(cfg, json.load(fh))
    tok = NgramTokenizer.load(str(run / "tokenizer.json"))
    model = build_model(cfg, tok.vocab_size)
    if weights:
        weight_path = str(weights)
    else:
        candidates = [
            run / "weights.h5",
            run / "checkpoints" / "best.weights.h5",
            run / "checkpoints" / "weights.h5",
            run / "best.weights.h5",
        ]
        weight_path = next((str(p) for p in candidates if p.exists()), str(candidates[0]))
    model.load_weights(weight_path)
    compile_model(model, cfg)
    return model, tok, cfg


def evaluate_run(run_dir: str, data_dir: str, split: str = "test") -> Dict:
    model, tok, cfg = load_run(run_dir)
    docs = load_documents(data_dir, split)
    encoded = encode_split(docs, tok, cfg, stride=cfg.data.eval_stride)
    ds = make_dataset(encoded, cfg, cfg.train.batch_size, shuffle=False)
    return evaluate_dataset(
        model, ds, cfg.model.heads.class_names, cfg.model.heads.completed_head
    )


def save_json(obj: Dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")
