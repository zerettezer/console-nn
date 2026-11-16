"""Simple baselines and one-change ablations for the console classifier."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import HashingVectorizer

from .data import load_documents
from .evaluate import classification_metrics, evaluate_documents, load_run

BLOCK_PATTERNS = [
    r"\bpermission denied\b",
    r"\baccess is denied\b",
    r"\boperation not permitted\b",
    r"\bexternally-managed-environment\b",
    r"\bmust be run as root\b",
    r"\brequires? (?:sudo|root|administrator)\b",
    r"\bunauthori[sz]ed\b",
    r"\bauthentication failed\b",
    r"\bforbidden\b",
    r"\bblocked\b",
    r"\bpolicy (?:denied|violation)\b",
]
FAIL_PATTERNS = [
    r"\btraceback \(most recent call last\)\b",
    r"\b[a-z_]*error\b",
    r"\bexception\b",
    r"\bfatal\b",
    r"\bfailed\b",
    r"\bfailure\b",
    r"\bno such file or directory\b",
    r"\bnot found\b",
    r"\bunable to\b",
    r"\bcannot \w+",
    r"\bcommand not found\b",
]
UNKNOWN_PATTERNS = [
    r"\bunknown\b",
    r"\bunclear\b",
    r"\binterrupted\b",
    r"\bcancelled\b",
]

_BLOCK_RE = re.compile("|".join(BLOCK_PATTERNS), re.IGNORECASE)
_FAIL_RE = re.compile("|".join(FAIL_PATTERNS), re.IGNORECASE)
_UNKNOWN_RE = re.compile("|".join(UNKNOWN_PATTERNS), re.IGNORECASE)


def document_text(document: Dict) -> str:
    return "\n".join(line.get("text", "") for line in document.get("lines", []))


def regex_status(text: str) -> str:
    """Keyword baseline: block before fail, unknown before success."""
    if _BLOCK_RE.search(text):
        return "block"
    if _FAIL_RE.search(text):
        return "fail"
    if _UNKNOWN_RE.search(text):
        return "unknown"
    return "success"


def _as_probabilities(predictions: Sequence[int], n_classes: int) -> np.ndarray:
    probs = np.zeros((len(predictions), n_classes), dtype=np.float64)
    for row, index in enumerate(predictions):
        probs[row, int(index)] = 1.0
    return probs


def evaluate_regex(documents: Sequence[Dict], class_names: Sequence[str]) -> Dict:
    status_to_index = {name: i for i, name in enumerate(class_names)}
    truth = np.asarray(
        [status_to_index.get(str(d["status"]), len(class_names) - 1)
         for d in documents],
        dtype=int,
    )
    predicted = np.asarray(
        [status_to_index.get(regex_status(document_text(d)), len(class_names) - 1)
         for d in documents],
        dtype=int,
    )
    metrics = classification_metrics(
        truth, _as_probabilities(predicted, len(class_names)), class_names
    )
    metrics["params"] = 0
    return metrics


class HashedNgramBaseline:
    """Logistic regression over hashed word 1..3-grams (Keras)."""

    def __init__(self, buckets: int = 8192, seed: int = 1):
        self.buckets = int(buckets)
        self.seed = int(seed)
        self.vectorizer = HashingVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            n_features=self.buckets,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            token_pattern=r"[a-z0-9_./:\\-]+",
        )
        tf.keras.utils.set_random_seed(self.seed)
        self.model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(self.buckets,), dtype=tf.float32),
                tf.keras.layers.Dense(4, activation="softmax"),
            ]
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def fit(self, documents: Sequence[Dict], labels: np.ndarray,
            epochs: int = 8, batch_size: int = 64):
        matrix = self.vectorizer.transform([document_text(d) for d in documents])

        def rows():
            for index in range(matrix.shape[0]):
                yield (
                    matrix.getrow(index).toarray()[0].astype(np.float32),
                    np.int32(labels[index]),
                )

        dataset = tf.data.Dataset.from_generator(
            rows,
            output_signature=(
                tf.TensorSpec(shape=(self.buckets,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        self.model.fit(dataset, epochs=epochs, verbose=0)
        return self

    def predict(self, documents: Sequence[Dict]) -> np.ndarray:
        matrix = self.vectorizer.transform([document_text(d) for d in documents])
        chunks = []
        for start in range(0, matrix.shape[0], 256):
            batch = matrix[start : start + 256].toarray().astype(np.float32)
            chunks.append(self.model.predict(batch, verbose=0))
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 4))


def _labels(documents: Sequence[Dict], class_names: Sequence[str]) -> np.ndarray:
    mapping = {name: i for i, name in enumerate(class_names)}
    return np.asarray(
        [mapping.get(str(d["status"]), len(class_names) - 1) for d in documents],
        dtype=int,
    )


def synthetic_test_documents(data_dir: str) -> List[Dict]:
    """Test documents generated by `console_nn.synth` only."""
    documents = load_documents(data_dir, "test")
    synthetic = [
        d for d in documents
        if str(d.get("source", "")).startswith("synthetic")
    ]
    return synthetic or documents


def run_baselines(
    data_dir: str,
    real_dir: str,
    class_names: Sequence[str],
    seeds: Sequence[int] = (1, 2, 3),
    buckets: int = 8192,
    epochs: int = 8,
) -> Dict:
    synth_train = load_documents(data_dir, "train")
    synth_test = synthetic_test_documents(data_dir)
    real_test = load_documents(real_dir, "test")
    names = list(class_names)
    train_labels = _labels(synth_train, names)
    synth_labels = _labels(synth_test, names)
    real_labels = _labels(real_test, names)

    regex = {
        "name": "regex",
        "params": 0,
        "synth_test": evaluate_regex(synth_test, names),
        "real_test": evaluate_regex(real_test, names),
    }

    runs = []
    for seed in seeds:
        print(f"[baseline] hashed n-grams seed={seed}", flush=True)
        baseline = HashedNgramBaseline(buckets=buckets, seed=int(seed))
        baseline.fit(synth_train, train_labels, epochs=epochs)
        synth_prob = baseline.predict(synth_test)
        real_prob = baseline.predict(real_test)
        runs.append(
            {
                "seed": int(seed),
                "params": int(baseline.model.count_params()),
                "synth_test": classification_metrics(
                    synth_labels, synth_prob, names
                ),
                "real_test": classification_metrics(real_labels, real_prob, names),
            }
        )

    def _mean(section: str, key: str) -> float:
        values = [r[section][key] for r in runs]
        return round(float(np.mean(values)), 5)

    hashed = {
        "name": "hashed_ngram_lr",
        "buckets": int(buckets),
        "params": runs[0]["params"] if runs else 0,
        "synth_test": {
            "accuracy_mean": _mean("synth_test", "accuracy"),
            "accuracy_std": round(
                float(np.std([r["synth_test"]["accuracy"] for r in runs])), 5
            ),
            "macro_f1_mean": _mean("synth_test", "macro_f1"),
            "macro_f1_std": round(
                float(np.std([r["synth_test"]["macro_f1"] for r in runs])), 5
            ),
        },
        "real_test": {
            "accuracy_mean": _mean("real_test", "accuracy"),
            "accuracy_std": round(
                float(np.std([r["real_test"]["accuracy"] for r in runs])), 5
            ),
            "macro_f1_mean": _mean("real_test", "macro_f1"),
            "macro_f1_std": round(
                float(np.std([r["real_test"]["macro_f1"] for r in runs])), 5
            ),
        },
        "runs": runs,
    }
    return {
        "data": {
            "train": data_dir,
            "synth_test": "source==synthetic in " + data_dir + "/test.jsonl",
            "real_test": real_dir,
        },
        "classes": names,
        "regex": regex,
        "hashed_ngram_lr": hashed,
    }


def _variant_config(base_config: str, name: str):
    from .config import load_config

    cfg = load_config(base_config)
    if name == "cpu_control":
        pass
    elif name == "no_digit_tokens":
        cfg.tokenizer.numeric_mode = "mask"
    elif name == "no_factorized_embedding":
        cfg.model.embedding.factorized = False
        cfg.model.embedding.embedding_dim = cfg.model.embedding.hidden_dim
    else:
        raise ValueError(name)
    return cfg


def _evaluate_splits(
    run_dir: str,
    synth_documents: Sequence[Dict],
    real_documents: Sequence[Dict],
) -> Dict:
    model, tokenizer, cfg = load_run(run_dir)
    return {
        "synth": evaluate_documents(model, tokenizer, cfg, synth_documents),
        "real": evaluate_documents(model, tokenizer, cfg, real_documents),
    }


def run_ablations(
    base_config: str,
    data_dir: str,
    real_dir: str,
    out_path: str,
    seeds: Sequence[int] = (1, 2, 3),
    epochs: int = 4,
    steps_per_epoch: int = 250,
    validation_steps: int = 60,
    variants: Sequence[str] = (
        "cpu_control",
        "no_digit_tokens",
        "no_factorized_embedding",
    ),
) -> Dict:
    from .train import set_seed, train_model

    target = Path(out_path)
    work_dir = target.parent / "ablation_runs"
    synth_documents = synthetic_test_documents(data_dir)
    real_documents = load_documents(real_dir, "test")
    payload: Dict = {
        "base_config": base_config,
        "data": {
            "train": data_dir,
            "synth_test": "synthetic documents from " + data_dir + "/test.jsonl",
            "real_test": real_dir,
        },
        "budget": {
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "validation_steps": validation_steps,
            "seeds": list(seeds),
        },
        "variants": [],
    }
    rows = []
    for variant in variants:
        runs = []
        for seed in seeds:
            print(f"[ablate] {variant} seed={seed}", flush=True)
            cfg = _variant_config(base_config, variant)
            cfg.data.seed = int(seed)
            cfg.train.epochs = int(epochs)
            cfg.train.steps_per_epoch = int(steps_per_epoch)
            cfg.train.validation_steps = int(validation_steps)
            cfg.train.early_stopping_patience = 2
            cfg.train.reduce_lr_patience = 1
            cfg.train.verbose = 0
            run_dir = work_dir / f"{variant}_seed{seed}"
            set_seed(seed)
            summary = train_model(cfg, data_dir, str(run_dir), verbose=0)
            measured = _evaluate_splits(
                str(run_dir), synth_documents, real_documents
            )
            runs.append(
                {
                    "seed": int(seed),
                    "params": summary.get("model", {}).get("params"),
                    "synth_test": {
                        "accuracy": measured["synth"]["doc"]["accuracy"],
                        "macro_f1": measured["synth"]["doc"]["macro_f1"],
                        "chunk_accuracy": measured["synth"]["chunk"]["accuracy"],
                    },
                    "real_test": {
                        "accuracy": measured["real"]["doc"]["accuracy"],
                        "macro_f1": measured["real"]["doc"]["macro_f1"],
                        "chunk_accuracy": measured["real"]["chunk"]["accuracy"],
                    },
                }
            )
        rows.append(
            {
                "name": variant,
                "params": runs[0]["params"] if runs else None,
                "synth_test": {
                    "accuracy_mean": round(
                        float(np.mean([r["synth_test"]["accuracy"] for r in runs])), 5
                    ),
                    "accuracy_std": round(
                        float(np.std([r["synth_test"]["accuracy"] for r in runs])), 5
                    ),
                    "macro_f1_mean": round(
                        float(np.mean([r["synth_test"]["macro_f1"] for r in runs])), 5
                    ),
                },
                "real_test": {
                    "accuracy_mean": round(
                        float(np.mean([r["real_test"]["accuracy"] for r in runs])), 5
                    ),
                    "accuracy_std": round(
                        float(np.std([r["real_test"]["accuracy"] for r in runs])), 5
                    ),
                    "macro_f1_mean": round(
                        float(np.mean([r["real_test"]["macro_f1"] for r in runs])), 5
                    ),
                },
                "runs": runs,
            }
        )
        payload["variants"] = rows
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
