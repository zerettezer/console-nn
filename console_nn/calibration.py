"""Temperature calibration and selective-risk analysis for the document head."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

from .config import save_config
from .data import encode_split, load_documents, make_dataset
from .evaluate import load_run, resolve_model_dir, save_json


def collect_doc_logits(model, dataset) -> Tuple[np.ndarray, np.ndarray]:
    logits: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for ids, target in dataset:
        prediction = model(ids, training=False)
        logits.append(prediction["doc_logits"].numpy())
        labels.append(target["doc_logits"].numpy())
    if not logits:
        return np.empty((0, 4)), np.empty((0,), dtype=int)
    return np.concatenate(logits, axis=0), np.concatenate(labels, axis=0)


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - np.max(values, axis=-1, keepdims=True)
    exp = np.exp(values)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def negative_log_likelihood(temperature: float, logits: np.ndarray,
                            labels: np.ndarray) -> float:
    scaled = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    log_sum = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    log_probs = shifted - log_sum
    return float(-np.mean(log_probs[np.arange(len(labels)), labels]))


def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                    bounds: Tuple[float, float] = (0.05, 20.0)) -> Dict:
    if len(labels) == 0:
        return {"temperature": 1.0, "nll_before": None, "nll_after": None}
    from scipy.optimize import minimize_scalar

    before = negative_log_likelihood(1.0, logits, labels)
    result = minimize_scalar(
        lambda t: negative_log_likelihood(t, logits, labels),
        bounds=bounds,
        method="bounded",
    )
    temperature = float(result.x)
    return {
        "temperature": round(temperature, 6),
        "nll_before": round(before, 6),
        "nll_after": round(
            negative_log_likelihood(temperature, logits, labels), 6
        ),
        "n": int(len(labels)),
    }


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 15) -> Dict:
    if len(labels) == 0:
        return {"ece": None, "n": 0}
    confidence = np.max(probs, axis=1)
    prediction = np.argmax(probs, axis=1)
    correct = (prediction == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        count = int(mask.sum())
        if count == 0:
            continue
        accuracy = float(correct[mask].mean())
        mean_confidence = float(confidence[mask].mean())
        ece += count / len(labels) * abs(accuracy - mean_confidence)
        bins.append(
            {
                "range": [round(float(low), 3), round(float(high), 3)],
                "count": count,
                "accuracy": round(accuracy, 5),
                "confidence": round(mean_confidence, 5),
            }
        )
    return {
        "ece": round(float(ece), 5),
        "n": int(len(labels)),
        "bins": bins,
    }


def risk_coverage(probs: np.ndarray, labels: np.ndarray,
                  class_names: Sequence[str], steps: int = 21) -> Dict:
    """Selective risk when ``unknown`` predictions are rejected."""
    if len(labels) == 0:
        return {"curve": [], "n": 0}
    names = list(class_names)
    unknown_index = names.index("unknown") if "unknown" in names else None
    prediction = np.argmax(probs, axis=1)
    correct = (prediction == labels).astype(np.float64)
    curve = []
    for threshold in np.linspace(0.0, 1.0, steps):
        if unknown_index is None:
            covered = np.ones_like(correct, dtype=bool)
        else:
            covered = probs[:, unknown_index] < threshold
        coverage = float(covered.mean())
        if covered.any():
            risk = float(1.0 - correct[covered].mean())
            rejected = ~covered
            unknown_precision = (
                float(np.mean(labels[rejected] == unknown_index))
                if rejected.any() and unknown_index is not None else None
            )
            curve.append(
                {
                    "threshold": round(float(threshold), 3),
                    "coverage": round(coverage, 5),
                    "risk": round(risk, 5),
                    "unknown_precision": (
                        round(unknown_precision, 5)
                        if unknown_precision is not None else None
                    ),
                }
            )
        else:
            curve.append(
                {
                    "threshold": round(float(threshold), 3),
                    "coverage": 0.0,
                    "risk": None,
                    "unknown_precision": None,
                }
            )
    return {"curve": curve, "n": int(len(labels)), "class_names": names}


def _dataset(docs, tokenizer, cfg):
    encoded = encode_split(docs, tokenizer, cfg, stride=cfg.data.eval_stride)
    return make_dataset(encoded, cfg, cfg.train.batch_size, shuffle=False)


def _synthetic_only(documents):
    synthetic = [
        d for d in documents
        if str(d.get("source", "")).startswith("synthetic")
    ]
    return synthetic or list(documents)


def _scale_doc_head(model, factor: float) -> None:
    root = model.get_layer("console_fcn")
    kernel, bias = root.doc_head.get_weights()
    root.doc_head.set_weights([kernel * factor, bias * factor])


def _counts(probs: np.ndarray, labels: np.ndarray, names: Sequence[str]) -> Dict:
    prediction = np.argmax(probs, axis=1)
    return {
        "accuracy": round(float(np.mean(prediction == labels)), 5)
        if len(labels) else None,
        "n": int(len(labels)),
    }


def calibrate_run(
    run_dir: str,
    data_dir: str,
    real_dir: str = "data/real",
    split: str = "val",
    apply: bool = True,
    requantize: bool = False,
    validate: bool = False,
    benchmark: bool = False,
) -> Dict:
    model, tokenizer, cfg = load_run(run_dir)
    names = list(cfg.model.heads.class_names)
    model_dir = resolve_model_dir(run_dir)
    run = model_dir.parent if model_dir.name == "model" else model_dir
    benchmark_before = None
    if benchmark and (model_dir / "model_int8.tflite").exists():
        from .benchmark import benchmark_run

        benchmark_before = benchmark_run(str(run), iters=100, save=False)

    documents = _synthetic_only(load_documents(data_dir, split))
    dataset = _dataset(documents, tokenizer, cfg)
    logits, labels = collect_doc_logits(model, dataset)

    current_temperature = float(getattr(cfg.inference, "doc_temperature", 1.0))
    if apply and current_temperature != 1.0:
        # Restore raw logits before fitting a new temperature.
        _scale_doc_head(model, current_temperature)
        logits, labels = collect_doc_logits(model, dataset)

    fitted = fit_temperature(logits, labels)

    real_split = "val"
    try:
        real_documents = load_documents(real_dir, real_split)
    except FileNotFoundError:
        real_documents = []
    real_logits, real_labels = (
        collect_doc_logits(model, _dataset(real_documents, tokenizer, cfg))
        if real_documents else (np.empty((0, 4)), np.empty((0,), dtype=int))
    )

    before_synth = softmax(logits)
    before_real = softmax(real_logits) if len(real_labels) else np.empty((0, 4))
    report = {
        "run": str(run),
        "data": {"synth_val": data_dir, "real_val": real_dir},
        "temperature": fitted["temperature"],
        "fit": fitted,
        "applied": bool(apply),
        "previous_temperature": current_temperature,
        "synth_val": {
            "before": _counts(before_synth, labels, names),
            "ece_before": expected_calibration_error(before_synth, labels),
            "ece_after": expected_calibration_error(
                softmax(logits / fitted["temperature"]), labels
            ),
        },
        "real_val": {
            "before": _counts(before_real, real_labels, names),
            "ece_before": expected_calibration_error(before_real, real_labels),
            "ece_after": expected_calibration_error(
                softmax(real_logits / fitted["temperature"]), real_labels
            ),
        },
        "risk_coverage": risk_coverage(
            softmax(logits / fitted["temperature"]), labels, names
        ),
    }

    if apply:
        _scale_doc_head(model, 1.0 / fitted["temperature"])
        model.save_weights(str(model_dir / "weights.h5"))
        cfg.inference.doc_temperature = fitted["temperature"]
        save_config(cfg, str(model_dir / "config.json"))
        report["applied_temperature"] = fitted["temperature"]
        report["weights"] = str(model_dir / "weights.h5")

    if requantize:
        from .quantize import _op_names, quantize_run

        tflite_path = model_dir / "model_int8.tflite"
        ops_before = _op_names(str(tflite_path)) if tflite_path.exists() else None
        export_cfg = cfg
        export_cfg.quant.verify = False
        quantize_run(str(run), data_dir, cfg=export_cfg)
        ops_after = _op_names(str(tflite_path))
        report["int8_graph"] = {
            "op_count_before": len(ops_before) if ops_before is not None else None,
            "op_count_after": len(ops_after),
            "ops_unchanged": ops_before == ops_after,
            "ops": ops_after,
        }
        if benchmark_before is not None:
            from .benchmark import benchmark_run

            benchmark_after = benchmark_run(str(run), iters=100, save=False)
            before_ms = benchmark_before["results"][0]["ms_per_window"]
            after_ms = benchmark_after["results"][0]["ms_per_window"]
            report["benchmark"] = {
                "before_ms_per_window": before_ms,
                "after_ms_per_window": after_ms,
                "delta_ms": round(after_ms - before_ms, 4),
                "before": benchmark_before,
                "after": benchmark_after,
            }

    if validate:
        from .int8_validation import validate_run

        report["int8_validation"] = validate_run(
            str(run), data_dir, split="test", save=True
        )

    save_json(report, str(run / "calibration.json"))
    return report
