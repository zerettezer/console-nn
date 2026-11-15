"""INT8 / float16 TFLite export with a verification report."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

from .config import Config, save_config
from .data import encode_split, load_documents
from .engine import ConsoleEngine
from .evaluate import load_run, resolve_model_dir, save_json
from .viz import render_ascii


def _converter(model, chunk: int, cfg: Config):
    # Dynamic chunk dimension: one TFLite model accepts any document length.
    spec = tf.TensorSpec([1, None, chunk], tf.int32, name="tokens")

    @tf.function
    def serving(x):
        return model(x, training=False)

    concrete = serving.get_concrete_function(spec)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    if cfg.quant.mode == "float16":
        converter.target_spec.supported_types = [tf.float16]
    elif cfg.quant.mode != "int8":
        raise ValueError(cfg.quant.mode)
    if cfg.quant.denylisted_ops:
        try:
            converter.target_spec.denylisted_ops = cfg.quant.denylisted_ops
        except Exception:
            pass
    return converter


def _convert(converter):
    return converter.convert()


def quantize_run(
    run_dir: str,
    data_dir: str,
    mode: Optional[str] = None,
    out_path: Optional[str] = None,
    cfg: Optional[Config] = None,
) -> Dict:
    model_dir = resolve_model_dir(run_dir)
    run = model_dir.parent if model_dir.name == "model" else model_dir
    reports_dir = run / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    model, tok, run_cfg = load_run(run_dir)
    cfg = cfg or run_cfg
    if mode:
        cfg.quant.mode = mode
    mode = cfg.quant.mode
    if mode == "none":
        raise ValueError("quant.mode is none; nothing to export")
    chunk = int(cfg.model.chunk_size)
    out = Path(out_path) if out_path else model_dir / f"model_{mode}.tflite"

    converter = _converter(model, chunk, cfg)
    t0 = time.time()
    blob = _convert(converter)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)

    interpreter = tf.lite.Interpreter(model_path=str(out))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_details = [
        {"name": d["name"], "shape": d["shape"].tolist(), "dtype": str(d["dtype"])}
        for d in interpreter.get_output_details()
    ]
    report = {
        "mode": mode,
        "path": str(out),
        "size_bytes": len(blob),
        "size_kb": round(len(blob) / 1024, 1),
        "seconds": round(time.time() - t0, 2),
        "input": {
            "name": input_detail["name"],
            "shape": input_detail["shape"].tolist(),
            "dtype": str(input_detail["dtype"]),
            "quantization": list(input_detail["quantization"]),
        },
        "outputs": output_details,
        "dynamic_chunks": True,
    }
    print(f"[quant] wrote {out} ({report['size_kb']} KB)")

    if cfg.quant.verify:
        report["verification"] = verify(
            run_dir, str(out), data_dir, cfg, max_docs=cfg.quant.verify_samples
        )
        print(json.dumps(report["verification"], indent=2))
    save_json(report, str(reports_dir / f"report_{mode}.json"))
    return report


def verify(
    run_dir: str,
    tflite_path: str,
    data_dir: str,
    cfg: Config,
    max_docs: int = 64,
) -> Dict:
    model, tok, cfg = load_run(run_dir)
    docs = load_documents(data_dir, "test")
    # keep the evaluation deterministic and cheap
    if max_docs and len(docs) > max_docs:
        step = max(1, len(docs) // max_docs)
        docs = docs[::step][:max_docs]
    # max_docs <= 0 means the full split

    float_engine = ConsoleEngine(model, tok, cfg)
    quant_engine = ConsoleEngine(model, tok, cfg, tflite_path=tflite_path)
    statuses = list(cfg.model.heads.class_names)
    y_true, y_float, y_quant = [], [], []
    prob_diffs = []
    chunk_agree, chunk_total = 0, 0
    comp_true, comp_float, comp_quant = [], [], []
    examples = []
    for doc in docs:
        lines = [line["text"] for line in doc["lines"]]
        a = float_engine.analyze_lines(lines)
        b = quant_engine.analyze_lines(lines)
        y_true.append(statuses.index(doc["status"]))
        y_float.append(statuses.index(a["status"]))
        y_quant.append(statuses.index(b["status"]))
        comp_true.append(1 if doc.get("completed", True) else 0)
        comp_float.append(1 if a["completed"] else 0)
        comp_quant.append(1 if b["completed"] else 0)
        prob_diffs.append(
            float(np.mean(np.abs(
                np.array(list(a["probs"].values())) - np.array(list(b["probs"].values()))
            )))
        )
        n = min(len(a["chunks"]), len(b["chunks"]))
        for i in range(n):
            chunk_agree += int(a["chunks"][i]["label"] == b["chunks"][i]["label"])
            chunk_total += 1
        if len(examples) < 3:
            examples.append(
                {
                    "true": doc["status"],
                    "float": a["status"],
                    "int8": b["status"],
                    "float_completed": a["completed"],
                    "int8_completed": b["completed"],
                }
            )
    y_true_a = np.asarray(y_true)
    y_float_a = np.asarray(y_float)
    y_quant_a = np.asarray(y_quant)
    report = {
        "docs": len(docs),
        "float_doc_accuracy": round(float(np.mean(y_float_a == y_true_a)), 5),
        "quant_doc_accuracy": round(float(np.mean(y_quant_a == y_true_a)), 5),
        "doc_agreement": round(float(np.mean(y_float_a == y_quant_a)), 5),
        "chunk_agreement": round(chunk_agree / max(1, chunk_total), 5),
        "mean_abs_prob_diff": round(float(np.mean(prob_diffs)), 5),
        "completed_agreement": round(
            float(np.mean(np.asarray(comp_float) == np.asarray(comp_quant))), 5
        ),
        "completed_accuracy_float": round(
            float(np.mean(np.asarray(comp_float) == np.asarray(comp_true))), 5
        ),
        "completed_accuracy_quant": round(
            float(np.mean(np.asarray(comp_quant) == np.asarray(comp_true))), 5
        ),
        "examples": examples,
    }
    drop = report["float_doc_accuracy"] - report["quant_doc_accuracy"]
    report["accuracy_drop"] = round(drop, 5)
    report["ok"] = bool(drop <= cfg.quant.verify_tolerance)
    return report
