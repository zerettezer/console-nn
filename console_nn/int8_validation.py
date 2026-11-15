"""End-to-end validation of the int8 model against the float model."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .data import encode_document, load_documents
from .engine import ConsoleEngine
from .evaluate import (
    classification_metrics,
    load_run,
    resolve_model_dir,
    save_json,
)
from .tokenizer import NgramTokenizer


def _collect(engine, docs, tok: NgramTokenizer, cfg, names: List[str]) -> Dict:
    doc_prob, chunk_prob = [], []
    doc_true, chunk_true, comp_true = [], [], []
    doc_pred, chunk_pred, comp_pred = [], [], []
    started = time.perf_counter()
    for doc in docs:
        result = engine.analyze_lines([line["text"] for line in doc["lines"]])
        ids, labels, doc_label, completed = encode_document(doc, tok, cfg.data)
        valid = labels > 0
        true_chunks = labels[valid] - 1
        pred_chunks = np.array(
            [names.index(c["label"]) for c in result["chunks"]], dtype=np.int32
        )
        probs = np.array(
            [[c["probs"][name] for name in names] for c in result["chunks"]],
            dtype=np.float64,
        )
        if len(pred_chunks) != len(true_chunks):
            n = min(len(pred_chunks), len(true_chunks))
            pred_chunks, true_chunks, probs = (
                pred_chunks[:n], true_chunks[:n], probs[:n]
            )
        chunk_true.append(true_chunks)
        chunk_pred.append(pred_chunks)
        chunk_prob.append(probs)
        doc_true.append(names.index(doc["status"]))
        doc_pred.append(names.index(result["status"]))
        doc_prob.append([result["probs"][name] for name in names])
        comp_true.append(1 if doc.get("completed", True) else 0)
        comp_pred.append(1 if result["completed"] else 0)
    elapsed = time.perf_counter() - started
    return {
        "doc_true": np.asarray(doc_true),
        "doc_pred": np.asarray(doc_pred),
        "doc_prob": np.asarray(doc_prob),
        "chunk_true": np.concatenate(chunk_true) if chunk_true else np.array([], dtype=int),
        "chunk_pred": np.concatenate(chunk_pred) if chunk_pred else np.array([], dtype=int),
        "chunk_prob": np.concatenate(chunk_prob, axis=0) if chunk_prob else np.empty((0, 4)),
        "completed_true": np.asarray(comp_true),
        "completed_pred": np.asarray(comp_pred),
        "seconds": elapsed,
        "docs": len(docs),
    }


def _backend_metrics(collected: Dict, names: List[str]) -> Dict:
    doc = classification_metrics(collected["doc_true"], collected["doc_prob"], names)
    chunk = classification_metrics(
        collected["chunk_true"], collected["chunk_prob"], names
    )
    completed = {
        "accuracy": round(float(np.mean(
            collected["completed_true"] == collected["completed_pred"]
        )), 5),
        "n": int(len(collected["completed_true"])),
    }
    seconds = max(collected["seconds"], 1e-9)
    return {
        "doc": doc,
        "chunk": chunk,
        "completed": completed,
        "seconds": round(seconds, 3),
        "docs_per_second": round(collected["docs"] / seconds, 2),
    }


def _agreement(a: Dict, b: Dict) -> Dict:
    doc = float(np.mean(a["doc_pred"] == b["doc_pred"]))
    chunk = float(np.mean(a["chunk_pred"] == b["chunk_pred"])) if len(a["chunk_pred"]) else 0.0
    completed = float(np.mean(a["completed_pred"] == b["completed_pred"]))
    prob_delta = float(np.mean(np.abs(a["doc_prob"] - b["doc_prob"])))
    return {
        "doc": round(doc, 5),
        "chunk": round(chunk, 5),
        "completed": round(completed, 5),
        "mean_abs_doc_prob_diff": round(prob_delta, 6),
    }


def validate_run(
    run_dir: str,
    data_dir: str,
    split: str = "test",
    max_docs: int = 0,
    tflite: Optional[str] = None,
    save: bool = True,
    min_doc_accuracy: float = 0.85,
    max_accuracy_drop: float = 0.10,
) -> Dict:
    model, tok, cfg = load_run(run_dir)
    names = list(cfg.model.heads.class_names)
    docs = load_documents(data_dir, split)
    if max_docs and len(docs) > max_docs:
        step = max(1, len(docs) // max_docs)
        docs = docs[::step][:max_docs]
    model_dir = resolve_model_dir(run_dir)
    run = model_dir.parent if model_dir.name == "model" else model_dir
    if tflite is None:
        tflite = str(model_dir / "model_int8.tflite")
    if not Path(tflite).exists():
        raise FileNotFoundError(tflite)

    float_engine = ConsoleEngine(model, tok, cfg)
    int8_engine = ConsoleEngine(model, tok, cfg, tflite_path=tflite)
    float_data = _collect(float_engine, docs, tok, cfg, names)
    int8_data = _collect(int8_engine, docs, tok, cfg, names)
    float_metrics = _backend_metrics(float_data, names)
    int8_metrics = _backend_metrics(int8_data, names)
    delta = {
        "doc_accuracy": round(
            int8_metrics["doc"]["accuracy"] - float_metrics["doc"]["accuracy"], 5
        ),
        "chunk_accuracy": round(
            int8_metrics["chunk"]["accuracy"] - float_metrics["chunk"]["accuracy"], 5
        ),
        "completed_accuracy": round(
            int8_metrics["completed"]["accuracy"]
            - float_metrics["completed"]["accuracy"], 5
        ),
    }
    agreement = _agreement(float_data, int8_data)
    gates = {
        "doc_accuracy": int8_metrics["doc"]["accuracy"] >= min_doc_accuracy,
        "accuracy_drop": delta["doc_accuracy"] >= -max_accuracy_drop,
    }
    report = {
        "run": str(run),
        "model": str(model_dir),
        "tflite": str(tflite),
        "split": split,
        "documents": len(docs),
        "float": float_metrics,
        "int8": int8_metrics,
        "delta": delta,
        "agreement": agreement,
        "gates": {**gates, "ok": all(gates.values())},
        "thresholds": {
            "min_doc_accuracy": min_doc_accuracy,
            "max_accuracy_drop": max_accuracy_drop,
        },
    }
    if save:
        reports = run / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        save_json(report, str(reports / "validation_int8.json"))
        (reports / "validation_int8.md").write_text(
            render_markdown(report), encoding="utf-8"
        )
    return report


def render_markdown(report: Dict) -> str:
    f, q = report["float"], report["int8"]
    lines = [
        f"# int8 validation: {report['run']}",
        "",
        f"- split: `{report['split']}` ({report['documents']} documents)",
        f"- int8 model: `{report['tflite']}`",
        f"- gate: **{'PASS' if report['gates']['ok'] else 'FAIL'}**",
        "",
        "| metric | float | int8 | delta |",
        "|---|---:|---:|---:|",
        f"| doc accuracy | {f['doc']['accuracy']:.4f} | {q['doc']['accuracy']:.4f} | "
        f"{report['delta']['doc_accuracy']:+.4f} |",
        f"| doc macro F1 | {f['doc']['macro_f1']:.4f} | {q['doc']['macro_f1']:.4f} | "
        f"{q['doc']['macro_f1'] - f['doc']['macro_f1']:+.4f} |",
        f"| chunk accuracy | {f['chunk']['accuracy']:.4f} | {q['chunk']['accuracy']:.4f} | "
        f"{report['delta']['chunk_accuracy']:+.4f} |",
        f"| chunk macro F1 | {f['chunk']['macro_f1']:.4f} | {q['chunk']['macro_f1']:.4f} | "
        f"{q['chunk']['macro_f1'] - f['chunk']['macro_f1']:+.4f} |",
        f"| completion accuracy | {f['completed']['accuracy']:.4f} | "
        f"{q['completed']['accuracy']:.4f} | "
        f"{report['delta']['completed_accuracy']:+.4f} |",
        f"| docs/s | {f['docs_per_second']:.1f} | {q['docs_per_second']:.1f} | |",
        "",
        f"Agreement: doc `{report['agreement']['doc']:.4f}`, chunk "
        f"`{report['agreement']['chunk']:.4f}`, completion "
        f"`{report['agreement']['completed']:.4f}`; mean doc probability delta "
        f"`{report['agreement']['mean_abs_doc_prob_diff']:.6f}`.",
        "",
        "## Per-class int8",
        "",
        "| class | precision | recall | f1 | support |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in q["doc"]["per_class"].items():
        lines.append(
            f"| {name} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1']:.4f} | {m['support']} |"
        )
    lines.append("")
    return "\n".join(lines)
