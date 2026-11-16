"""Consolidate a run into one log, one metrics file, one benchmark and one PNG."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from .evaluate import resolve_model_dir


def _read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_history(path: Path) -> List[Dict]:
    if not path.exists():
        alt = path.with_suffix(".json")
        if alt.exists():
            try:
                payload = json.loads(alt.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    keys = list(payload)
                    return [
                        {"epoch": i + 1, **{k: payload[k][i] for k in keys
                                              if i < len(payload[k])}}
                        for i in range(len(payload.get("loss", [])))
                    ]
            except Exception:
                return []
        return []
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                rows.append({k: float(v) for k, v in row.items() if v not in ("", None)})
            except ValueError:
                pass
    return rows


def _fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_run_reports(run_dir: str) -> Dict:
    model_dir = resolve_model_dir(run_dir)
    run = model_dir.parent if model_dir.name == "model" else model_dir
    existing = _read_json(run / "metrics.json") or {}
    summary = (
        _read_json(run / "metrics" / "summary.json")
        or _read_json(run / "summary.json")
        or existing
        or {}
    )
    val = (
        _read_json(run / "metrics" / "val_metrics.json")
        or _read_json(run / "val_metrics.json")
        or existing.get("val", {})
        or {}
    )
    test = (
        _read_json(run / "metrics" / "test_metrics.json")
        or _read_json(run / "test_metrics.json")
        or existing.get("float", {})
        or {}
    )
    validation = (
        _read_json(run / "reports" / "validation_int8.json")
        or _read_json(run / "validation_int8.json")
        or existing.get("int8_validation", {})
        or {}
    )
    quant = (
        _read_json(run / "reports" / "report_int8.json")
        or _read_json(run / "quant.json")
        or existing.get("quant", {})
        or {}
    )
    speed = (
        _read_json(run / "metrics" / "speed.json")
        or _read_json(run / "benchmark.json")
        or {}
    )
    history = _read_history(run / "logs" / "history.csv") or existing.get("history", [])
    calibration = _read_json(run / "calibration.json") or existing.get("calibration")
    contrast = _read_json(run / "contrast.json") or existing.get("contrast")
    holdout = _read_json(run / "domain_holdout.json") or existing.get(
        "domain_holdout"
    )

    existing_training = existing.get("training", {})
    metrics = {
        "run": str(run),
        "model": summary.get("model", {}),
        "vocab_size": summary.get("vocab_size"),
        "training": {
            "epochs": summary.get("epochs_run", existing_training.get("epochs")),
            "seconds": summary.get("seconds", existing_training.get("seconds")),
        },
        "val": val,
        "float": test,
        "int8": validation.get("int8", {}),
        "int8_validation": validation,
        "quant": quant,
        "history": history,
        "calibration": calibration,
        "contrast_pair_acc": (
            contrast.get("contrast_pair_acc") if contrast else None
        ),
        "contrast": (
            {
                "pairs": contrast.get("pairs"),
                "contrast_pair_acc": contrast.get("contrast_pair_acc"),
                "verdict_change_rate": contrast.get("verdict_change_rate"),
            }
            if contrast else None
        ),
        "domain_holdout": (
            {
                "summary": holdout.get("summary"),
                "table": holdout.get("table"),
            }
            if holdout else None
        ),
    }
    run.mkdir(parents=True, exist_ok=True)
    (run / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    benchmark = {
        "run": str(run),
        "cpu": speed.get("cpu"),
        "threads": speed.get("threads"),
        "iterations": speed.get("iterations"),
        "window_chunks": speed.get("window_chunks", 128),
        "results": speed.get("results", []),
    }
    (run / "benchmark.json").write_text(
        json.dumps(benchmark, indent=2) + "\n", encoding="utf-8"
    )
    (run / "run.log").write_text(render_run_log(metrics, benchmark), encoding="utf-8")
    return {"metrics": metrics, "benchmark": benchmark}


def render_run_log(metrics: Dict, benchmark: Dict) -> str:
    m = metrics.get("model", {})
    t = metrics.get("training", {})
    lines = [
        f"run          : {metrics.get('run')}",
        f"params       : {m.get('params')}",
        f"fp32         : {m.get('approx_fp32_mb')} MB",
        f"int8         : {m.get('approx_int8_mb')} MB",
        f"vocab        : {metrics.get('vocab_size')}",
        f"epochs       : {t.get('epochs')}",
        f"train time   : {t.get('seconds')} s",
        "",
        "test metrics (float):",
        f"  doc accuracy      : {_fmt(metrics.get('float', {}).get('doc', {}).get('accuracy'))}",
        f"  doc macro F1      : {_fmt(metrics.get('float', {}).get('doc', {}).get('macro_f1'))}",
        f"  chunk accuracy    : {_fmt(metrics.get('float', {}).get('chunk', {}).get('accuracy'))}",
        f"  chunk macro F1    : {_fmt(metrics.get('float', {}).get('chunk', {}).get('macro_f1'))}",
        f"  completion acc    : {_fmt(metrics.get('float', {}).get('completed', {}).get('accuracy'))}",
        "",
        "test metrics (int8):",
        f"  doc accuracy      : {_fmt(metrics.get('int8', {}).get('doc', {}).get('accuracy'))}",
        f"  doc macro F1      : {_fmt(metrics.get('int8', {}).get('doc', {}).get('macro_f1'))}",
        f"  chunk accuracy    : {_fmt(metrics.get('int8', {}).get('chunk', {}).get('accuracy'))}",
        f"  chunk macro F1    : {_fmt(metrics.get('int8', {}).get('chunk', {}).get('macro_f1'))}",
        f"  completion acc    : {_fmt(metrics.get('int8', {}).get('completed', {}).get('accuracy'))}",
        f"  gate              : {'PASS' if metrics.get('int8_validation', {}).get('gates', {}).get('ok') else 'FAIL'}",
        "",
        "benchmark (CPU):",
    ]
    for row in benchmark.get("results", []):
        lines.append(
            f"  {row.get('mode', '-'):8s} {row.get('ms_per_window', 0):.3f} ms/window  "
            f"{row.get('tokens_per_second', 0):,} tok/s  {row.get('size_kb', 0):.1f} KB"
        )
    calibration = metrics.get("calibration")
    if calibration:
        lines.extend(
            [
                "",
                "calibration:",
                f"  temperature       : {_fmt(calibration.get('temperature'), 4)}",
                f"  ECE synth val     : "
                f"{_fmt(calibration.get('synth_val', {}).get('ece_after', {}).get('ece'))}",
                f"  ECE real val      : "
                f"{_fmt(calibration.get('real_val', {}).get('ece_after', {}).get('ece'))}",
            ]
        )
    if metrics.get("contrast_pair_acc") is not None:
        lines.append(
            f"contrast pair acc    : {_fmt(metrics.get('contrast_pair_acc'))}"
        )
    holdout = metrics.get("domain_holdout")
    if holdout:
        summary = holdout.get("summary", {})
        lines.append(
            f"domain holdout macro : {_fmt(summary.get('macro_doc_accuracy'))}"
        )
    if metrics.get("history"):
        lines.extend(["", "epochs:"])
        for row in metrics["history"]:
            epoch = int(row.get("epoch", 0)) if "epoch" in row else len(lines)
            lines.append(
                f"  {epoch:3d}  loss={_fmt(row.get('loss'), 3)}  "
                f"doc_acc={_fmt(row.get('console_fcn_2_doc_acc'), 3)}  "
                f"val_loss={_fmt(row.get('val_loss'), 3)}  "
                f"val_doc_acc={_fmt(row.get('val_console_fcn_2_doc_acc'), 3)}"
            )
    return "\n".join(lines) + "\n"


