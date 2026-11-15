"""Deployment checks and model manifests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

from .evaluate import load_run, resolve_model_dir


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_files(model_dir: Path) -> Dict[str, Path]:
    return {
        "config.json": model_dir / "config.json",
        "tokenizer.json": model_dir / "tokenizer.json",
        "weights.h5": model_dir / "weights.h5",
        "model_int8.tflite": model_dir / "model_int8.tflite",
    }


def write_manifest(run_dir: str, extra: Optional[Dict] = None) -> Dict:
    model_dir = resolve_model_dir(run_dir)
    run = model_dir.parent if model_dir.name == "model" else model_dir
    files = model_files(model_dir)
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing model files: {missing}")
    manifest = {
        "format": 1,
        "model_dir": str(model_dir),
        "files": {
            name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for name, path in files.items()
        },
        "notes": extra or {},
    }
    path = model_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def doctor(run_dir: str, data_dir: Optional[str] = None) -> Dict:
    model_dir = resolve_model_dir(run_dir)
    run = model_dir.parent if model_dir.name == "model" else model_dir
    checks = []

    def add(name: str, ok: bool, detail: str = ""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    files = model_files(model_dir)
    add("model_folder", model_dir.is_dir(), str(model_dir))
    for name, path in files.items():
        add(f"file:{name}", path.exists(), str(path))

    manifest_path = model_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for name, meta in manifest.get("files", {}).items():
                p = model_dir / name
                add(
                    f"sha256:{name}",
                    p.exists() and sha256(p) == meta.get("sha256"),
                    str(p),
                )
        except Exception as exc:
            add("manifest", False, f"{type(exc).__name__}: {exc}")
    else:
        add("manifest", False, "manifest.json missing; run `manifest`")

    try:
        model, tok, cfg = load_run(run_dir)
        add("load_run", True, f"vocab={tok.vocab_size}")
    except Exception as exc:
        add("load_run", False, f"{type(exc).__name__}: {exc}")
        return {"ok": all(c["ok"] for c in checks), "run": str(run), "checks": checks}

    try:
        import numpy as np
        import tensorflow as tf

        interp = tf.lite.Interpreter(
            model_path=str(model_dir / "model_int8.tflite"),
            num_threads=cfg.inference.tflite_threads,
        )
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        sig = inp.get("shape_signature", inp["shape"]).tolist()
        add(
            "tflite_dynamic",
            sig[1] == -1 and sig[2] == cfg.model.chunk_size,
            str(sig),
        )
        interp.resize_tensor_input(inp["index"], [1, 3, cfg.model.chunk_size])
        interp.allocate_tensors()
        interp.set_tensor(
            inp["index"],
            np.ones((1, 3, cfg.model.chunk_size), dtype=np.int32),
        )
        interp.invoke()
        add("tflite_invoke", True, "3 chunks")
    except Exception as exc:
        add("tflite", False, f"{type(exc).__name__}: {exc}")

    sample = "$ python -m pip install numpy\nSuccessfully installed numpy-2.2.2\n"
    try:
        from .engine import ConsoleEngine

        engine = ConsoleEngine(model, tok, cfg, tflite_path=str(model_dir / "model_int8.tflite"))
        result = engine.analyze_lines(sample.splitlines())
        add(
            "smoke_inference",
            result.get("status") in cfg.model.heads.class_names,
            result.get("status", "missing"),
        )
    except Exception as exc:
        add("smoke", False, f"{type(exc).__name__}: {exc}")

    metrics_path = run / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            add(
                "int8_gate",
                bool(metrics.get("int8_validation", {}).get("gates", {}).get("ok")),
                str(metrics.get("int8_validation", {}).get("gates")),
            )
        except Exception:
            add("int8_gate", False, "metrics unreadable")

    if data_dir:
        try:
            from .evaluate import evaluate_run

            real = evaluate_run(run_dir, data_dir, "test")
            add(
                "data_eval",
                real["doc"]["accuracy"] >= 0.85,
                f"doc={real['doc']['accuracy']:.4f}",
            )
        except Exception as exc:
            add("data_eval", False, f"{type(exc).__name__}: {exc}")

    return {"ok": all(c["ok"] for c in checks), "run": str(run), "checks": checks}
