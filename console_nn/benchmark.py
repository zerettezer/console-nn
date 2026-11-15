"""CPU latency benchmarks for exported TFLite models."""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

from .evaluate import resolve_model_dir


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return f"{platform.machine()} CPU"

def _measure(path: Path, iters: int, threads: int, window_chunks: int = 128) -> Dict:
    interp = tf.lite.Interpreter(model_path=str(path), num_threads=threads)
    interp.allocate_tensors()
    detail = interp.get_input_details()[0]
    signature = detail.get("shape_signature", detail["shape"]).tolist()
    chunk = int(detail["shape"][2]) if int(detail["shape"][2]) > 0 else 16
    if signature[1] == -1:
        interp.resize_tensor_input(detail["index"], [1, window_chunks, chunk])
        interp.allocate_tensors()
        detail = interp.get_input_details()[0]
    shape = [int(x) for x in detail["shape"]]
    tokens = int(np.prod(shape))
    rng = np.random.default_rng(0)
    data = rng.integers(1, 2999, size=shape, dtype=np.int32)
    for _ in range(10):
        interp.set_tensor(detail["index"], data)
        interp.invoke()
    start = time.perf_counter()
    for _ in range(iters):
        interp.set_tensor(detail["index"], data)
        interp.invoke()
    elapsed = (time.perf_counter() - start) / max(1, iters)
    return {
        "mode": "float16" if "float16" in path.name else "int8",
        "file": path.name,
        "size_kb": round(path.stat().st_size / 1024, 1),
        "input_shape": shape,
        "tokens_per_window": tokens,
        "ms_per_window": round(elapsed * 1000, 3),
        "tokens_per_second": int(round(tokens / elapsed)),
    }


def benchmark_run(
    run_dir: str,
    tflite: Optional[str] = None,
    iters: int = 300,
    threads: int = 4,
    window_chunks: int = 128,
    save: bool = True,
) -> Dict:
    model_dir = resolve_model_dir(run_dir)
    run = model_dir.parent if model_dir.name == "model" else model_dir
    if tflite:
        paths = [Path(tflite)]
    else:
        paths = [model_dir / "model_int8.tflite"]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise FileNotFoundError(f"no TFLite files in {model_dir}")
    results: List[Dict] = [
        _measure(p, iters=iters, threads=threads, window_chunks=window_chunks)
        for p in paths
    ]
    report = {
        "cpu": cpu_name(),
        "threads": threads,
        "iterations": iters,
        "window_chunks": window_chunks,
        "tensorflow": tf.__version__,
        "results": results,
    }
    if save:
        (run / "benchmark.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    return report
