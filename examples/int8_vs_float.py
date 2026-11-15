#!/usr/bin/env python3
"""Compare float Keras and int8 TFLite on one log."""
from pathlib import Path
import sys
import time

from console_nn import ConsoleAnalyzer

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("examples/logs/02-pytest-failure.log")
float_model = ConsoleAnalyzer()
int8_model = ConsoleAnalyzer(tflite=True)

t0 = time.perf_counter()
a = float_model.analyze_file(path)
t_float = time.perf_counter() - t0

t0 = time.perf_counter()
b = int8_model.analyze_file(path)
t_int8 = time.perf_counter() - t0

print(f"float: {a.status:7s} conf={a.confidence:.4f} time={t_float*1000:.1f} ms")
print(f"int8 : {b.status:7s} conf={b.confidence:.4f} time={t_int8*1000:.1f} ms")
print(f"doc agreement: {a.status == b.status}")
