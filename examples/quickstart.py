#!/usr/bin/env python3
"""Minimal API use: text in, LogAnalysis out."""
from pathlib import Path
import sys

from console_nn import analyze

SAMPLE = """$ python -m pip install requests
Collecting requests
Successfully installed requests-2.32.3
"""

text = Path(sys.argv[1]).read_text() if len(sys.argv) > 1 else SAMPLE
result = analyze(text, tflite=True)

print("status:", result.status)
print("confidence:", f"{result.confidence:.3f}")
print("completed:", result.completed)
print("chunks:", result.n_chunks)
print()
print(result.to_brief())
