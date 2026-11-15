#!/usr/bin/env python3
"""Analyze every *.log in a directory and aggregate the verdicts."""
from collections import Counter
from pathlib import Path
import argparse
import json

from console_nn import ConsoleAnalyzer

parser = argparse.ArgumentParser()
parser.add_argument("directory", nargs="?", default="examples/logs")
parser.add_argument("--tflite", action="store_true")
args = parser.parse_args()

analyzer = ConsoleAnalyzer(tflite=args.tflite or None)
counts = Counter()
rows = []
for path in sorted(Path(args.directory).glob("*.log")):
    result = analyzer.analyze_file(path)
    counts[result.status] += 1
    rows.append({
        "file": path.name,
        "status": result.status,
        "confidence": round(result.confidence, 4),
        "completed": result.completed,
        "chunks": result.n_chunks,
    })

print(json.dumps({"counts": dict(counts), "results": rows}, indent=2))
