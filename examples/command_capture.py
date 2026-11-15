#!/usr/bin/env python3
"""Run a command, capture stdout+stderr, classify the transcript.

Only run commands you are allowed to execute. The command is executed locally.
"""
import argparse

from console_nn import ConsoleAnalyzer

parser = argparse.ArgumentParser()
parser.add_argument("command", nargs="?", default='python3 -c "import sys; sys.exit(1)"')
args = parser.parse_args()

analyzer = ConsoleAnalyzer(tflite=True)
result = analyzer.analyze_command(args.command)
print(result.to_brief())
print("return code:", result.meta.get("returncode"))
