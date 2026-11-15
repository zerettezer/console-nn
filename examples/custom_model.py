#!/usr/bin/env python3
"""Load a specific packaged model and list available checkpoints."""
import json

from console_nn import ConsoleAnalyzer, available_models

print(json.dumps(available_models(), indent=2))
print()

analyzer = ConsoleAnalyzer(model_dir="console_nn/pretrained/rms", tflite=True)
result = analyzer.analyze("""
$ systemctl start app.service
Job for app.service failed because the control process exited with error code.
""")
print(result.status, result.confidence)
