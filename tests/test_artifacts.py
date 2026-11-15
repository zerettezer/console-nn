"""Artifact layout, TFLite and public API tests."""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from console_nn import ConsoleAnalyzer, analyze, available_models

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
RUNS = ("base", "cpu", "rms")

SUCCESS_LOG = """$ python -m pip install numpy==2.2.2
Collecting numpy==2.2.2
Installing collected packages: numpy
Successfully installed numpy-2.2.2
"""

FAIL_LOG = """$ python train.py
Traceback (most recent call last):
  File "train.py", line 42, in <module>
    load_config()
ModuleNotFoundError: No module named 'yaml'
"""

BLOCK_LOG = """$ cp main.py /opt/app/
cp: cannot create regular file '/opt/app/main.py': Permission denied
Access denied by security policy.
"""


class LayoutTest(unittest.TestCase):
    def test_one_model_folder_per_run(self):
        self.assertEqual({p.name for p in ARTIFACTS.iterdir()}, set(RUNS))
        for run in RUNS:
            model = ARTIFACTS / run / "model"
            self.assertTrue((model / "config.json").is_file())
            self.assertTrue((model / "tokenizer.json").is_file())
            self.assertTrue((model / "weights.h5").is_file())
            self.assertTrue((model / "model_int8.tflite").is_file())
            # exactly one log, one benchmark and one metrics file per run
            self.assertTrue((ARTIFACTS / run / "run.log").is_file())
            self.assertTrue((ARTIFACTS / run / "benchmark.json").is_file())
            self.assertTrue((ARTIFACTS / run / "metrics.json").is_file())
            self.assertFalse((ARTIFACTS / run / "report.png").exists())
            self.assertFalse((ARTIFACTS / run / "logs").exists())
            self.assertFalse((ARTIFACTS / run / "reports").exists())

    def test_metrics_and_benchmark(self):
        for run in RUNS:
            metrics = json.loads((ARTIFACTS / run / "metrics.json").read_text())
            self.assertEqual(metrics["run"], f"artifacts/{run}")
            self.assertGreater(metrics["float"]["doc"]["accuracy"], 0.90)
            self.assertGreater(metrics["int8"]["doc"]["accuracy"], 0.75)
            self.assertTrue(metrics["int8_validation"]["gates"]["ok"])
            self.assertIn("breakdown", metrics["model"])
            benchmark = json.loads((ARTIFACTS / run / "benchmark.json").read_text())
            self.assertEqual(len(benchmark["results"]), 1)
            self.assertEqual(benchmark["results"][0]["mode"], "int8")
            self.assertGreater(benchmark["results"][0]["tokens_per_second"], 100000)


class TFLiteTest(unittest.TestCase):
    def test_int8_tensor_contract(self):
        import tensorflow as tf

        path = ARTIFACTS / "cpu" / "model" / "model_int8.tflite"
        interp = tf.lite.Interpreter(model_path=str(path))
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        self.assertEqual(inp["dtype"], np.int32)
        self.assertEqual(list(inp["shape_signature"]), [1, -1, 16])
        self.assertEqual([o["dtype"] for o in interp.get_output_details()],
                         [np.float32, np.float32, np.float32])
        for chunks in (1, 7, 100):
            interp.resize_tensor_input(inp["index"], [1, chunks, 16])
            interp.allocate_tensors()
            interp.set_tensor(inp["index"],
                              np.ones((1, chunks, 16), dtype=np.int32))
            interp.invoke()
            outputs = interp.get_output_details()
            shapes = sorted([list(o["shape"]) for o in outputs])
            self.assertIn([1, chunks, 4], shapes)


class ValidationTest(unittest.TestCase):
    def test_validation_contract(self):
        from console_nn.int8_validation import validate_run

        report = validate_run(
            str(ARTIFACTS / "cpu"), str(ROOT / "data" / "synth"),
            split="test", max_docs=16, save=False,
        )
        self.assertEqual(report["documents"], 16)
        self.assertIn("doc", report["float"])
        self.assertIn("per_class", report["int8"]["doc"])
        self.assertIn("agreement", report)
        self.assertIn("gates", report)


class ApiTest(unittest.TestCase):
    def test_pretrained_tflite(self):
        analyzer = ConsoleAnalyzer(tflite=True)
        result = analyzer.analyze(SUCCESS_LOG)
        self.assertEqual(result.backend, "tflite")
        self.assertEqual(result.status, "success")
        self.assertTrue(result.completed)
        self.assertGreater(result.confidence, 0.9)
        self.assertEqual(len(result.chunks), result.n_chunks)

    def test_classification_and_formats(self):
        analyzer = ConsoleAnalyzer()
        fail = analyzer.analyze(FAIL_LOG)
        block = analyzer.analyze(BLOCK_LOG)
        self.assertEqual(fail.status, "fail")
        self.assertEqual(block.status, "block")
        self.assertFalse(block.completed)
        self.assertIn("RESULT: FAIL", fail.to_ascii(show_chunks=False))
        self.assertIn("PROBLEM LINES", fail.to_brief())
        json.loads(fail.to_json())

    def test_one_call_api_and_models(self):
        result = analyze(SUCCESS_LOG, tflite=True)
        self.assertEqual(result.status, "success")
        models = available_models()
        self.assertTrue(any("artifacts/cpu" in m["path"] for m in models))


if __name__ == "__main__":
    unittest.main()
