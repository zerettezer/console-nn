"""Manifest and deployment-doctor checks."""
from __future__ import annotations

import unittest

from console_nn.doctor import doctor, write_manifest


class DoctorTest(unittest.TestCase):
    def test_manifest_and_doctor(self):
        manifest = write_manifest("artifacts/cpu")
        self.assertIn("config.json", manifest["files"])
        self.assertIn("model_int8.tflite", manifest["files"])
        report = doctor("artifacts/cpu")
        failed = [c for c in report["checks"] if not c["ok"]]
        self.assertEqual(failed, [])


if __name__ == "__main__":
    unittest.main()
