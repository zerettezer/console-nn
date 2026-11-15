"""Real-data collection and mixed-corpus contract."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RealDataTest(unittest.TestCase):
    def test_real_splits_exist(self):
        total = 0
        for split in ("train", "val", "test"):
            path = ROOT / "data" / "real" / f"{split}.jsonl"
            self.assertTrue(path.is_file(), path)
            docs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
            self.assertTrue(docs, split)
            total += len(docs)
            for doc in docs:
                self.assertIn(doc["source"], ("real", "real_user"))
                self.assertTrue(doc["lines"])
        self.assertGreaterEqual(total, 50)

    def test_mixed_contains_both_sources(self):
        docs = [
            json.loads(x)
            for x in (ROOT / "data" / "mixed" / "train.jsonl").read_text().splitlines()
            if x.strip()
        ]
        sources = {d.get("source") for d in docs}
        self.assertIn("synthetic", sources)
        self.assertIn("real", sources)

    def test_no_exit_code_literals(self):
        for split in ("train", "val", "test"):
            text = (ROOT / "data" / "real" / f"{split}.jsonl").read_text()
            self.assertNotIn("[exit code", text)
            self.assertNotIn('"exit_code"', text)


if __name__ == "__main__":
    unittest.main()
