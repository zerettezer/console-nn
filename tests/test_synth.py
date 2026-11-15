"""Synthetic generator and split-profile tests."""
from __future__ import annotations

import os
import random
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from console_nn.synth import SynthConfig, corpus_stats, generate_document
from console_nn.synth_commands import _COMPLEX
from console_nn.synth_errors import ERROR_SPECS
from console_nn.synth_real_errors import SPECS as REAL_SPECS
from console_nn.text import normalize_text, tokenize


class SynthCoverageTest(unittest.TestCase):
    def test_real_error_and_complex_counts(self):
        self.assertGreaterEqual(len(REAL_SPECS), 20)
        self.assertGreaterEqual(len(ERROR_SPECS), 20)
        self.assertGreaterEqual(len(_COMPLEX), 20)

    def test_splits_are_different_and_have_no_exit_literal(self):
        cfg = SynthConfig()
        averages = {}
        domains = set()
        for split, offset in (("train", 0), ("val", 100003), ("test", 200003)):
            rng = random.Random(7 + offset)
            docs = [generate_document(rng, cfg, i, split=split) for i in range(180)]
            stats = corpus_stats(docs)
            averages[split] = stats["avg_lines"]
            domains.update(stats["domain"])
            text = "\n".join(
                line["text"] for doc in docs for line in doc["lines"]
            )
            self.assertNotIn("[exit code", text)
            for doc in docs:
                self.assertEqual(doc["split"], split)
                for line in doc["lines"]:
                    self.assertIn(
                        line["label"], ("success", "fail", "block", "neutral")
                    )
        self.assertNotEqual(round(averages["train"], 1), round(averages["test"], 1))
        self.assertIn("multi", domains)
        self.assertIn("real_errors", domains)

    def test_number_tokens(self):
        self.assertNotEqual(
            tokenize(normalize_text("port 8080")),
            tokenize(normalize_text("port 8081")),
        )
        self.assertIn("<NUM>", tokenize(normalize_text("port 8080")))


if __name__ == "__main__":
    unittest.main()
