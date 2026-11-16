"""Unit tests for tokenizer, dataset encoding, model and rendering."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from console_nn.config import Config, TokenizerConfig, load_config
from console_nn.data import encode_with_stride
from console_nn.model import build_model, model_breakdown
from console_nn.synth import generate_document, SynthConfig
from console_nn.tokenizer import NgramTokenizer
from console_nn.viz import render_ascii, render_brief, render_png

ROOT = Path(__file__).resolve().parents[1]


class TokenizerTest(unittest.TestCase):
    def setUp(self):
        self.lines = [
            "$ python -m pip install numpy==2.2.2",
            "Collecting numpy==2.2.2",
            "ERROR: Could not find a version that satisfies the requirement",
            "Permission denied: /usr/lib/python3.10/site-packages",
        ]

    def test_ngrams_and_budget(self):
        cfg = TokenizerConfig(max_vocab=300, min_freq=1, ngram_max=3)
        tok = NgramTokenizer(cfg).fit([self.lines])
        self.assertLessEqual(tok.vocab_size, 300)
        tokens = tok.ngrams("pip install numpy")
        self.assertIn("pip", tokens)
        self.assertIn("pip install", tokens)
        self.assertIn("pip install numpy", tokens)

    def test_numbers_are_digit_tokens(self):
        tiny = NgramTokenizer(TokenizerConfig(max_vocab=500, min_freq=1)).fit(
            [["port 8080", "port 8081"]]
        )
        a = tiny.encode_line("port 8080")[1]
        b = tiny.encode_line("port 8081")[1]
        self.assertNotEqual(a, b)
        self.assertIn("<NUM>", a)
        self.assertIn("<8>", a)

    def test_hash_buckets(self):
        cfg = TokenizerConfig(max_vocab=256, min_freq=1, hash_buckets=16)
        tok = NgramTokenizer(cfg).fit([self.lines])
        self.assertLessEqual(tok.vocab_size, 256)
        ids, _ = tok.encode_line("completely_unknown_token_xyz")
        self.assertTrue(any(i >= tok.hash_base for i in ids))

    def test_save_load_roundtrip(self):
        tok = NgramTokenizer(TokenizerConfig(max_vocab=512, min_freq=1)).fit(
            [self.lines]
        )
        ids, _ = tok.encode_line("pip install numpy")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tokenizer.json"
            tok.save(str(path))
            loaded = NgramTokenizer.load(str(path))
        self.assertEqual(loaded.vocab_size, tok.vocab_size)
        self.assertEqual(loaded.encode_line("pip install numpy")[0], ids)


class DatasetTest(unittest.TestCase):
    def test_encode_shapes_and_labels(self):
        cfg = Config()
        cfg.data.chunk_size = 16
        cfg.model.chunk_size = 16
        rng = __import__("random").Random(4)
        doc = generate_document(rng, SynthConfig(), 0)
        tok = NgramTokenizer(TokenizerConfig(max_vocab=500, min_freq=1)).fit(
            [[line["text"] for line in doc["lines"]]]
        )
        ids, labels, doc_label, completed = encode_with_stride(
            doc, tok, cfg.data, cfg.data.eval_stride
        )
        self.assertEqual(ids.shape[1], 16)
        self.assertEqual(labels.shape[0], ids.shape[0])
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels <= 4))
        self.assertIn(doc_label, (0, 1, 2, 3))
        self.assertIn(completed, (0.0, 1.0))
        # padding is id 0 and never carries a positive chunk label
        fully_padded = np.all(ids == 0, axis=1)
        self.assertTrue(np.all(labels[fully_padded] == 0))


class ModelTest(unittest.TestCase):
    def test_forward_and_size(self):
        cfg = load_config(str(ROOT / "configs" / "cpu.json"))
        model = build_model(cfg, 3000)
        ids = np.random.randint(1, 3000, size=(2, 4, 16)).astype(np.int32)
        out = model(ids, training=False)
        self.assertEqual(tuple(out["chunk_logits"].shape), (2, 4, 4))
        self.assertEqual(tuple(out["doc_logits"].shape), (2, 4))
        self.assertEqual(tuple(out["completed_logits"].shape), (2, 1))
        for value in out.values():
            self.assertTrue(np.all(np.isfinite(value.numpy())))
        breakdown = model_breakdown(model)
        self.assertLess(breakdown["total"], 200_000)
        # fully padded chunks must not leak -1e9 sentinels
        padded = np.zeros((1, 4, 16), dtype=np.int32)
        out_pad = model(padded, training=False)
        self.assertLess(float(np.max(np.abs(out_pad["doc_logits"]))), 100.0)

    def test_split_heads_match_full_call(self):
        cfg = load_config(str(ROOT / "configs" / "cpu.json"))
        model = build_model(cfg, 3000)
        ids = np.random.randint(1, 3000, size=(2, 5, 16)).astype(np.int32)
        full = model(ids, training=False)
        root = model.get_layer("console_fcn")
        feat, chunk_logits, mask = root.encode_chunks(ids, training=False)
        split = root.document_logits(feat, mask, chunk_logits, training=False)
        np.testing.assert_allclose(
            split["chunk_logits"].numpy(), full["chunk_logits"].numpy(),
            rtol=1e-5, atol=1e-5,
        )
        np.testing.assert_allclose(
            split["doc_logits"].numpy(), full["doc_logits"].numpy(),
            rtol=1e-5, atol=1e-5,
        )
        np.testing.assert_allclose(
            split["completed_logits"].numpy(), full["completed_logits"].numpy(),
            rtol=1e-5, atol=1e-5,
        )


class VizTest(unittest.TestCase):
    def _result(self):
        return {
            "status": "fail",
            "confidence": 0.91,
            "probs": {"success": 0.01, "fail": 0.91, "block": 0.07, "unknown": 0.01},
            "completed": True,
            "completed_confidence": 0.93,
            "n_tokens": 32,
            "n_chunks": 2,
            "backend": "keras",
            "chunks": [
                {"index": 0, "label": "success", "confidence": 0.8,
                 "token_start": 0, "token_end": 16, "lines": [0],
                 "probs": {"success": 0.8, "fail": 0.1, "block": 0.05, "unknown": 0.05}},
                {"index": 1, "label": "fail", "confidence": 0.95,
                 "token_start": 16, "token_end": 32, "lines": [1],
                 "probs": {"success": 0.01, "fail": 0.95, "block": 0.03, "unknown": 0.01}},
            ],
            "lines": [
                {"index": 0, "text": "running", "label": "success", "confidence": 0.8,
                 "severity": 0.15, "probs": {}},
                {"index": 1, "text": "ERROR: failed", "label": "fail", "confidence": 0.95,
                 "severity": 0.98, "probs": {}},
            ],
        }

    def test_render(self):
        result = self._result()
        text = render_ascii(result, color=False, show_chunks=False)
        self.assertIn("RESULT: FAIL", text)
        self.assertIn("LOG LINES", text)
        brief = render_brief(result, color=False)
        self.assertIn("PROBLEM LINES", brief)
        self.assertIn("ERROR: failed", brief)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "report.png"
            render_png(result, str(path))
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
