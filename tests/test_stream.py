"""Streaming analyzer equivalence and buffering tests."""
from __future__ import annotations

import unittest

from console_nn import ConsoleAnalyzer
from console_nn.stream import StreamAnalyzer


LOG = """$ python train.py
loading config
Traceback (most recent call last):
  File "train.py", line 4, in <module>
ModuleNotFoundError: No module named 'yaml'
"""


class StreamTest(unittest.TestCase):
    def test_incremental_matches_full(self):
        analyzer = ConsoleAnalyzer()
        whole = analyzer.analyze(LOG)
        stream = StreamAnalyzer(analyzer)
        pieces = [LOG[i : i + 7] for i in range(0, len(LOG), 7)]
        result = None
        for piece in pieces:
            result = stream.feed(piece)
        result = stream.flush()
        self.assertEqual(result.status, whole.status)
        self.assertAlmostEqual(result.confidence, whole.confidence, places=5)
        self.assertEqual(result.completed, whole.completed)
        self.assertEqual(result.n_tokens, whole.n_tokens)

    def test_partial_line_is_buffered(self):
        analyzer = ConsoleAnalyzer()
        stream = StreamAnalyzer(analyzer)
        stream.feed("$ python train.py\n")
        result = stream.feed("ModuleNot")
        self.assertNotIn("ModuleNot", [line.text for line in result.lines])
        self.assertEqual(stream.buffered_text.split("\n")[-1], "ModuleNot")
        result = stream.feed("FoundError: No module named 'yaml'\n")
        self.assertIn("ModuleNotFoundError", result.lines[-1].text)

    def test_carriage_return_rewrites_line(self):
        analyzer = ConsoleAnalyzer()
        stream = StreamAnalyzer(analyzer)
        stream.feed("progress  0%\r")
        stream.feed("progress 50%\r")
        result = stream.feed("progress 100%\n")
        self.assertEqual(result.lines[-1].text, "progress 100%")
        self.assertNotIn("0%", [line.text for line in result.lines])

    def test_crlf_split_across_feeds(self):
        analyzer = ConsoleAnalyzer()
        stream = StreamAnalyzer(analyzer)
        stream.feed("line one\r")
        result = stream.feed("\n")
        self.assertEqual([line.text for line in result.lines], ["line one"])


if __name__ == "__main__":
    unittest.main()
