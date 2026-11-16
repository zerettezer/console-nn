"""Incremental console-log analysis.

The analyzer keeps tokenization and model state for every completed chunk, so
a growing log is not re-encoded or re-run from the beginning.  An unfinished
line stays in a small text buffer until a newline arrives.  Carriage returns
are interpreted as in-place rewrites and survive chunk boundaries.

The exact incremental path uses the float Keras backend, which exposes the
per-chunk features that feed the document head.  The exported int8 TFLite
graph only exposes final document outputs, so streaming with that backend
falls back to a full re-analysis of committed lines.

Carriage returns are treated as terminal rewrites (the latest text on a line
wins).  A log full of progress-bar rewrites therefore yields fewer lines than
`ConsoleAnalyzer.analyze`, which uses Python's `splitlines`.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from .api import LogAnalysis


class StreamAnalyzer:
    """Feed console output in pieces and receive an updated verdict.

    Parameters
    ----------
    analyzer:
        A :class:`console_nn.api.ConsoleAnalyzer` instance.  The streaming
        state is independent, so one analyzer can serve several streams.

    Attributes
    ----------
    last_latency_ms:
        Wall-clock time spent in the most recent :meth:`feed` or
        :meth:`flush`, including tokenization and result assembly.
    last_chunk_latency_ms:
        Model time for each newly completed chunk in the most recent call.
    mean_chunk_latency_ms:
        Model time per completed chunk over the whole stream.
    """

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.engine = analyzer.engine
        self.tokenizer = analyzer.engine.tokenizer
        self.chunk_size = int(analyzer.engine.cfg.model.chunk_size)
        self.class_names = list(analyzer.engine.cfg.model.heads.class_names)

        self._lines: List[str] = []
        self._partial = ""
        self._tokens: List[int] = []
        self._line_ids: List[int] = []
        self._consumed_tokens = 0

        self._feat_sum: Optional[np.ndarray] = None
        self._feat_max: Optional[np.ndarray] = None
        self._prob_sum: Optional[np.ndarray] = None
        self._prob_max: Optional[np.ndarray] = None
        self._closed_count = 0
        self._chunk_records: List[Dict] = []

        self.last_latency_ms = 0.0
        self.last_chunk_latency_ms = 0.0
        self.mean_chunk_latency_ms = 0.0
        self._model_ms = 0.0
        self._model_chunks = 0

    # ------------------------------------------------------------------
    @property
    def buffered_text(self) -> str:
        """Committed lines followed by the unfinished line."""
        if self._lines:
            return "\n".join(self._lines + [self._partial])
        return self._partial

    @property
    def n_lines(self) -> int:
        return len(self._lines)

    @property
    def n_tokens(self) -> int:
        return len(self._tokens)

    # ------------------------------------------------------------------
    def feed(self, text: str) -> LogAnalysis:
        """Append a chunk of output and return the current analysis."""
        started = time.perf_counter()
        self._append_text(str(text))
        if self.engine.interpreter is not None:
            result = self._full_reanalysis()
        else:
            self._process_new_chunks()
            result = self._build_result()
        result.meta["streaming"] = True
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        return result

    def flush(self) -> LogAnalysis:
        """Commit the unfinished line and return the final analysis."""
        started = time.perf_counter()
        if self._partial:
            self._commit_line(self._collapse(self._partial))
            self._partial = ""
        if self.engine.interpreter is not None:
            result = self._full_reanalysis()
        else:
            self._process_new_chunks()
            result = self._build_result()
        result.meta["streaming"] = True
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        return result

    # ------------------------------------------------------------------
    # text and token state
    # ------------------------------------------------------------------
    @staticmethod
    def _collapse(text: str) -> str:
        """Apply terminal carriage-return rewrites to one physical line."""
        if "\r" not in text:
            return text
        parts = text.split("\r")
        if text.endswith("\r") and len(parts) > 1:
            return parts[-2]
        return parts[-1]

    def _append_text(self, text: str) -> None:
        data = (self._partial + text).replace("\r\n", "\n")
        *complete, remainder = data.split("\n")
        for line in complete:
            self._commit_line(self._collapse(line))
        if "\r" in remainder:
            parts = remainder.split("\r")
            if remainder.endswith("\r") and len(parts) > 1:
                # Keep the trailing CR so a split CRLF is recognised next time.
                remainder = parts[-2] + "\r"
            else:
                remainder = parts[-1]
        self._partial = remainder

    def _commit_line(self, line: str) -> None:
        line_no = len(self._lines)
        self._lines.append(line)
        ids, _ = self.tokenizer.encode_line(line)
        self._tokens.extend(ids)
        self._line_ids.extend([line_no] * len(ids))

    # ------------------------------------------------------------------
    # model state
    # ------------------------------------------------------------------
    def _process_new_chunks(self) -> None:
        pending: List[np.ndarray] = []
        while len(self._tokens) - self._consumed_tokens >= self.chunk_size:
            start = self._consumed_tokens
            end = start + self.chunk_size
            pending.append(np.asarray(self._tokens[start:end], dtype=np.int32))
            self._chunk_records.append(
                {
                    "index": len(self._chunk_records),
                    "token_start": start,
                    "token_end": end,
                    "lines": sorted({int(x) for x in self._line_ids[start:end]}),
                }
            )
            self._consumed_tokens = end
        if not pending:
            self.last_chunk_latency_ms = 0.0
            return

        batch = np.stack(pending, axis=0)
        model_started = time.perf_counter()
        feat, logits = self.engine.encode_chunk_features(batch)
        elapsed_ms = (time.perf_counter() - model_started) * 1000.0
        probs = self._softmax(logits)
        self._add_to_state(feat, probs)
        self.last_chunk_latency_ms = elapsed_ms / len(pending)
        self._model_ms += elapsed_ms
        self._model_chunks += len(pending)
        self.mean_chunk_latency_ms = self._model_ms / max(1, self._model_chunks)

        for offset, record in enumerate(self._chunk_records[-len(pending):]):
            self._fill_chunk_record(record, probs[offset])

    def _add_to_state(self, feat: np.ndarray, probs: np.ndarray) -> None:
        if self._feat_sum is None:
            self._feat_sum = np.zeros_like(feat[0])
            self._feat_max = np.full_like(feat[0], -np.inf)
            self._prob_sum = np.zeros_like(probs[0])
            self._prob_max = np.full_like(probs[0], -np.inf)
        self._feat_sum += feat.sum(axis=0)
        self._feat_max = np.maximum(self._feat_max, feat.max(axis=0))
        self._prob_sum += probs.sum(axis=0)
        self._prob_max = np.maximum(self._prob_max, probs.max(axis=0))
        self._closed_count += int(feat.shape[0])

    def _tail(self):
        if len(self._tokens) <= self._consumed_tokens:
            return None
        ids = np.zeros((1, self.chunk_size), dtype=np.int32)
        tail = self._tokens[self._consumed_tokens:]
        ids[0, : len(tail)] = tail
        feat, logits = self.engine.encode_chunk_features(ids)
        return {
            "index": len(self._chunk_records),
            "token_start": self._consumed_tokens,
            "token_end": len(self._tokens),
            "lines": sorted(
                {int(x) for x in self._line_ids[self._consumed_tokens:]}
            ),
            "feat": feat[0],
            "probs": self._softmax(logits)[0],
        }

    def measure_chunk_latency(self, iterations: int = 20) -> float:
        """Median cost of one new chunk plus the document-head update."""
        if not self._tokens:
            return 0.0
        ids = np.zeros((1, self.chunk_size), dtype=np.int32)
        tail = self._tokens[-self.chunk_size :]
        ids[0, : len(tail)] = tail
        samples = []
        for _ in range(max(1, iterations)):
            started = time.perf_counter()
            feat, logits = self.engine.encode_chunk_features(ids)
            probs = self._softmax(logits)
            self.engine.document_from_pools(
                feat[0], feat[0], probs[0], probs[0], 1
            )
            samples.append((time.perf_counter() - started) * 1000.0)
        samples.sort()
        return samples[len(samples) // 2]

    # ------------------------------------------------------------------
    # result assembly
    # ------------------------------------------------------------------
    def _build_result(self) -> LogAnalysis:
        tail = self._tail()
        if self._closed_count == 0 and tail is None:
            return self.analyzer.analyze_lines(self._lines)

        feat_sum = self._feat_sum
        feat_max = self._feat_max
        prob_sum = self._prob_sum
        prob_max = self._prob_max
        count = self._closed_count
        if tail is not None:
            feat_sum = tail["feat"] if feat_sum is None else feat_sum + tail["feat"]
            feat_max = (
                tail["feat"]
                if feat_max is None
                else np.maximum(feat_max, tail["feat"])
            )
            prob_sum = (
                tail["probs"] if prob_sum is None else prob_sum + tail["probs"]
            )
            prob_max = (
                tail["probs"]
                if prob_max is None
                else np.maximum(prob_max, tail["probs"])
            )
            count += 1

        doc_logits, completed_logit = self.engine.document_from_pools(
            np.asarray(feat_sum, dtype=np.float32),
            np.asarray(feat_max, dtype=np.float32),
            np.asarray(prob_sum, dtype=np.float32),
            np.asarray(prob_max, dtype=np.float32),
            count,
        )
        doc_prob = self._softmax(doc_logits[None, :])[0]
        completed_prob = float(1.0 / (1.0 + np.exp(-completed_logit)))

        records = [dict(r) for r in self._chunk_records]
        if tail is not None:
            records.append(tail)
        for record in records:
            if "probs" in record and not isinstance(record["probs"], dict):
                self._fill_chunk_record(record, record["probs"])

        line_records = self._aggregate_lines(records)
        status_idx = int(np.argmax(doc_prob))
        completed = completed_prob >= 0.5
        n_chunks = self._closed_count + (1 if tail is not None else 0)
        raw = {
            "status": self.class_names[status_idx],
            "confidence": float(doc_prob[status_idx]),
            "probs": {
                name: float(doc_prob[i])
                for i, name in enumerate(self.class_names)
            },
            "completed": bool(completed),
            "completed_confidence": float(
                completed_prob if completed else 1.0 - completed_prob
            ),
            "n_tokens": len(self._tokens),
            "n_chunks": n_chunks,
            "chunk_size": self.chunk_size,
            "chunks": [
                {
                    key: value
                    for key, value in record.items()
                    if key in (
                        "index", "token_start", "token_end", "lines", "probs",
                        "label", "confidence",
                    )
                }
                for record in records
            ],
            "lines": line_records,
            "backend": "keras",
        }
        return LogAnalysis.from_raw(raw, model_dir=self.analyzer.model_dir)

    def _aggregate_lines(self, records: List[Dict]) -> List[Dict]:
        n = len(self._lines)
        sums = np.zeros((n, len(self.class_names)), dtype=np.float64)
        hits = np.zeros((n,), dtype=np.float64)
        for record in records:
            probs = np.asarray(
                [record["probs"][name] for name in self.class_names],
                dtype=np.float64,
            )
            for line_no in record["lines"]:
                if 0 <= line_no < n:
                    sums[line_no] += probs
                    hits[line_no] += 1.0

        fail_idx = (
            self.class_names.index("fail") if "fail" in self.class_names else None
        )
        block_idx = (
            self.class_names.index("block") if "block" in self.class_names else None
        )
        out = []
        for i, text in enumerate(self._lines):
            if hits[i] > 0:
                probs = sums[i] / hits[i]
                label_idx = int(np.argmax(probs))
                severity = float(
                    (probs[fail_idx] if fail_idx is not None else 0.0)
                    + (probs[block_idx] if block_idx is not None else 0.0)
                )
                out.append(
                    {
                        "index": i,
                        "text": text,
                        "label": self.class_names[label_idx],
                        "confidence": float(probs[label_idx]),
                        "severity": severity,
                        "probs": {
                            name: float(probs[j])
                            for j, name in enumerate(self.class_names)
                        },
                    }
                )
            else:
                probs = np.full(
                    (len(self.class_names),), 1.0 / len(self.class_names)
                )
                out.append(
                    {
                        "index": i,
                        "text": text,
                        "label": "unknown",
                        "confidence": float(
                            probs[self.class_names.index("unknown")]
                        )
                        if "unknown" in self.class_names
                        else 0.0,
                        "severity": 0.0,
                        "probs": {
                            name: float(probs[j])
                            for j, name in enumerate(self.class_names)
                        },
                    }
                )
        return out

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _fill_chunk_record(self, record: Dict, probs: np.ndarray) -> None:
        probs = np.asarray(probs, dtype=np.float64)
        label_idx = int(np.argmax(probs))
        record["label"] = self.class_names[label_idx]
        record["confidence"] = float(probs[label_idx])
        record["probs"] = {
            name: float(probs[i]) for i, name in enumerate(self.class_names)
        }
        record.pop("feat", None)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        x = np.asarray(logits, dtype=np.float64)
        x = x - np.max(x, axis=-1, keepdims=True)
        exp = np.exp(x)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def _full_reanalysis(self) -> LogAnalysis:
        started = time.perf_counter()
        result = self.analyzer.analyze_lines(self._lines)
        self.last_chunk_latency_ms = (time.perf_counter() - started) * 1000.0
        return result
