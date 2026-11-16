"""Inference engine: Keras or int8 TFLite, arbitrary log length."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from .config import Config, nested_update
from .tokenizer import NgramTokenizer


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


class ConsoleEngine:
    """Low-level engine used by :class:`console_nn.api.ConsoleAnalyzer`."""

    def __init__(
        self,
        model,
        tokenizer: NgramTokenizer,
        cfg: Config,
        tflite_path: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.tflite_path = tflite_path
        self.interpreter = None
        self.tflite_input = None
        self.tflite_shape = None
        self.tflite_outputs = {}
        self.tflite_probabilities = False
        self.window_chunks = int(cfg.inference.window_chunks)
        self.window_stride = int(cfg.inference.window_stride)
        self.tflite_dynamic = False
        if tflite_path:
            self._load_tflite(tflite_path)

    # ------------------------------------------------------------------
    @classmethod
    def from_run(
        cls,
        run_dir: str,
        tflite_path: Optional[str] = None,
        weights: Optional[str] = None,
    ) -> "ConsoleEngine":
        from .evaluate import load_run

        model, tok, cfg = load_run(run_dir, weights=weights)
        return cls(model, tok, cfg, tflite_path=tflite_path)

    def _load_tflite(self, path: str) -> None:
        self.interpreter = tf.lite.Interpreter(
            model_path=path, num_threads=self.cfg.inference.tflite_threads
        )
        self.interpreter.allocate_tensors()
        inp = self.interpreter.get_input_details()[0]
        self.tflite_input = inp
        shape = inp["shape"].tolist()
        signature = inp.get("shape_signature", inp["shape"]).tolist()
        if len(shape) != 3:
            raise ValueError(f"unexpected TFLite input shape: {shape}")
        if signature[1] == -1:
            self.tflite_dynamic = True
            self.window_chunks = 0
        elif shape[1] > 0:
            self.tflite_dynamic = False
            self.window_chunks = int(shape[1])
        if shape[2] > 0 and int(shape[2]) != self.cfg.model.chunk_size:
            raise ValueError(
                f"TFLite chunk size {shape[2]} != model.chunk_size "
                f"{self.cfg.model.chunk_size}"
            )
        self.tflite_shape = shape
        for detail in self.interpreter.get_output_details():
            name = detail["name"].lower()
            if "chunk" in name:
                self.tflite_outputs["chunk"] = detail
            elif "completed" in name:
                self.tflite_outputs["completed"] = detail
            elif "doc" in name:
                self.tflite_outputs["doc"] = detail
        # The concrete-function converter often names outputs
        # StatefulPartitionedCall:N, so fall back to shapes:
        #   chunk [1, C, 4], doc [1, 4], completed [1, 1]
        for detail in self.interpreter.get_output_details():
            shape = [int(x) for x in detail["shape"]]
            if len(shape) == 3 and shape[-1] == int(self.cfg.model.heads.chunk_classes):
                self.tflite_outputs.setdefault("chunk", detail)
            elif len(shape) == 2 and shape[-1] == int(self.cfg.model.heads.doc_classes):
                self.tflite_outputs.setdefault("doc", detail)
            elif len(shape) >= 1 and shape[-1] == 1:
                self.tflite_outputs.setdefault("completed", detail)
        missing = [k for k in ("chunk", "doc") if k not in self.tflite_outputs]
        if missing:
            raise ValueError(
                "could not identify TFLite outputs "
                f"{[(d['name'], d['shape'].tolist()) for d in self.interpreter.get_output_details()]}"
            )
        if self.window_stride > 0 and self.window_chunks > 0:
            self.window_stride = min(self.window_stride, self.window_chunks)
        meta_path = Path(str(path) + ".json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                self.tflite_probabilities = bool(meta.get("outputs_probabilities"))
            except Exception:
                self.tflite_probabilities = False

    # ------------------------------------------------------------------
    def _windows(self, n_chunks: int) -> List[Tuple[int, int]]:
        if (self.interpreter is not None and self.tflite_dynamic) or not self.window_chunks:
            return [(0, n_chunks)]
        size = max(1, self.window_chunks)
        stride = max(1, min(self.window_stride, size))
        if n_chunks <= size:
            return [(0, n_chunks)]
        starts = list(range(0, max(1, n_chunks - size + 1), stride))
        if starts[-1] + size < n_chunks:
            starts.append(n_chunks - size)
        return [(s, s + size) for s in starts]

    def _keras_window(self, batch: np.ndarray):
        pred = self.model(tf.constant(batch, dtype=tf.int32), training=False)
        chunk = pred["chunk_logits"].numpy()
        doc = pred["doc_logits"].numpy()
        completed = (
            pred["completed_logits"].numpy().reshape(-1)
            if "completed_logits" in pred
            else np.zeros((batch.shape[0],), dtype=np.float32)
        )
        return chunk, doc, completed

    def _dequantize(self, detail, data: np.ndarray) -> np.ndarray:
        if detail["dtype"] in (np.int8, np.uint8):
            scale, zero = detail["quantization"]
            if scale:
                return (data.astype(np.float32) - zero) * scale
        return data.astype(np.float32)

    def _tflite_window(self, batch: np.ndarray):
        chunk_all, doc_all, comp_all = [], [], []
        for row in batch:
            if self.tflite_dynamic:
                self.interpreter.resize_tensor_input(
                    self.tflite_input["index"],
                    [1, int(row.shape[0]), int(row.shape[1])],
                )
                self.interpreter.allocate_tensors()
            self.interpreter.set_tensor(self.tflite_input["index"], row[None, ...])
            self.interpreter.invoke()
            chunk = self._dequantize(
                self.tflite_outputs["chunk"],
                self.interpreter.get_tensor(self.tflite_outputs["chunk"]["index"]),
            )
            doc = self._dequantize(
                self.tflite_outputs["doc"],
                self.interpreter.get_tensor(self.tflite_outputs["doc"]["index"]),
            )
            chunk_all.append(chunk[0])
            doc_all.append(doc[0])
            if "completed" in self.tflite_outputs:
                comp = self._dequantize(
                    self.tflite_outputs["completed"],
                    self.interpreter.get_tensor(
                        self.tflite_outputs["completed"]["index"]
                    ),
                )
                comp_all.append(float(np.asarray(comp).reshape(-1)[0]))
            else:
                comp_all.append(0.0)
        return (
            np.stack(chunk_all, axis=0),
            np.stack(doc_all, axis=0),
            np.asarray(comp_all, dtype=np.float32),
        )

    # ------------------------------------------------------------------
    def encode_chunk_features(self, ids_arr: np.ndarray):
        """Return per-chunk features and logits for ``ids_arr [C, 16]``.

        Only the float Keras backend exposes intermediate features; int8
        streaming falls back to :meth:`analyze_lines`.
        """
        if self.model is None:
            raise RuntimeError("chunk features require the Keras backend")
        root = self.model.get_layer("console_fcn")
        batch = tf.constant(ids_arr[None, ...], dtype=tf.int32)
        feat, logits, _ = root.encode_chunks(batch, training=False)
        return feat.numpy()[0], logits.numpy()[0]

    def document_from_pools(
        self,
        feat_sum: np.ndarray,
        feat_max: np.ndarray,
        prob_sum: np.ndarray,
        prob_max: np.ndarray,
        count: int,
    ):
        """Run the document/completion heads over accumulated chunk pools."""
        if self.model is None:
            raise RuntimeError("document pools require the Keras backend")
        root = self.model.get_layer("console_fcn")
        denom = max(float(count), 1.0)
        mode = self.cfg.model.pool.doc_pool
        parts = []
        if mode in ("mean", "mean_max"):
            parts.append(feat_sum / denom)
        if mode in ("max", "mean_max"):
            parts.append(feat_max)
        feat = np.concatenate(parts, axis=0)
        if self.cfg.model.pool.doc_use_chunk_probs:
            prob_parts = []
            if mode in ("mean", "mean_max"):
                prob_parts.append(prob_sum / denom)
            if mode in ("max", "mean_max"):
                prob_parts.append(prob_max)
            feat = np.concatenate([feat] + prob_parts, axis=0)
        tensor = tf.constant(feat[None, :], dtype=tf.float32)
        tensor = root.doc_dense(tensor)
        tensor = root.doc_dropout(tensor, training=False)
        doc_logits = root.doc_head(tensor).numpy()[0]
        if root.completed_head is not None:
            completed_logit = float(root.completed_head(tensor).numpy()[0, 0])
        else:
            completed_logit = 0.0
        return doc_logits, completed_logit

    # ------------------------------------------------------------------
    def analyze_lines(self, lines: Sequence[str]) -> Dict:
        lines = [str(x).rstrip("\r\n") for x in lines]
        ids, line_ids, tokens = self.tokenizer.encode_document(lines)
        if not ids:
            ids = [self.tokenizer.pad_id]
            line_ids = []
            tokens = [self.tokenizer.config.pad_token]
        chunk_size = self.cfg.model.chunk_size
        n_chunks = int(math.ceil(len(ids) / chunk_size))
        ids_arr = np.zeros((n_chunks, chunk_size), dtype=np.int32)
        for c in range(n_chunks):
            piece = ids[c * chunk_size : (c + 1) * chunk_size]
            ids_arr[c, : len(piece)] = piece

        size = max(1, self.window_chunks or n_chunks)
        chunk_prob_sum = np.zeros((n_chunks, self.cfg.model.heads.chunk_classes))
        chunk_prob_count = np.zeros((n_chunks,))
        doc_probs: List[np.ndarray] = []
        completed_probs: List[float] = []

        windows = self._windows(n_chunks)
        batch = np.zeros((len(windows), size, chunk_size), dtype=np.int32)
        for wi, (start, end) in enumerate(windows):
            piece = ids_arr[start:end]
            batch[wi, : piece.shape[0]] = piece
        bs = max(1, self.cfg.inference.batch_windows)
        for off in range(0, len(windows), bs):
            sub = batch[off : off + bs]
            if self.interpreter is not None:
                chunk_logits, doc_logits, comp_logits = self._tflite_window(sub)
            else:
                chunk_logits, doc_logits, comp_logits = self._keras_window(sub)
            if self.interpreter is not None and self.tflite_probabilities:
                chunk_p = np.asarray(chunk_logits, dtype=np.float64)
                doc_p = np.asarray(doc_logits, dtype=np.float64)
                comp_p = np.asarray(comp_logits, dtype=np.float64)
            else:
                chunk_p = _softmax(chunk_logits)
                doc_p = _softmax(doc_logits)
                comp_p = _sigmoid(comp_logits)
            for k, (start, end) in enumerate(windows[off : off + bs]):
                real = min(end, n_chunks) - start
                chunk_prob_sum[start:end] += chunk_p[k, :real]
                chunk_prob_count[start:end] += 1.0
                doc_probs.append(doc_p[k])
                completed_probs.append(float(comp_p[k]))
        chunk_probs = chunk_prob_sum / np.maximum(chunk_prob_count[:, None], 1.0)

        doc_prob_mat = np.stack(doc_probs, axis=0)
        if self.cfg.inference.doc_aggregate == "max_severity" and len(doc_probs) > 1:
            doc_p = np.max(doc_prob_mat, axis=0)
            doc_p = doc_p / max(float(doc_p.sum()), 1e-9)
        else:
            doc_p = doc_prob_mat.mean(axis=0)
        completed_p = float(np.mean(completed_probs)) if completed_probs else 0.0

        class_names = list(self.cfg.model.heads.class_names)
        chunk_records = []
        for c in range(n_chunks):
            start = c * chunk_size
            end = min((c + 1) * chunk_size, len(ids))
            chunk_tokens = [
                i for i in range(start, end) if i < len(line_ids)
            ]
            touched = sorted({int(line_ids[i]) for i in chunk_tokens})
            probs = chunk_probs[c]
            label_idx = int(np.argmax(probs))
            chunk_records.append(
                {
                    "index": c,
                    "token_start": start,
                    "token_end": end,
                    "lines": touched,
                    "probs": {n: float(probs[i]) for i, n in enumerate(class_names)},
                    "label": class_names[label_idx],
                    "confidence": float(probs[label_idx]),
                }
            )

        # line-level aggregation
        line_sum = np.zeros((len(lines), len(class_names)))
        line_hits = np.zeros((len(lines),))
        for c, rec in enumerate(chunk_records):
            for line_no in rec["lines"]:
                if 0 <= line_no < len(lines):
                    line_sum[line_no] += chunk_probs[c]
                    line_hits[line_no] += 1.0
        line_records = []
        for i, text in enumerate(lines):
            if line_hits[i] > 0:
                probs = line_sum[i] / line_hits[i]
                label_idx = int(np.argmax(probs))
                severity = float(probs[class_names.index("fail")] + probs[class_names.index("block")]) \
                    if "fail" in class_names and "block" in class_names else 0.0
                line_records.append(
                    {
                        "index": i,
                        "text": text,
                        "label": class_names[label_idx],
                        "confidence": float(probs[label_idx]),
                        "severity": severity,
                        "probs": {n: float(probs[j]) for j, n in enumerate(class_names)},
                    }
                )
            else:
                probs = np.full((len(class_names),), 1.0 / len(class_names))
                line_records.append(
                    {
                        "index": i,
                        "text": text,
                        "label": "unknown",
                        "confidence": float(probs[class_names.index("unknown")])
                        if "unknown" in class_names
                        else 0.0,
                        "severity": 0.0,
                        "probs": {n: float(probs[j]) for j, n in enumerate(class_names)},
                    }
                )

        doc_label_idx = int(np.argmax(doc_p))
        status = class_names[doc_label_idx]
        completed = completed_p >= 0.5
        return {
            "status": status,
            "confidence": float(doc_p[doc_label_idx]),
            "probs": {n: float(doc_p[i]) for i, n in enumerate(class_names)},
            "completed": bool(completed),
            "completed_confidence": float(completed_p if completed else 1.0 - completed_p),
            "n_tokens": len(ids),
            "n_chunks": n_chunks,
            "chunk_size": chunk_size,
            "chunks": chunk_records,
            "lines": line_records,
            "tokens": tokens,
            "line_ids": line_ids,
            "backend": "tflite" if self.interpreter is not None else "keras",
        }
