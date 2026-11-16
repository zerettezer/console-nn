"""Public, low-ceremony API.

Typical use::

    from console_nn import analyze

    result = analyze(open("build.log").read())
    print(result.status)          # success | fail | block | unknown
    print(result.completed)       # did the operation reach a terminal state?
    for line in result.lines:
        print(line.label, line.text)

The default pretrained model is resolved from (in order):

1. an explicit ``model_dir`` argument,
2. the ``CONSOLE_NN_MODEL`` environment variable,
3. ``console_nn/pretrained/cpu`` shipped with the package,
4. ``artifacts/cpu`` (then ``artifacts/default``) relative to the current directory.

A run directory contains ``config.json``, ``tokenizer.json`` and
``weights.h5``.  An int8 ``.tflite`` file can be used instead of Keras weights
by passing ``tflite=True``.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

__all__ = [
    "ChunkPrediction",
    "LinePrediction",
    "LogAnalysis",
    "ConsoleAnalyzer",
    "analyze",
    "analyze_lines",
    "available_models",
    "get_default_analyzer",
]


# --------------------------------------------------------------------------
# result objects
# --------------------------------------------------------------------------
@dataclass
class ChunkPrediction:
    index: int
    label: str
    confidence: float
    token_start: int
    token_end: int
    lines: List[int]
    probs: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "label": self.label,
            "confidence": round(self.confidence, 6),
            "token_start": self.token_start,
            "token_end": self.token_end,
            "lines": [int(x) for x in self.lines],
            "probs": {k: round(v, 6) for k, v in self.probs.items()},
        }


@dataclass
class LinePrediction:
    index: int
    text: str
    label: str
    confidence: float
    severity: float
    probs: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "text": self.text,
            "label": self.label,
            "confidence": round(self.confidence, 6),
            "severity": round(self.severity, 6),
            "probs": {k: round(v, 6) for k, v in self.probs.items()},
        }


@dataclass
class LogAnalysis:
    status: str
    confidence: float
    probs: Dict[str, float]
    completed: bool
    completed_confidence: float
    chunks: List[ChunkPrediction]
    lines: List[LinePrediction]
    n_tokens: int
    n_chunks: int
    backend: str = "keras"
    model_dir: Optional[str] = None
    meta: Dict = field(default_factory=dict)

    # --------------------------------------------------------------
    @classmethod
    def from_raw(cls, raw: Dict, model_dir: Optional[str] = None) -> "LogAnalysis":
        chunks = [
            ChunkPrediction(
                index=int(c["index"]),
                label=str(c["label"]),
                confidence=float(c["confidence"]),
                token_start=int(c["token_start"]),
                token_end=int(c["token_end"]),
                lines=[int(x) for x in c["lines"]],
                probs={k: float(v) for k, v in c["probs"].items()},
            )
            for c in raw["chunks"]
        ]
        lines = [
            LinePrediction(
                index=int(l["index"]),
                text=str(l["text"]),
                label=str(l["label"]),
                confidence=float(l["confidence"]),
                severity=float(l["severity"]),
                probs={k: float(v) for k, v in l["probs"].items()},
            )
            for l in raw["lines"]
        ]
        return cls(
            status=str(raw["status"]),
            confidence=float(raw["confidence"]),
            probs={k: float(v) for k, v in raw["probs"].items()},
            completed=bool(raw["completed"]),
            completed_confidence=float(raw["completed_confidence"]),
            chunks=chunks,
            lines=lines,
            n_tokens=int(raw["n_tokens"]),
            n_chunks=int(raw["n_chunks"]),
            backend=str(raw.get("backend", "keras")),
            model_dir=model_dir,
        )

    # --------------------------------------------------------------
    def to_dict(self, include_tokens: bool = False, raw: Optional[Dict] = None) -> Dict:
        out = {
            "status": self.status,
            "confidence": round(self.confidence, 6),
            "probs": {k: round(v, 6) for k, v in self.probs.items()},
            "completed": self.completed,
            "completed_confidence": round(self.completed_confidence, 6),
            "n_tokens": self.n_tokens,
            "n_chunks": self.n_chunks,
            "backend": self.backend,
            "chunks": [c.to_dict() for c in self.chunks],
            "lines": [l.to_dict() for l in self.lines],
        }
        if include_tokens and raw is not None:
            out["tokens"] = raw.get("tokens", [])
            out["line_ids"] = raw.get("line_ids", [])
        return out

    def to_json(self, path: Optional[str] = None, indent: Optional[int] = 2) -> str:
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
        if path:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text

    def to_ascii(self, color: Optional[bool] = None, width: int = 24,
                 max_line_chars: int = 160, show_chunks: bool = True) -> str:
        from .viz import render_ascii

        if color is None:
            color = _auto_color()
        return render_ascii(
            self._viz_dict(), width=width, max_line_chars=max_line_chars,
            color=color, show_chunks=show_chunks,
        )

    def to_brief(self, color: Optional[bool] = None,
                 max_line_chars: int = 200) -> str:
        from .viz import render_brief

        if color is None:
            color = _auto_color()
        return render_brief(self._viz_dict(), color=color,
                            max_line_chars=max_line_chars)

    def to_png(self, path: str = "console_report.png", dpi: int = 120) -> str:
        from .viz import render_png

        return render_png(self._viz_dict(), path=path, dpi=dpi)

    def print(self, **kwargs) -> None:
        print(self.to_ascii(**kwargs))

    def _viz_dict(self) -> Dict:
        return {
            "status": self.status,
            "confidence": self.confidence,
            "probs": self.probs,
            "completed": self.completed,
            "completed_confidence": self.completed_confidence,
            "n_tokens": self.n_tokens,
            "n_chunks": self.n_chunks,
            "backend": self.backend,
            "chunks": [
                {
                    "index": c.index,
                    "label": c.label,
                    "confidence": c.confidence,
                    "token_start": c.token_start,
                    "token_end": c.token_end,
                    "lines": c.lines,
                    "probs": c.probs,
                }
                for c in self.chunks
            ],
            "lines": [
                {
                    "index": l.index,
                    "text": l.text,
                    "label": l.label,
                    "confidence": l.confidence,
                    "severity": l.severity,
                    "probs": l.probs,
                }
                for l in self.lines
            ],
        }

    def __str__(self) -> str:
        return self.to_ascii()

    def __repr__(self) -> str:
        return (
            f"LogAnalysis(status={self.status!r}, confidence={self.confidence:.3f}, "
            f"completed={self.completed}, lines={len(self.lines)}, "
            f"chunks={self.n_chunks})"
        )


# --------------------------------------------------------------------------
# analyzer
# --------------------------------------------------------------------------
class ConsoleAnalyzer:
    """Load once, analyze many logs.

    Parameters
    ----------
    model_dir:
        Run directory with config/tokenizer/weights.  ``None`` uses the
        default resolution described in the module docstring.
    tflite:
        ``True`` uses ``model_int8.tflite`` from the run directory (or the
        package's pretrained folder); a path selects a specific file.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        tflite: Union[bool, str, None] = False,
    ):
        self.model_dir = _resolve_model_dir(model_dir)
        self.tflite_path = _resolve_tflite(self.model_dir, tflite)
        self._engine = None
        self._cfg = None

    # --------------------------------------------------------------
    @property
    def config(self):
        self._ensure()
        return self._cfg

    @property
    def engine(self):
        self._ensure()
        return self._engine

    def _ensure(self):
        if self._engine is not None:
            return
        from .config import Config, nested_update
        from .engine import ConsoleEngine
        from .tokenizer import NgramTokenizer

        run = Path(self.model_dir)
        with open(run / "config.json", "r", encoding="utf-8") as fh:
            cfg = Config()
            nested_update(cfg, json.load(fh))
        tok = NgramTokenizer.load(str(run / "tokenizer.json"))
        model = None
        if self.tflite_path is None:
            from .evaluate import load_run

            model, tok, cfg = load_run(str(run))
        self._cfg = cfg
        self._engine = ConsoleEngine(
            model, tok, cfg, tflite_path=self.tflite_path
        )

    # --------------------------------------------------------------
    def analyze(self, text: str) -> LogAnalysis:
        """Analyze a raw console transcript."""
        self._ensure()
        raw = self._engine.analyze_lines(str(text).splitlines())
        return LogAnalysis.from_raw(raw, model_dir=self.model_dir)

    def analyze_lines(self, lines: Sequence[str]) -> LogAnalysis:
        self._ensure()
        raw = self._engine.analyze_lines([str(x) for x in lines])
        return LogAnalysis.from_raw(raw, model_dir=self.model_dir)

    def analyze_file(self, path: Union[str, Path]) -> LogAnalysis:
        return self.analyze(Path(path).read_text(encoding="utf-8", errors="replace"))

    def stream(self):
        from .stream import StreamAnalyzer

        return StreamAnalyzer(self)

    def analyze_command(
        self,
        command: Union[str, Sequence[str]],
        shell: bool = True,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
    ) -> LogAnalysis:
        """Run a command, capture stdout+stderr and analyze the transcript."""
        if isinstance(command, str):
            proc = subprocess.run(
                command,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
        else:
            proc = subprocess.run(
                list(command),
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
        text = (proc.stdout or "") + (proc.stderr or "")
        result = self.analyze(text)
        result.meta.update(
            {
                "command": command if isinstance(command, str) else " ".join(command),
                "returncode": proc.returncode,
            }
        )
        return result

    def __repr__(self) -> str:
        return (
            f"ConsoleAnalyzer(model_dir={self.model_dir!r}, "
            f"tflite={self.tflite_path!r})"
        )


# --------------------------------------------------------------------------
# defaults / helpers
# --------------------------------------------------------------------------
def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def _resolve_model_dir(path: Optional[str]) -> str:
    candidates: List[Path] = []
    if path:
        candidates.append(Path(path).expanduser())
    env = os.environ.get("CONSOLE_NN_MODEL")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(_package_dir() / "pretrained" / "cpu")
    candidates.append(_package_dir() / "pretrained")
    candidates.append(Path.cwd() / "artifacts" / "cpu")
    candidates.append(Path.cwd() / "artifacts" / "default")
    for cand in candidates:
        for folder in (cand, cand / "model"):
            if (folder / "config.json").exists() and (folder / "tokenizer.json").exists():
                return str(folder.resolve())
    raise FileNotFoundError(
        "No pretrained console_nn model found. Train one with "
        "`python -m console_nn train ...`, then copy the run to "
        "console_nn/pretrained/ or set CONSOLE_NN_MODEL=/path/to/run.\n"
        "Tried: " + ", ".join(str(c) for c in candidates)
    )


def _resolve_tflite(model_dir: str, tflite: Union[bool, str, None]) -> Optional[str]:
    if not tflite:
        return None
    if isinstance(tflite, str):
        path = Path(tflite).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"TFLite model not found: {path}")
        return str(path)
    run = Path(model_dir)
    for name in ("model_int8.tflite", "model_float16.tflite"):
        for folder in (run, run / "quant"):
            path = folder / name
            if path.exists():
                return str(path)
    raise FileNotFoundError(
        f"No quantized model in {run}. Run `python -m console_nn quantize` first."
    )


def _auto_color() -> bool:
    import sys

    return bool(getattr(sys.stdout, "isatty", lambda: False)())


_DEFAULT_ANALYZER: Optional[ConsoleAnalyzer] = None


def get_default_analyzer() -> ConsoleAnalyzer:
    global _DEFAULT_ANALYZER
    if _DEFAULT_ANALYZER is None:
        _DEFAULT_ANALYZER = ConsoleAnalyzer()
    return _DEFAULT_ANALYZER


def analyze(text: str, model_dir: Optional[str] = None,
            tflite: Union[bool, str, None] = False) -> LogAnalysis:
    """One-call convenience: text in, :class:`LogAnalysis` out."""
    if model_dir is None and not tflite:
        analyzer = get_default_analyzer()
    else:
        analyzer = ConsoleAnalyzer(model_dir=model_dir, tflite=tflite)
    return analyzer.analyze(text)


def analyze_lines(lines: Sequence[str], model_dir: Optional[str] = None,
                  tflite: Union[bool, str, None] = False) -> LogAnalysis:
    analyzer = (
        get_default_analyzer()
        if model_dir is None and not tflite
        else ConsoleAnalyzer(model_dir=model_dir, tflite=tflite)
    )
    return analyzer.analyze_lines(lines)


def available_models() -> List[Dict]:
    """List discoverable run directories and their metadata."""
    roots = []
    env = os.environ.get("CONSOLE_NN_MODEL")
    if env:
        roots.append(Path(env).expanduser())
    roots.append(_package_dir() / "pretrained")
    roots.append(Path.cwd() / "artifacts")
    found = []
    for root in roots:
        if not root.exists():
            continue
        if (root / "config.json").exists():
            found.append(root)
        elif (root / "model" / "config.json").exists():
            found.append(root / "model")
        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue
            if (sub / "config.json").exists():
                found.append(sub)
            elif (sub / "model" / "config.json").exists():
                found.append(sub / "model")
    out = []
    seen = set()
    for run in found:
        key = str(run.resolve())
        if key in seen:
            continue
        seen.add(key)
        entry = {"path": key}
        root = run.parent if run.name == "model" else run
        summary = next(
            (p for p in (root / "metrics.json", root / "metrics" / "summary.json",
                         root / "summary.json") if p.exists()),
            None,
        )
        if summary is not None:
            try:
                entry.update(json.loads(summary.read_text(encoding="utf-8")))
            except Exception:
                pass
        entry["model"] = str(run)
        entry["tflite"] = sorted(
            p.name for p in list(run.glob("*.tflite")) + list((run / "quant").glob("*.tflite"))
        )
        out.append(entry)
    return out
