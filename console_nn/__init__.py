"""console_nn — length-unlimited console-log analysis with a tiny FCN.

Quick start::

    from console_nn import analyze

    result = analyze(open("build.log").read())
    print(result.status, result.completed)
    print(result.to_ascii())
"""
from __future__ import annotations

from .api import (
    ChunkPrediction,
    ConsoleAnalyzer,
    LinePrediction,
    LogAnalysis,
    analyze,
    analyze_lines,
    available_models,
    get_default_analyzer,
)

__version__ = "0.1.0"

__all__ = [
    "ChunkPrediction",
    "ConsoleAnalyzer",
    "LinePrediction",
    "LogAnalysis",
    "analyze",
    "analyze_lines",
    "available_models",
    "get_default_analyzer",
    "__version__",
]
