"""ASCII and standalone HTML rendering of an analysis result."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

from .text import strip_ansi, truncate

RESET = "\x1b[0m"
COLORS = {
    "success": "\x1b[32m",
    "fail": "\x1b[31m",
    "block": "\x1b[33m",
    "unknown": "\x1b[90m",
}
SYMBOLS = {"success": "✓", "fail": "✗", "block": "⊘", "unknown": "?"}
CSS_CLASS = {
    "success": "success",
    "fail": "fail",
    "block": "block",
    "unknown": "unknown",
}


def _c(text: str, label: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{COLORS.get(label, '')}{text}{RESET}"


def _bar(value: float, width: int, char: str = "█") -> str:
    value = max(0.0, min(1.0, float(value)))
    filled = int(round(value * width))
    return char * filled + "·" * max(0, width - filled)


def _counts(result: Dict) -> Dict[str, int]:
    counts = {"success": 0, "fail": 0, "block": 0, "unknown": 0}
    for chunk in result["chunks"]:
        counts[chunk["label"]] = counts.get(chunk["label"], 0) + 1
    return counts


def render_ascii(
    result: Dict,
    width: int = 24,
    max_line_chars: int = 160,
    color: bool = False,
    show_chunks: bool = True,
) -> str:
    status = result["status"]
    counts = _counts(result)
    lines: List[str] = []
    head = (
        f"RESULT: {_c(status.upper(), status, color)}  "
        f"doc_conf={result['confidence']:.3f}  "
        f"completed={'yes' if result['completed'] else 'no'} "
        f"({result['completed_confidence']:.3f})  "
        f"tokens={result['n_tokens']} chunks={result['n_chunks']} "
        f"backend={result.get('backend', 'keras')}"
    )
    lines.append(head)
    lines.append(
        "probs: " + "  ".join(
            f"{k}={v:.3f}" for k, v in result["probs"].items()
        )
    )
    lines.append(
        "chunks: " + "  ".join(f"{k}={counts.get(k, 0)}" for k in counts)
    )
    lines.append("")
    lines.append("CHUNK TIMELINE (each cell = 16 tokens, color = predicted class)")
    raw = "".join(SYMBOLS.get(c["label"], "?") for c in result["chunks"])
    for start in range(0, max(1, len(raw)), 110):
        piece = result["chunks"][start : start + 110]
        if color:
            lines.append("".join(_c(SYMBOLS.get(c["label"], "?"), c["label"], True) for c in piece))
        else:
            lines.append("".join(SYMBOLS.get(c["label"], "?") for c in piece))
    lines.append("")
    lines.append(f"LOG LINES ({len(result['lines'])})  bar = P(fail)+P(block)")
    for line in result["lines"]:
        label = line["label"]
        glyph = SYMBOLS.get(label, "?")
        bar = _bar(line["severity"], width)
        conf = line["confidence"]
        text = truncate(strip_ansi(line["text"]), max_line_chars)
        prefix = f"{line['index'] + 1:4d} {glyph} {label:7s} {bar} {line['severity']:.2f}  "
        if color:
            lines.append(_c(prefix, label, True) + text)
        else:
            lines.append(prefix + text)
    if not show_chunks:
        return "\n".join(lines)
    lines.append("")
    lines.append("CHUNK DETAILS")
    for chunk in result["chunks"]:
        probs = chunk["probs"]
        top = " ".join(f"{k}={v:.2f}" for k, v in probs.items())
        line_range = (
            f"lines {chunk['lines'][0] + 1}-{chunk['lines'][-1] + 1}"
            if chunk["lines"]
            else "lines -"
        )
        detail = (
            f"  #{chunk['index'] + 1:04d} {chunk['label']:7s} "
            f"conf={chunk['confidence']:.2f} tokens={chunk['token_start']}-"
            f"{chunk['token_end']} {line_range}  {top}"
        )
        lines.append(_c(detail, chunk["label"], color))
    return "\n".join(lines)


def print_ascii(result: Dict, **kwargs) -> None:
    print(render_ascii(result, **kwargs))


def render_brief(result: Dict, color: bool = False, max_line_chars: int = 200) -> str:
    """Short report: verdict + only high-severity lines."""
    status = result["status"]
    counts = _counts(result)
    out = [
        f"RESULT: {_c(status.upper(), status, color)}  "
        f"doc_conf={result['confidence']:.3f}  "
        f"completed={'yes' if result['completed'] else 'no'} "
        f"({result['completed_confidence']:.3f})",
        "chunks: " + "  ".join(f"{k}={v}" for k, v in counts.items()),
        "",
    ]
    problems = [
        line for line in result["lines"]
        if line["severity"] >= 0.6
    ]
    if not problems:
        out.append("No high-severity lines detected.")
        return "\n".join(out)
    out.append(f"PROBLEM LINES ({len(problems)} of {len(result['lines'])})")
    for line in problems:
        label = line["label"]
        text = truncate(strip_ansi(line["text"]), max_line_chars)
        prefix = (
            f"{line['index'] + 1:4d} {SYMBOLS.get(label, '?')} {label:7s} "
            f"{line['severity']:.2f}  "
        )
        out.append((_c(prefix, label, color) if color else prefix) + text)
    return "\n".join(out)


def render_png(result: Dict, path: str, dpi: int = 120, width: int = 1200) -> str:
    """Render a self-contained PNG report with matplotlib (Agg backend)."""
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    bg = "#0f1115"
    panel = "#171a21"
    track = "#242a35"
    fg = "#d7dae0"
    muted = "#8b93a3"
    colors = {"success": "#3fb950", "fail": "#f85149",
              "block": "#d29922", "unknown": "#6e7681"}
    status = result["status"]
    status_color = colors.get(status, "#6e7681")
    lines = result["lines"]
    top = 238
    row_h = 30
    height = top + row_h * len(lines) + 28
    fig = plt.figure(figsize=(width / 100, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_facecolor(bg)
    ax.axis("off")
    mono = "monospace"

    ax.text(24, 30, "CONSOLE LOG ANALYSIS", color=muted, fontsize=11,
            family=mono, weight="bold", va="center")
    ax.text(24, 62, status.upper(), color=status_color, fontsize=25,
            family=mono, weight="bold", va="center")
    meta = (f"doc_conf={result['confidence']:.3f}   "
            f"completed={'yes' if result['completed'] else 'no'} "
            f"({result['completed_confidence']:.3f})   "
            f"tokens={result['n_tokens']}   chunks={result['n_chunks']}   "
            f"backend={result.get('backend', 'keras')}")
    ax.text(24, 88, meta, color=muted, fontsize=9.5, family=mono, va="center")

    # class cards
    x = 24
    for name, prob in result["probs"].items():
        c = colors.get(name, muted)
        ax.add_patch(Rectangle((x, 106), 150, 40, facecolor=panel, edgecolor="none"))
        ax.add_patch(Rectangle((x, 106), 4, 40, facecolor=c, edgecolor="none"))
        ax.text(x + 14, 119, name.upper(), color=muted, fontsize=7.5,
                family=mono, va="center")
        ax.text(x + 14, 135, f"{prob:.3f}", color=fg, fontsize=12,
                family=mono, va="center")
        x += 162

    # chunk timeline
    counts = _counts(result)
    ax.text(24, 166, "CHUNK TIMELINE", color=muted, fontsize=8,
            family=mono, va="center")
    ax.text(150, 166, "   ".join(f"{k}={v}" for k, v in counts.items()),
            color=muted, fontsize=8, family=mono, va="center")
    chunks = result["chunks"]
    if chunks:
        cell = min(18.0, (width - 48) / len(chunks))
        for i, c in enumerate(chunks):
            prob = max(c["probs"].values())
            ax.add_patch(Rectangle((24 + i * cell, 176), max(1.5, cell - 1.5), 14,
                                   facecolor=colors.get(c["label"], muted),
                                   edgecolor="none", alpha=0.35 + 0.65 * prob))

    ax.text(24, 212, "LOG LINES", color=muted, fontsize=8, family=mono,
            va="center")
    ax.text(92, 212, "bar = P(fail)+P(block)", color=muted, fontsize=8,
            family=mono, va="center")
    for i, line in enumerate(lines):
        y = top + i * row_h
        label = line["label"]
        c = colors.get(label, muted)
        if i % 2 == 0:
            ax.add_patch(Rectangle((16, y - 1), width - 32, row_h - 2,
                                   facecolor="#151820", edgecolor="none"))
        ax.text(24, y + 9, f"{line['index'] + 1:4d}", color="#6e7681",
                fontsize=8, family=mono, va="center")
        ax.add_patch(Rectangle((62, y + 1), 82, 17, facecolor=c,
                               edgecolor="none", alpha=0.20))
        ax.text(68, y + 9, label, color=c, fontsize=8, family=mono, va="center")
        ax.add_patch(Rectangle((152, y + 5), 210, 9, facecolor=track,
                               edgecolor="none"))
        ax.add_patch(Rectangle((152, y + 5), 210 * max(0.0, min(1.0, line["severity"])),
                               9, facecolor=c, edgecolor="none"))
        ax.text(370, y + 9, f"{line['severity']:.2f}", color=muted, fontsize=8,
                family=mono, va="center")
        text = truncate(strip_ansi(line["text"]), 104)
        ax.text(428, y + 9, text, color=fg, fontsize=8.5, family=mono,
                va="center")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=bg, edgecolor="none",
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return str(path)
