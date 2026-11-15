"""Small text utilities shared by the tokenizer, synth and visualiser."""
from __future__ import annotations

import html
import re
from typing import Iterable, List

ANSI_RE = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"          # CSI
    r"|\x1b\][^\x07]*(?:\x07|\x1b\\)"   # OSC
    r"|\x1b[@-_]"                        # two byte escapes
)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PATH_RE = re.compile(
    r"(?<![\w])("
    r"(?:[A-Za-z]:\\|\\\\)[^\s\"'<>|]+"          # windows path
    r"|~/[^\s\"'<>|]+"                            # home path
    r"|/[\w.\-]+(?:/[\w.\-]+)+"                   # unix path
    r")"
)
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b")
NUM_RE = re.compile(r"\b(?:0x[0-9a-fA-F]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?\b")
UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
NUMERIC_RE = re.compile(r"\d+(?:\.\d+)?")
WORD_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_./\\:-]*"      # words, identifiers, flags, paths
    r"|\d+(?:\.\d+)?"                     # plain numbers
    r"|[^\sA-Za-z0-9_]"                   # punctuation, one char at a time
)


def strip_ansi(text: str) -> str:
    text = ANSI_RE.sub("", text)
    return CONTROL_RE.sub("", text)


def normalize_text(
    text: str,
    lowercase: bool = True,
    normalize_numbers: bool = True,
    normalize_paths: bool = True,
    normalize_ips: bool = True,
    collapse_spaces: bool = True,
    numeric_mode: str = "digits",
) -> str:
    text = strip_ansi(text)
    if "\r" in text:
        # progress bars: keep only the final state of the carriage-return line
        text = text.split("\r")[-1]
    if normalize_paths:
        text = PATH_RE.sub("<path>", text)
    if normalize_ips:
        text = IP_RE.sub("<ip>", text)
    text = TIME_RE.sub("<time>", text)
    text = UUID_RE.sub("<uuid>", text)
    if numeric_mode == "mask" and normalize_numbers:
        text = NUM_RE.sub("<num>", text)
    if lowercase:
        text = text.lower()
    if collapse_spaces:
        text = re.sub(r"[ \t]+", " ", text).strip()
    return text


def tokenize(text: str, numeric_mode: str = "digits") -> List[str]:
    """Tokenize text; numbers are split into a marker plus digit tokens.

    ``digits`` preserves the actual value with a tiny vocabulary: ``8080``
    becomes ``<NUM> <8> <0> <8> <0>``.  ``bucket`` maps numbers to magnitude
    buckets, ``keep`` keeps them as whole tokens, ``mask`` is handled during
    normalization.
    """
    out: List[str] = []
    for token in WORD_RE.findall(text):
        if not NUMERIC_RE.fullmatch(token):
            out.append(token)
            continue
        if numeric_mode == "digits":
            out.append("<NUM>")
            for char in token:
                out.append(f"<{char}>" if char.isdigit() else "<.>")
        elif numeric_mode == "bucket":
            try:
                value = abs(float(token))
            except ValueError:
                value = 0.0
            if value <= 15:
                out.append(f"<NUM{int(value)}>")
            elif value <= 255:
                out.append("<NUM_BYTE>")
            elif value <= 65535:
                out.append("<NUM_16>")
            else:
                out.append("<NUM_BIG>")
        else:
            out.append(token)
    return out


def truncate(text: str, width: int, ellipsis: str = "...") -> str:
    if len(text) <= width:
        return text
    if width <= len(ellipsis):
        return text[:width]
    return text[: width - len(ellipsis)] + ellipsis


def html_escape(text: str) -> str:
    return html.escape(text, quote=True)


def redact(text: str) -> str:
    """Remove obviously secret-looking material before logging."""
    text = re.sub(
        r"(?i)(password|passwd|token|secret|api[_-]?key)\s*[:=]\s*\S+",
        r"\1=<redacted>",
        text,
    )
    return text


def iter_lines(text: str) -> Iterable[str]:
    for line in text.splitlines():
        yield line.rstrip("\n")
