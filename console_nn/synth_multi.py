"""Multi-command sessions: 2-4 real commands composed into one transcript."""
from __future__ import annotations

import random
from typing import List

from .synth import Line, _num, _p, _prompt, context_lines

_SUB_DOMAINS = [
    "pip", "pytest", "git", "build", "compile", "network", "database",
    "cloud", "systemd", "python", "package", "package_update", "media",
    "data_process", "docker", "commands", "conda", "filesystem", "backup",
    "archive", "security",
]


def gen_multi(rng: random.Random, platform: str, status: str, cfg):
    import console_nn.synth as S

    n = _num(rng, 2, 4)
    domains = rng.sample(_SUB_DOMAINS, n)
    chunks: List[Line] = []
    for i, domain in enumerate(domains):
        if i == 0:
            sub_status = status
        elif status == "fail":
            sub_status = "success"
        elif status == "block":
            sub_status = "success"
        elif status == "unknown":
            sub_status = "unknown"
        else:
            sub_status = "success"
        try:
            lines, _ = S._GENERATORS[domain](rng, platform, sub_status, cfg)
        except Exception:
            lines = []
        if lines:
            if chunks:
                chunks.append(("", "neutral"))
            chunks.extend(lines)
        if rng.random() < 0.45:
            chunks.extend(context_lines(rng, _num(rng, 1, 4)))
    if not chunks:
        chunks = [(_prompt(rng, platform, "~", "true"), "neutral")]
    return chunks, (0 if status == "success" else 1)


MULTI_GENERATORS = {"multi": gen_multi, "multi_chain": gen_multi}
