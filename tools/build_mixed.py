"""Mix synthetic splits with collected real sessions and deduplicate them.

Exact duplicate transcripts are common in synthetic output because a template
can be sampled more than once.  Test documents have priority, then validation,
then training, so no evaluation document is also present in training.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def signature(document):
    lines = "\n".join(
        line["text"].strip().lower() for line in document.get("lines", [])
    )
    return hashlib.sha256(
        (str(document.get("command", "")) + "\n" + lines).encode()
    ).hexdigest()


def unique(documents):
    seen = set()
    out = []
    for document in documents:
        key = signature(document)
        if key in seen:
            continue
        seen.add(key)
        out.append(document)
    return out


def main():
    out = ROOT / "data" / "mixed"
    out.mkdir(parents=True, exist_ok=True)
    merged = {}
    for split in ("train", "val", "test"):
        syn = load(ROOT / "data" / "synth" / f"{split}.jsonl")
        real = load(ROOT / "data" / "real" / f"{split}.jsonl")
        for d in syn:
            d.setdefault("source", "synthetic")
        for d in real:
            d.setdefault("source", "real")
        merged[split] = syn + real

    test = unique(merged["test"])
    test_sigs = {signature(d) for d in test}
    val = [
        d for d in unique(merged["val"])
        if signature(d) not in test_sigs
    ]
    val_sigs = {signature(d) for d in val}
    train = [
        d for d in unique(merged["train"])
        if signature(d) not in test_sigs and signature(d) not in val_sigs
    ]

    for split, docs in (("train", train), ("val", val), ("test", test)):
        removed = len(merged[split]) - len(docs)
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            for d in docs:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(
            split,
            "synthetic",
            sum(d.get("source") == "synthetic" for d in docs),
            "real",
            sum(d.get("source") != "synthetic" for d in docs),
            "total",
            len(docs),
            "removed",
            removed,
        )


if __name__ == "__main__":
    main()
