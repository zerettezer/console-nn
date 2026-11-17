"""Audit exact and command-level overlap between the mixed splits."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "mixed"


def load(split):
    return [
        json.loads(line)
        for line in (DATA / f"{split}.jsonl").read_text().splitlines()
        if line.strip()
    ]


def signature(doc):
    text = "\n".join(line["text"].strip().lower() for line in doc["lines"])
    return hashlib.sha256((str(doc.get("command", "")) + "\n" + text).encode()).hexdigest()


def command_key(doc):
    return (str(doc.get("platform", "")), str(doc.get("command", "")).strip())


def unique(rows):
    seen = set()
    out = []
    for row in rows:
        key = signature(row)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def main():
    splits = {name: load(name) for name in ("train", "val", "test")}
    sets = {
        name: {signature(doc) for doc in rows}
        for name, rows in splits.items()
    }
    report = {
        "sizes": {name: len(rows) for name, rows in splits.items()},
        "exact": {
            "train_test": len(sets["train"] & sets["test"]),
            "val_test": len(sets["val"] & sets["test"]),
            "train_val": len(sets["train"] & sets["val"]),
            "within_train": len(splits["train"]) - len(sets["train"]),
            "within_val": len(splits["val"]) - len(sets["val"]),
            "within_test": len(splits["test"]) - len(sets["test"]),
        },
        "by_source": {},
        "command_overlap": {},
    }
    for source in ("synthetic", "real", "curated", "real_user"):
        by = {
            name: [d for d in rows if d.get("source") == source]
            for name, rows in splits.items()
        }
        report["by_source"][source] = {
            "sizes": {name: len(rows) for name, rows in by.items()},
            "train_test_exact": len(
                {signature(d) for d in by["train"]}
                & {signature(d) for d in by["test"]}
            ),
        }

    for source in ("real", "curated"):
        train_commands = Counter(
            command_key(d) for d in splits["train"] if d.get("source") == source
        )
        test_commands = Counter(
            command_key(d) for d in splits["test"] if d.get("source") == source
        )
        overlap = {
            f"{platform}:{command}": {
                "train": train_commands[(platform, command)],
                "test": test_commands[(platform, command)],
            }
            for platform, command in test_commands
            if (platform, command) in train_commands
        }
        report["command_overlap"][source] = {
            "test_documents_with_train_command": sum(v["test"] for v in overlap.values()),
            "test_documents": sum(test_commands.values()),
            "pairs": overlap,
        }

    # The synthetic real_errors family and the example sessions are separate
    # sources, but check whether the generator reproduced the same transcripts.
    real_errors = {
        signature(d)
        for d in splits["train"]
        if d.get("source") == "synthetic" and d.get("domain") == "real_errors"
    }
    examples = [
        d for d in splits["test"] if d.get("source") == "real_user"
    ]
    report["example_vs_synthetic_real_errors"] = {
        "examples": len(examples),
        "exact_matches": sum(signature(d) in real_errors for d in examples),
    }
    target = ROOT / "results" / "cpu" / "split_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
