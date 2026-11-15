"""Mix synthetic splits with collected real sessions."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(path: Path):
    if not path.exists(): return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def main():
    out = ROOT / "data" / "mixed"
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        syn = load(ROOT / "data" / "synth" / f"{split}.jsonl")
        real = load(ROOT / "data" / "real" / f"{split}.jsonl")
        for d in syn: d.setdefault("source", "synthetic")
        for d in real: d.setdefault("source", "real")
        docs = syn + real
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            for d in docs: fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(split, "synthetic", len(syn), "real", len(real), "total", len(docs))

if __name__ == "__main__": main()
