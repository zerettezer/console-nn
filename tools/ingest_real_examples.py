"""Ingest the concrete logs in examples/corpora/real as real holdout sessions."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCK = re.compile(r"(?i)(permission denied|access is denied|operation not permitted|denied|unauthorized|publickey|host key verification|requires? (?:root|elevation)|lock|policy|quota)")
FAIL = re.compile(r"(?i)(error|failed|failure|fatal|cannot|can't|no such|not found|undefined|invalid|exception|refused|timed out|timeout|segmentation|out of memory|killed|not recognized|no matching|unable to|missing|conflict|rejected)")
SUCCESS = re.compile(r"(?i)(success|completed|passed|active \(running\)|0 errors|no errors|up to date|done\.?$)")


def command_of(first: str) -> str:
    for sep in ("$ ", "] $ ", "> "):
        if sep in first:
            return first.split(sep, 1)[1].strip()
    return first.strip()


def label(status: str, text: str) -> str:
    if status == "block" and BLOCK.search(text):
        return "block"
    if status == "fail" and FAIL.search(text):
        return "fail"
    if status == "success" and SUCCESS.search(text):
        return "success"
    return "neutral"


def main() -> None:
    out = []
    for path in sorted((ROOT / "examples" / "corpora" / "real").glob("*.log")):
        lines = [x.rstrip() for x in path.read_text(errors="replace").splitlines() if x.strip()]
        if not lines:
            continue
        text = "\n".join(lines)
        if BLOCK.search(text):
            status = "block"
        elif FAIL.search(text):
            status = "fail"
        else:
            status = "success"
        platform = "windows" if any(
            token in path.name for token in ("msbuild", "powershell", "windows", "winget", "npm")
        ) else "linux"
        recs = [{"text": x, "label": label(status, x)} for x in lines[:200]]
        out.append({
            "id": f"realex-{path.stem}",
            "source": "real_user",
            "platform": platform,
            "domain": "real_errors",
            "command": command_of(lines[0]),
            "status": status,
            "completed": True,
            "lines": recs,
        })
    # add to the real test holdout only
    target = ROOT / "data" / "real" / "test.jsonl"
    existing = [json.loads(x) for x in target.read_text().splitlines() if x.strip()]
    seen = {d["id"] for d in existing}
    with open(target, "a", encoding="utf-8") as fh:
        for d in out:
            if d["id"] not in seen:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("added", len(out), "to", target)


if __name__ == "__main__":
    main()
