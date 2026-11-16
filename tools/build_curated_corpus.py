"""Build the curated transcript corpus from the notes in examples/corpora.

The source files are line-oriented catalogues: a ``## Linux`` / ``## Windows``
header, a backticked command, and one or more fenced output blocks.  This is a
small ad-hoc format, not a general markdown parser.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "examples" / "corpora" / "curated"
OUT = ROOT / "data" / "real"

BLOCK = re.compile(
    r"(?i)(permission denied|access is denied|operation not permitted|"
    r"forbidden|denied|not authorized|unauthorized|publickey|"
    r"requires? (?:root|elevation|administrator)|locked|lock|policy|"
    r"read-only|no space left|quota|execution policy|not accessible|"
    r"bad credentials|wrong password|encrypted archive|"
    r"running scripts is disabled|swap file|too open)"
)
FAIL = re.compile(
    r"(?i)(error|failed|failure|fatal|cannot|can't|no such|not found|"
    r"undefined|invalid|exception|refused|timed out|timeout|segmentation|"
    r"out of memory|killed|not recognized|no matching|unable to|missing|"
    r"corrupt|conflict|rejected|not available|does not exist|"
    r"could not|not in gzip|not a zipfile|no upstream|mismatched types|"
    r"no rule|missing separator|expected|not overwriting|"
    r"no installed package|hash check failed|unsupportedclassversion|"
    r"wrong password|nvidia driver)"
)
SUCCESS = re.compile(
    r"(?i)(success|completed|passed|active \(running\)|0 errors|"
    r"no errors|up to date|done\.?$|ok$)"
)

DOMAIN_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^(gcc|g\+\+|make|ninja|cmake|cl|msbuild|javac)\b"), "compile"),
    (re.compile(r"^(apt|dpkg|pipx|conda|flatpak|snap|scoop|choco|nvm)\b"), "package_update"),
    (re.compile(r"^(npm|npx|yarn|pnpm)\b"), "npm"),
    (re.compile(r"^node\b"), "node"),
    (re.compile(r"^(python|pip|jupyter|ollama|whisper)\b"), "python"),
    (re.compile(r"^(docker|docker-compose)\b"), "docker"),
    (re.compile(r"^(kubectl)\b"), "kubectl"),
    (re.compile(r"^(helm)\b"), "helm"),
    (re.compile(r"^(terraform)\b"), "terraform"),
    (re.compile(r"^(ansible|ansible-playbook)\b"), "ansible"),
    (re.compile(r"^(psql|pg_isready)\b"), "postgres"),
    (re.compile(r"^(mysql|mysqladmin)\b"), "mysql"),
    (re.compile(r"^(redis-cli)\b"), "redis"),
    (re.compile(r"^(systemctl|journalctl|crontab)\b"), "systemd"),
    (re.compile(r"^(git)\b"), "git"),
    (re.compile(r"^(cargo|rustc)\b"), "cargo"),
    (re.compile(r"^(go)\b"), "go"),
    (re.compile(r"^(java)\b"), "build"),
    (re.compile(r"^(scp|rsync|ssh|ping|curl|wget|aria2c|rclone)\b"), "network"),
    (re.compile(r"^(tar|unzip|gzip|zip|7z)\b"), "archive"),
    (re.compile(r"^(ffmpeg|ffprobe|yt-dlp|convert|magick|pandoc|pdflatex)\b"), "media"),
    (re.compile(r"^(gh)\b"), "github_actions"),
    (re.compile(r"^(Get-ChildItem|Copy-Item|Remove-Item|copy|mkdir)\b"), "windows_cmd"),
    (re.compile(r"^(wsl|winget|reg|schtasks|taskkill|net|netstat|cmd)\b"), "windows_cmd"),
    (re.compile(r"^(Set-ExecutionPolicy)\b"), "powershell"),
    (re.compile(r"^(dotnet)\b"), "dotnet"),
    (re.compile(r"^(adb|fastboot|wine|steam)\b"), "system"),
    (re.compile(r"^(ssh-add|gpg)\b"), "security"),
    (re.compile(r"^(code|vim)\b"), "shell"),
]


def sanitize(text: str) -> str:
    return re.sub(r"/home/[^/\s\"']+", "/home/dev", text)


def command_of(line: str) -> str:
    spans = re.findall(r"`([^`]+)`", line)
    if not spans:
        return ""
    if line.lstrip().startswith("`cmd`:"):
        return sanitize(spans[-1].strip())
    return sanitize(spans[0].strip())


def domain_of(command: str) -> str:
    for pattern, domain in DOMAIN_RULES:
        if pattern.search(command):
            return domain
    return "real_errors"


def status_of(text: str) -> str:
    if BLOCK.search(text):
        return "block"
    if FAIL.search(text):
        return "fail"
    if SUCCESS.search(text):
        return "success"
    return "unknown"


def label_line(status: str, text: str) -> str:
    if status == "block" and BLOCK.search(text):
        return "block"
    if status == "fail" and FAIL.search(text):
        return "fail"
    if status == "success" and SUCCESS.search(text):
        return "success"
    return "neutral"


def parse_catalogue(text: str) -> List[Dict]:
    docs: List[Dict] = []
    platform = "linux"
    index = 0
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## "):
            platform = "windows" if "windows" in line.lower() else "linux"
            i += 1
            continue
        command = (
            command_of(line)
            if line.startswith("`") and not line.startswith("```")
            else ""
        )
        if not command:
            i += 1
            continue
        i += 1
        output: List[str] = []
        while i < len(lines) and not lines[i].startswith("## ") and not (
            lines[i].startswith("`") and not lines[i].startswith("```")
        ):
            if lines[i].startswith("```"):
                i += 1
                while i < len(lines) and not lines[i].startswith("```"):
                    output.append(sanitize(lines[i]))
                    i += 1
                if i < len(lines):
                    i += 1
            else:
                i += 1
        if not output:
            continue
        status = status_of("\n".join(output))
        records = [
            {"text": item, "label": label_line(status, item)}
            for item in output if item.strip()
        ]
        if not records:
            continue
        prompt = (
            f"PS C:\\src> {command}" if platform == "windows"
            else f"dev@workstation:~/proj$ {command}"
        )
        records.insert(0, {"text": prompt, "label": "neutral"})
        if status in ("fail", "block") and not any(
            record["label"] == status for record in records
        ):
            records.append(
                {
                    "text": "job failed" if status == "fail" else "access denied",
                    "label": status,
                }
            )
        docs.append(
            {
                "id": f"curated-{index:04d}",
                "source": "curated",
                "platform": platform,
                "domain": domain_of(command),
                "command": command,
                "status": status,
                "completed": status != "unknown",
                "lines": records,
            }
        )
        index += 1
    return docs


def split_of(index: int) -> str:
    if index % 10 < 7:
        return "train"
    if index % 10 < 8:
        return "val"
    return "test"


def main() -> None:
    docs: List[Dict] = []
    for path in sorted(SOURCE_DIR.glob("*.txt")):
        docs.extend(parse_catalogue(path.read_text(encoding="utf-8")))
    existing: Dict[str, List[Dict]] = {}
    for split in ("train", "val", "test"):
        path = OUT / f"{split}.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ] if path.exists() else []
        existing[split] = [row for row in rows if row.get("source") != "curated"]
    added = {"train": 0, "val": 0, "test": 0}
    for index, doc in enumerate(docs):
        split = split_of(index)
        existing[split].append(doc)
        added[split] += 1
    for split, rows in existing.items():
        path = OUT / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for doc in rows:
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print("parsed", len(docs), "added", added)


if __name__ == "__main__":
    main()
