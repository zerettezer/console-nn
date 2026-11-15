"""Collect real command sessions from the local Linux machine.

Runs only safe, local commands in a temporary workspace.  The resulting JSONL
is used as a real-world holdout and as additional training data, so evaluation
is no longer synthetic-only.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import getpass
import socket
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "real"

BLOCK_RE = re.compile(
    r"(?i)(permission denied|access is denied|operation not permitted|"
    r"are you root|requires? (?:root|elevation)|authentication failed|"
    r"publickey|host key verification|forbidden|not authorized|"
    r"lock-frontend|policy|quota|unauthorized|denied)"
)
FAIL_RE = re.compile(
    r"(?i)(traceback|modulenotfounderror|importerror|syntaxerror|"
    r"error[:\[]|fatal|failed|failure|cannot|can't|"
    r"no such file|not found|undefined reference|undefined:|"
    r"invalid|exception|refused|timed out|timeout|"
    r"segmentation fault|out of memory|killed|not recognized|"
    r"no matching|unable to|missing|corrupt)"
)
SUCCESS_RE = re.compile(
    r"(?i)(success|completed|passed|active \(running\)|"
    r"0 errors|no errors|up to date|done\.?$|ok$)"
)


def _label(status: str, text: str) -> str:
    if status == "block" and BLOCK_RE.search(text):
        return "block"
    if status == "fail" and FAIL_RE.search(text):
        return "fail"
    if status == "success" and SUCCESS_RE.search(text):
        return "success"
    return "neutral"


def run(cmd: str, cwd: Path, timeout: float = 20.0) -> Tuple[int, str]:
    env = dict(os.environ)
    env.update({"LC_ALL": "C", "LANG": "C", "NO_COLOR": "1"})
    try:
        proc = subprocess.run(
            cmd, shell=True, cwd=str(cwd), capture_output=True,
            text=True, errors="replace", timeout=timeout, env=env,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + (exc.stderr or "")
        return 124, str(out) + "\ntimeout: command exceeded limit"


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _setup(root: Path) -> None:
    (root / "hello.py").write_text("print('hello from python')\n")
    (root / "bad.py").write_text("def f(:\n    pass\n")
    (root / "raise.py").write_text("raise ValueError('bad value')\n")
    (root / "main.c").write_text(
        '#include <stdio.h>\nint main(void){ printf("hello c\\n"); return 0; }\n'
    )
    (root / "bad.c").write_text("int main(void){ return missing; }\n")
    (root / "lib.c").write_text("int add(int a,int b){return a+b;}\n")
    (root / "Makefile").write_text(
        "all: hello\nhello: main.c\n\tgcc -Wall -o hello main.c\n"
        "fail:\n\tgcc -Wall -o bad bad.c\n"
    )
    (root / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.10)\nproject(demo C)\n"
        "add_executable(demo main.c)\n"
    )
    (root / "hello.go").write_text('package main\nimport "fmt"\nfunc main(){fmt.Println("hello go")}\n')
    (root / "bad.go").write_text("package main\nfunc main(){ x := }\n")
    (root / "app.js").write_text("console.log('hello node')\n")
    (root / "bad.js").write_text("function {\n")
    (root / "Hello.java").write_text('public class Hello { public static void main(String[] a){ System.out.println("hello java"); } }\n')
    (root / "Bad.java").write_text('public class Bad { public static void main(String[] a){ int x = ; } }\n')
    (root / "data.csv").write_text("id,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30\n")
    (root / "access.log").write_text(
        '10.0.0.14 - - [06/Nov/2026:10:14:22 +0000] "GET /api/orders HTTP/1.1" 200 512\n'
        '10.0.0.22 - - [06/Nov/2026:10:14:23 +0000] "GET /api/search HTTP/1.1" 503 128\n'
    )
    (root / "secret.txt").write_text("top secret\n")
    os.chmod(root / "secret.txt", 0)
    (root / "dir").mkdir(exist_ok=True)
    (root / "dir" / "nested.txt").write_text("nested\n")


def scenarios(root: Path) -> List[Tuple[str, str]]:
    cases: List[Tuple[str, str]] = []
    add = cases.append

    # Python
    for cmd in [
        "python3 hello.py",
        "python3 -c \"print(sum(range(10)))\"",
        "python3 raise.py",
        "python3 missing.py",
        "python3 bad.py",
        "python3 -c \"import not_a_real_module\"",
        "cat secret.txt",
        "python3 -c \"open('secret.txt').read()\"",
    ]:
        add(("python", cmd))

    # C / make / cmake
    for cmd in [
        "gcc -Wall -o hello main.c",
        "gcc -Wall -o bad bad.c",
        "gcc -Wall -o missing missing.c",
        "gcc -Wall -o hello main.c -lnotreal",
        "make", "make fail", "make clean",
        "cmake -S . -B build",
        "cmake --build build",
        "cmake -S missing -B build2",
    ]:
        add(("compile", cmd))

    # Go / Node / Java
    for cmd in [
        "go run hello.go", "go build bad.go", "go build missing.go",
        "node app.js", "node bad.js", "node missing.js",
        "npm run does-not-exist",
        "javac Hello.java", "javac Bad.java", "java Hello", "java Missing",
    ]:
        add(("build", cmd))

    # Git
    for cmd in [
        "git init repo", "git -C repo status", "git -C repo log",
        "git -C repo checkout missing-branch",
        "git -C repo push origin main",
        "git clone missing-repo cloned",
        "git -C repo remote -v",
    ]:
        add(("git", cmd))

    # Shell / filesystem
    for cmd in [
        "ls -la", "ls missing-dir", "pwd", "uname -a", "id",
        "whoami", "df -h", "free -h", "ps -ef | head -n 5",
        "cat data.csv", "cat missing.txt", "head -n 2 data.csv",
        "tail -n 2 access.log", "wc -l access.log",
        "grep -n 503 access.log", "grep -n nomatch access.log",
        "awk -F, 'NR>1 {sum+=$3} END {print sum}' data.csv",
        "sed -n '1,2p' data.csv", "find . -name '*.log'",
        "find missing-dir -name '*'\n", "stat secret.txt",
        "cp secret.txt copy.txt", "mkdir -p dir/new",
        "rm missing-file", "touch new-file",
    ]:
        add(("shell", cmd))

    # Local HTTP and networking
    add(("network", "curl --max-time 2 http://127.0.0.1:9/"))
    add(("network", "getent hosts no-such-host.invalid"))
    add(("network", "ping -c 1 127.0.0.1"))
    add(("network", "ss -tulpn"))
    add(("network", "python3 -c \"import socket;"
                     "s=socket.socket(); s.bind(('127.0.0.1',0));"
                     "p=s.getsockname()[1]; s.listen(1);"
                     "t=socket.socket(); t.bind(('127.0.0.1',p))\""))
    add(("media", "ffmpeg -hide_banner -f lavfi -i color=c=black:s=64x64:d=1 -y out.mp4"))
    add(("media", "ffprobe -v error -show_entries stream=codec_name -of csv=p=0 out.mp4"))
    add(("media", "ffmpeg -hide_banner -i missing.mp4 -f null -"))

    # System
    for cmd in [
        "systemctl status definitely-not-a-service --no-pager",
        "systemctl list-units --no-pager | head -n 5",
        "journalctl -n 5 --no-pager",
        "apt-cache policy bash",
        "dpkg -l | head -n 5",
        "docker ps", "kubectl version --client", "terraform version",
        "cargo --version", "cmake --version", "gcc --version | head -n 1",
    ]:
        add(("system", cmd))

    return cases


def _split(index: int) -> str:
    # deterministic split, ~70/15/15
    if index % 10 < 7:
        return "train"
    if index % 10 < 8:
        return "val"
    return "test"


def collect() -> Dict[str, int]:
    root = Path(tempfile.mkdtemp(prefix="console-nn-real-"))
    _setup(root)
    docs = []
    cases = scenarios(root)
    for i, (domain, cmd) in enumerate(cases):
        code, text = run(cmd, root)
        user = getpass.getuser()
        host = socket.gethostname()
        text = text.replace(f"/home/{user}", "/home/dev")
        text = text.replace(user, "dev").replace(host, "workstation")
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        if not lines:
            lines = ["command produced no output"]
        if code == 0:
            status = "success"
        else:
            status = "block" if BLOCK_RE.search(text) else "fail"
        if status == "success" and not any(SUCCESS_RE.search(x) for x in lines):
            # command succeeded but had no marker; add explicit terminal line
            lines.append("command completed successfully")
        records = [
            {"text": line, "label": _label(status, line)}
            for line in lines[:200]
        ]
        docs.append({
            "id": f"real-{i:04d}",
            "source": "real",
            "platform": "linux",
            "domain": domain,
            "command": cmd,
            "status": status,
            "completed": True,
            "lines": records,
        })
    shutil.rmtree(root, ignore_errors=True)

    for split in ("train", "val", "test"):
        subset = [d for i, d in enumerate(docs) if _split(i) == split]
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for doc in subset:
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(split, len(subset), path)
    return {s: sum(1 for i, _ in enumerate(docs) if _split(i) == s) for s in ("train", "val", "test")}


if __name__ == "__main__":
    print(collect())
