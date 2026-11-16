"""Deterministic synthetic console-log corpus.

The generator composes realistic transcripts for Linux, Windows, Python and
system diagnostics.  Every document carries:

    lines: [{text, label}]   label in {success, fail, block, neutral}
    status: success | fail | block | unknown
    completed: bool          operation reached a terminal state
    platform / domain / command

The corpus is intentionally noisy: ANSI colours, timestamps, progress bars,
truncated sessions, benign lines that merely *mention* errors, and block
cases (permissions, policy, authentication) instead of plain failures.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .config import SynthConfig

Line = Tuple[str, str]  # (text, label)


# --------------------------------------------------------------------------
# random material
# --------------------------------------------------------------------------
USERS = ["alex", "dev", "root", "ci", "builder", "maria", "ops", "sam", "jenkins",
         "dmitry", "katya", "sre", "deploy", "qa", "ml", "anna", "ivan", "test",
         "runner", "admin", "analyst", "nightly", "backup", "service"]
HOSTS = ["workstation", "buildbox", "prod-01", "dev-laptop", "runner-7", "srv-edge",
         "edge-02", "gpu-01", "runner-03", "lab-pc", "build-07", "prod-eu-2",
         "staging-01", "db-primary", "cache-02", "ci-node-4"]
PROJECTS = ["checkout-api", "vision-train", "etl-pipeline", "firmware", "webapp",
            "data-sync", "edge-agent", "invoice-bot", "sim-core", "ml-service",
            "auth-gateway", "billing", "search-index", "media-worker", "obs-stack",
            "notify-svc", "ledger", "parser", "scheduler", "mobile-api",
            "infra-live", "event-router", "report-builder", "risk-engine"]
PACKAGES = ["numpy", "pandas", "torch", "tensorflow", "requests", "flask",
            "pydantic", "scipy", "opencv-python", "transformers", "pytest",
            "cuDNN", "openssl", "libcurl4", "python3-dev", "build-essential",
            "docker-ce", "nginx", "redis-server", "ffmpeg", "scikit-learn",
            "matplotlib", "httpx", "sqlalchemy", "alembic", "ruff", "mypy",
            "gunicorn", "uvicorn", "celery", "kafka-python", "psycopg2-binary",
            "libpq-dev", "zlib1g-dev", "cmake", "ninja-build", "clang",
            "nodejs", "typescript", "eslint", "prettier", "rustc", "cargo"]
VERSIONS = ["1.24.4", "2.15.1", "3.11.9", "2.2.2", "0.8.4", "4.9.2", "2.3.1",
            "1.5.2", "0.19.2", "2025.1.0", "2.4.2", "1.13.2", "3.1.4", "8.7.0",
            "6.2.1", "9.2.0", "17.1.2", "20.12.1", "3.13.0", "1.0.92",
            "2.1.10", "0.2.171", "1.39.1", "0.8.1", "5.7.0"]
MODULES = ["yaml", "cv2", "torch", "tensorflow", "numpy", "sklearn", "PIL",
           "requests", "pydantic_core", "psycopg2", "mysqlclient", "flash_attn"]
FILES = ["main.py", "train.py", "config.yaml", "requirements.txt", "Dockerfile",
         "dataset.csv", "model.h5", "run.log", "setup.cfg", "Makefile",
         "worker.py", "schema.sql", "migration.py", "compose.yml", "tsconfig.json",
         "Cargo.toml", "go.mod", "package.json", "pyproject.toml", "values.yaml",
         "deployment.yaml", "nginx.conf", "access.log", "error.log", "backup.tar.gz"]
DIRS = ["/var/lib/docker", "/usr/local/lib/python3.10", "/etc/nginx", "~/.cache/pip",
        "/mnt/data", "/opt/app", "C:\\Program Files\\App", "C:\\Windows\\System32",
        "D:\\builds\\release", "C:\\Users\\dev\\AppData\\Local\\Temp"]
SERVICES = ["nginx", "docker", "ssh", "postgresql", "redis", "cron", "systemd-resolved",
            "NetworkManager", "containerd", "prometheus", "grafana-server", "kubelet",
            "certbot", "fail2ban", "postfix", "chronyd", "auditd", "logrotate"]
IPS = ["10.0.0.14", "192.168.1.22", "172.17.0.2", "8.8.8.8", "203.0.113.7",
       "10.20.30.40", "127.0.0.1", "192.168.100.5"]
DOMAINS = ["pypi.org", "files.pythonhosted.org", "github.com", "registry-1.docker.io",
           "deb.debian.org", "archive.ubuntu.com", "internal.corp.local", "example.com"]
PORTS = [22, 80, 443, 5432, 6379, 8080, 8000, 5000, 3306]
PYFILES = ["tests/test_api.py", "tests/test_model.py", "src/core.py", "app/server.py",
           "scripts/train.py", "utils/io.py", "tests/test_parse.py",
           "src/worker.py", "src/net.py", "src/db.py", "tests/test_worker.py",
           "tests/integration/test_deploy.py", "jobs/nightly.py", "lib/config.py"]
BRANCHES = ["main", "develop", "feature/auth", "release/2.4", "hotfix/login"]
FUNCS = ["load_config", "parse_args", "forward", "fit", "connect", "read_csv",
         "decode_frame", "validate", "save_checkpoint", "run_epoch",
         "build_index", "flush_queue", "open_stream", "retry_request",
         "migrate_schema", "collect_metrics", "render_page", "hash_password",
         "resolve_dns", "parse_payload", "write_batch", "healthcheck"]
COLORS = ["31", "32", "33", "34", "35", "36", "91", "92", "93", "94"]


def _p(rng: random.Random, items: Sequence) -> object:
    return items[rng.randrange(len(items))]


def _maybe(rng: random.Random, prob: float) -> bool:
    return rng.random() < prob


def _num(rng: random.Random, lo: int, hi: int) -> int:
    return rng.randint(lo, hi)


def _ansi(rng: random.Random, text: str, rate: float, label: str = "neutral") -> Line:
    if _maybe(rng, rate):
        color = _p(rng, COLORS)
        return (f"\x1b[{color}m{text}\x1b[0m", label)
    return (text, label)


def _stamp(rng: random.Random, rate: float, platform: str) -> str:
    if not _maybe(rng, rate):
        return ""
    if platform == "windows":
        return (
            f"{_num(rng, 2024, 2026)}-{_num(rng, 1, 12):02d}-{_num(rng, 1, 28):02d} "
            f"{_num(rng, 0, 23):02d}:{_num(rng, 0, 59):02d}:{_num(rng, 0, 59):02d} "
        )
    return (
        f"{_p(rng, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Sep', 'Oct', 'Nov'])} "
        f"{_num(rng, 1, 28):2d} {_num(rng, 0, 23):02d}:{_num(rng, 0, 59):02d}:"
        f"{_num(rng, 0, 59):02d} "
    )


def _host(rng: random.Random) -> str:
    return f"{_p(rng, USERS)}@{_p(rng, HOSTS)}"


def _prompt(rng: random.Random, platform: str, cwd: str, cmd: str) -> str:
    if platform == "windows":
        if _maybe(rng, 0.35):
            return f"PS {cwd}> {cmd}"
        return f"{cwd}> {cmd}"
    return f"{_host(rng)}:{cwd}$ {cmd}"


def _footer(rng: random.Random, platform: str, exit_code: int) -> Line:
    # Status is inferred from command output; no terminal exit-code footer.
    return ("", "neutral")


def _fillers(rng: random.Random, n: int, pool: Sequence[str]) -> List[Line]:
    return [(_p(rng, pool), "neutral") for _ in range(max(0, n))]


def _join(*groups: Sequence[Line]) -> List[Line]:
    out: List[Line] = []
    for g in groups:
        if g is None:
            continue
        if isinstance(g, str):
            out.append((g, "neutral"))
            continue
        if isinstance(g, tuple) and len(g) == 2 and isinstance(g[0], str):
            out.append((g[0], str(g[1])))
            continue
        for item in g:
            if isinstance(item, str):
                out.append((item, "neutral"))
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                out.append((str(item[0]), str(item[1])))
            else:
                raise TypeError(f"bad log line: {item!r}")
    return out


def _maybe_blank(rng: random.Random, rate: float = 0.25) -> List[Line]:
    return [("", "neutral")] if _maybe(rng, rate) else []


def _warn(rng: random.Random, text: str) -> Line:
    return (text, "neutral")


# --------------------------------------------------------------------------
# domain generators.  Each returns (lines, exit_code).
# --------------------------------------------------------------------------
def _gen_python(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    cmd = f"python {_p(rng, ['train.py', 'app.py', 'scripts/run.py', 'main.py'])}"
    head = [_prompt(rng, platform, "~/proj" if platform == "linux" else "C:\\proj", cmd)]
    if status == "success":
        lines = _fillers(rng, _num(rng, 2, 6), [
            f"Loading config from {_p(rng, FILES)}",
            f"device=cuda:0 batch_size={_num(rng, 8, 64)}",
            f"epoch {_num(rng, 1, 20)}/{_num(rng, 20, 50)} loss={rng.uniform(0.1, 2.8):.4f}",
            f"processed {_num(rng, 100, 9000)} samples",
            f"checkpoint saved to checkpoints/model_{_num(rng, 1, 99)}.h5",
        ])
        end = [
            ("Training finished successfully", "success"),
            _footer(rng, platform, 0),
        ]
        return _join(head, lines, end), 0
    if status == "fail":
        err = _p(rng, [
            "ModuleNotFoundError",
            "ValueError",
            "RuntimeError",
            "KeyError",
            "FileNotFoundError",
            "IndexError",
        ])
        msg = {
            "ModuleNotFoundError": f"No module named '{_p(rng, MODULES)}'",
            "ValueError": f"not enough values to unpack (expected {_num(rng, 2, 4)}, got 1)",
            "RuntimeError": f"CUDA out of memory. Tried to allocate {_num(rng, 1, 8)}.00 GiB",
            "KeyError": f"'{_p(rng, ['model', 'data', 'lr', 'checkpoint'])}'",
            "FileNotFoundError": f"[Errno 2] No such file or directory: '{_p(rng, FILES)}'",
            "IndexError": "list index out of range",
        }[err]
        body = _fillers(rng, _num(rng, 0, 3), [
            f"Loading config from {_p(rng, FILES)}",
            f"epoch 0 loss={rng.uniform(1.0, 3.0):.4f}",
        ])
        tb = [
            (f"Traceback (most recent call last):", "fail"),
            (f'  File "{_p(rng, PYFILES)}", line {_num(rng, 10, 400)}, in <module>', "fail"),
            (f"    {_p(rng, FUNCS)}({_p(rng, ['cfg', 'args', 'data', 'x'])})", "fail"),
            (f"  File \"{_p(rng, PYFILES)}\", line {_num(rng, 10, 400)}, in {_p(rng, FUNCS)}", "fail"),
            (f"{err}: {msg}", "fail"),
        ]
        return _join(head, body, tb, [_footer(rng, platform, 1)]), 1
    if status == "block":
        blk = _p(rng, [
            f"PermissionError: [Errno 13] Permission denied: '{_p(rng, DIRS)}/{_p(rng, FILES)}'",
            "OSError: [Errno 13] Permission denied: '/dev/nvidia0'",
            "PermissionError: [WinError 5] Access is denied",
            f"BlockingIOError: [Errno 11] Resource temporarily unavailable: '{_p(rng, FILES)}'",
        ])
        body = [
            (f"Traceback (most recent call last):", "fail"),
            (f'  File "{_p(rng, PYFILES)}", line {_num(rng, 10, 300)}, in <module>', "fail"),
            (blk, "block"),
            ("The operation was blocked by the security policy.", "block"),
        ]
        return _join(head, body, [_footer(rng, platform, 126)]), 126
    body = _fillers(rng, _num(rng, 1, 4), [
        f"Loading config from {_p(rng, FILES)}",
        f"device=cpu threads={_num(rng, 1, 16)}",
        f"processed {_num(rng, 10, 900)} samples",
    ])
    return _join(head, body, [("", "neutral"), ("  0%|          | 0/100 [00:00<?, ?it/s]", "neutral")]), None


def _gen_pip(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    pkg = _p(rng, PACKAGES)
    ver = _p(rng, VERSIONS)
    python = "python" if _maybe(rng, 0.5) else "python3"
    cmd = f"{python} -m pip install {pkg}=={ver}" if platform == "linux" else f"{python}.exe -m pip install {pkg}=={ver}"
    head = [_prompt(rng, platform, "~/proj" if platform == "linux" else "C:\\proj", cmd)]
    if status == "success":
        body = [
            (f"Collecting {pkg}=={ver}", "neutral"),
            (f"  Downloading {pkg}-{ver}-cp310-cp310-manylinux_2_17_x86_64.whl "
             f"({rng.uniform(0.4, 80):.1f} MB)", "neutral"),
            (f"     {'━' * 18} {rng.uniform(0.4, 80):.1f}/{rng.uniform(0.4, 80):.1f} MB "
             f"{rng.uniform(2, 60):.1f} MB/s eta 0:00:00", "neutral"),
            (f"Requirement already satisfied: {_p(rng, PACKAGES)} in "
             f"/usr/lib/python3.10/site-packages", "neutral"),
            (f"Installing collected packages: {pkg}", "neutral"),
            (f"Successfully installed {pkg}-{ver}", "success"),
            _footer(rng, platform, 0),
        ]
        return _join(head, _maybe_blank(rng), body), 0
    if status == "fail":
        errs = _p(rng, [
            [(f"ERROR: Could not find a version that satisfies the requirement {pkg}=={ver} "
              f"(from versions: none)", "fail")],
            [(f"ERROR: No matching distribution found for {pkg}=={ver}", "fail")],
            [(f"WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, "
              f"status=None)) after connection broken by 'ReadTimeoutError'", "neutral")],
            [(f"ERROR: Could not install packages due to an OSError: [Errno 28] "
              f"No space left on device", "fail")],
            [(f"error: subprocess-exited-with-error", "fail")],
            [(f"  ERROR: Failed building wheel for {pkg}", "fail")],
        ])
        body = _fillers(rng, _num(rng, 1, 4), [
            f"Collecting {pkg}=={ver}",
            f"  Using cached {pkg}-{ver}-py3-none-any.whl ({rng.uniform(0.1, 9):.1f} MB)",
            "  Preparing metadata (setup.py) ... done",
        ])
        return _join(head, body, errs, [_footer(rng, platform, 1)]), 1
    if status == "block":
        body = [
            (f"Collecting {pkg}=={ver}", "neutral"),
            (f"ERROR: Could not install packages due to an OSError: [Errno 13] "
             f"Permission denied: '/usr/lib/python3.10/site-packages/{pkg}'", "block"),
            ("Consider using the `--user` option or check the permissions.", "block"),
            ("Defaulting to user installation because normal site-packages is not writeable",
             "neutral"),
            ("error: externally-managed-environment", "block"),
            ("× This environment is externally managed", "block"),
            ("╰─> To install Python packages system-wide, try apt install", "block"),
            ("    python3-xyz, where xyz is the package you are trying to", "block"),
            ("    install.", "block"),
            ("note: If you believe this is a mistake, please contact your Python", "block"),
            ("installation or OS distribution provider.", "block"),
            ("hint: See PEP 668 for the detailed specification.", "block"),
            _footer(rng, platform, 126),
        ]
        return _join(head, body), 126
    body = _fillers(rng, _num(rng, 1, 3), [
        f"Collecting {pkg}=={ver}",
        f"  Downloading {pkg}-{ver}-py3-none-any.whl ({rng.uniform(0.1, 9):.1f} MB)",
        "  Preparing metadata (setup.py) ...",
    ])
    return _join(head, body, [("", "neutral"), ("  Building wheel for " + pkg + " ...", "neutral")]), None


def _gen_pytest(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    cmd = "pytest -q" if _maybe(rng, 0.5) else "python -m pytest tests/"
    head = [_prompt(rng, platform, "~/proj" if platform == "linux" else "C:\\proj", cmd)]
    if status == "success":
        passed = _num(rng, 5, 240)
        body = _fillers(rng, _num(rng, 2, 8), [
            f"tests/test_{_p(rng, ['api', 'model', 'parse', 'io'])}.py ....",
            f"collected {passed} items",
            f"tests/test_api.py::test_{_p(rng, FUNCS)} PASSED",
            f"tests/test_model.py::test_forward PASSED",
        ])
        end = [
            (f"===================== {passed} passed in {rng.uniform(0.4, 42):.2f}s "
             f"=====================", "success"),
            _footer(rng, platform, 0),
        ]
        return _join(head, body, end), 0
    if status == "fail":
        failed = _num(rng, 1, 6)
        passed = _num(rng, 3, 90)
        body = [
            (f"tests/test_{_p(rng, ['api', 'model', 'parse'])}.py ..F.", "neutral"),
            (f"=================================== FAILURES "
             f"===================================", "fail"),
            (f"_________________________ test_{_p(rng, FUNCS)} _________________________", "fail"),
            (f"    assert result == expected", "fail"),
            (f"E   AssertionError: assert {_num(rng, 0, 9)} == {_num(rng, 0, 9)}", "fail"),
            (f"FAILED tests/test_api.py::test_{_p(rng, FUNCS)} - AssertionError", "fail"),
            (f"=================== {failed} failed, {passed} passed in "
             f"{rng.uniform(0.4, 42):.2f}s ===================", "fail"),
            _footer(rng, platform, 1),
        ]
        return _join(head, body), 1
    if status == "block":
        body = [
            ("tests/test_io.py E", "block"),
            ("E   PermissionError: [Errno 13] Permission denied: '/mnt/data/input.csv'", "block"),
            ("ERROR: could not open test fixture, access denied by policy", "block"),
            _footer(rng, platform, 1),
        ]
        return _join(head, body), 1
    body = _fillers(rng, _num(rng, 1, 5), [
        "collected 128 items",
        "tests/test_api.py .....",
        "tests/test_model.py ...",
        "tests/test_parse.py ..",
    ])
    return _join(head, body, [("tests/test_io.py ...", "neutral")]), None


def _gen_conda(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    env = _p(rng, ["ml", "app", "research", "vision"])
    pkg = _p(rng, PACKAGES)
    head = [_prompt(rng, platform, "~", f"conda install -n {env} {pkg}")]
    if status == "success":
        body = [
            (f"Collecting package metadata (current_repodata.json): done", "neutral"),
            (f"Solving environment: done", "neutral"),
            ("", "neutral"),
            ("## Package Plan ##", "neutral"),
            (f"  environment location: /home/{_p(rng, USERS)}/miniconda3/envs/{env}", "neutral"),
            (f"  added / updated specs:", "neutral"),
            (f"    - {pkg}", "neutral"),
            (f"Preparing transaction: done", "neutral"),
            (f"Verifying transaction: done", "neutral"),
            (f"Executing transaction: done", "success"),
            _footer(rng, platform, 0),
        ]
        return _join(head, body), 0
    if status == "fail":
        errs = _p(rng, [
            [("PackagesNotFoundError: The following packages are not available from current channels:", "fail"),
             (f"  - {pkg}", "fail")],
            [("CondaHTTPError: HTTP 000 CONNECTION FAILED for url "
              f"<https://conda.anaconda.org/conda-forge/linux-64/repodata.json>", "fail"),
             ("Elapsed: -", "fail")],
            [("LibMambaUnsatisfiableError: Encountered problems while solving:", "fail"),
             (f"  - package {pkg}-{_p(rng, VERSIONS)} requires python >=3.11, but none of the providers can be installed", "fail")],
        ])
        return _join(head, _maybe_blank(rng), errs, [_footer(rng, platform, 1)]), 1
    if status == "block":
        body = [
            ("Collecting package metadata (repodata.json): done", "neutral"),
            (f"EnvironmentNotWritableError: The current user does not have write permissions "
             f"to the target environment.", "block"),
            (f"  environment location: /opt/conda/envs/{env}", "block"),
            ("  uid: 1000  gid: 1000", "neutral"),
            _footer(rng, platform, 1),
        ]
        return _join(head, body), 1
    body = _fillers(rng, _num(rng, 1, 4), [
        "Collecting package metadata (repodata.json): done",
        "Solving environment: ...",
        "## Package Plan ##",
    ])
    return _join(head, body), None


def _gen_package(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    pkg = _p(rng, PACKAGES)
    manager = _p(rng, ["apt-get", "apt", "dnf", "pacman"])
    cmd = {
        "apt-get": f"sudo apt-get install -y {pkg}",
        "apt": f"sudo apt install {pkg}",
        "dnf": f"sudo dnf install -y {pkg}",
        "pacman": f"sudo pacman -S {pkg}",
    }[manager]
    head = [_prompt(rng, "linux", "~", cmd)]
    if status == "success":
        body = [
            f"Reading package lists... Done",
            f"Building dependency tree... Done",
            f"Reading state information... Done",
            f"The following NEW packages will be installed:",
            f"  {pkg}",
            f"0 upgraded, 1 newly installed, 0 to remove and 3 not upgraded.",
            f"Get:1 http://deb.debian.org/debian bookworm/main amd64 {pkg} "
            f"{_p(rng, VERSIONS)} [{rng.uniform(0.1, 12):.1f} MB]",
            f"Fetched {rng.uniform(0.1, 12):.1f} MB in {_num(rng, 1, 20)}s "
            f"({rng.uniform(0.1, 9):.2f} MB/s)",
            f"Setting up {pkg} ({_p(rng, VERSIONS)}) ...",
            f"Processing triggers for man-db ({_p(rng, VERSIONS)}) ...",
        ]
        return _join(head, [(_ansi(rng, x, cfg.ansi_rate), "neutral") for x in body],
                     [("done.", "success"), _footer(rng, "linux", 0)]), 0
    if status == "fail":
        errs = _p(rng, [
            [(f"E: Unable to locate package {pkg}", "fail"),
             (f"E: Couldn't find any package by glob '{pkg}'", "fail")],
            [(f"Err:1 http://deb.debian.org/debian bookworm/main amd64 {pkg} "
              f"{_p(rng, VERSIONS)}", "fail"),
             ("  404  Not Found [IP: 151.101.2.132 80]", "fail"),
             (f"E: Failed to fetch http://deb.debian.org/debian/{pkg}.deb  404  Not Found", "fail"),
             ("E: Unable to fetch some archives, maybe run apt-get update or try with --fix-missing?", "fail")],
            [(f"dpkg: error processing package {pkg} (--configure):", "fail"),
             (" installed " + pkg + " package post-installation script subprocess returned error exit status 1", "fail")],
            [("error: target not found: " + pkg, "fail")],
        ])
        body = _fillers(rng, _num(rng, 2, 6), [
            "Reading package lists... Done",
            "Building dependency tree... Done",
            "Reading state information... Done",
            f"The following NEW packages will be installed: {pkg}",
            "0 upgraded, 1 newly installed, 0 to remove and 3 not upgraded.",
        ])
        return _join(head, body, errs, [_footer(rng, "linux", 100)]), 100
    if status == "block":
        body = [
            ("Reading package lists... Done", "neutral"),
            ("E: Could not open lock file /var/lib/dpkg/lock-frontend - open (13: Permission denied)",
             "block"),
            ("E: Unable to acquire the dpkg frontend lock (/var/lib/dpkg/lock-frontend), "
             "are you root?", "block"),
            ("sudo: a password is required", "block"),
            _footer(rng, "linux", 1),
        ]
        return _join(head, body), 1
    body = _fillers(rng, _num(rng, 1, 4), [
        "Reading package lists... Done",
        "Building dependency tree... Done",
        f"The following NEW packages will be installed: {pkg}",
        "0 upgraded, 1 newly installed, 0 to remove and 3 not upgraded.",
        f"Get:1 http://deb.debian.org/debian bookworm/main amd64 {pkg} {_p(rng, VERSIONS)}",
    ])
    return _join(head, body, [("Fetched 4.2 MB in 2s (2.1 MB/s)", "neutral")]), None


def _gen_systemd(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    svc = _p(rng, SERVICES)
    cmd = f"systemctl start {svc}"
    head = [_prompt(rng, "linux", "~", cmd)]
    if status == "success":
        body = [
            (f"● {svc}.service - {svc.capitalize()} service", "neutral"),
            (f"     Loaded: loaded (/lib/systemd/system/{svc}.service; enabled; "
             f"vendor preset: enabled)", "neutral"),
            ("     Active: active (running) since "
             f"{_p(rng, ['Mon', 'Tue', 'Wed', 'Fri'])} {_num(rng, 1, 28)} "
             f"{_num(rng, 0, 23):02d}:{_num(rng, 0, 59):02d}:{_num(rng, 0, 59):02d} UTC; "
             f"{_num(rng, 1, 40)}s ago", "neutral"),
            (f"   Main PID: {_num(rng, 400, 9000)} ({svc})", "neutral"),
            (f"      Tasks: {_num(rng, 1, 40)} (limit: 4915)", "neutral"),
            (f"     Memory: {rng.uniform(1, 900):.1f}M", "neutral"),
            (f"        CPU: {rng.uniform(0.1, 40):.2f}s", "neutral"),
            (f"Started {svc}.service.", "success"),
            _footer(rng, "linux", 0),
        ]
        return _join(head, body), 0
    if status == "fail":
        body = [
            (f"Job for {svc}.service failed because the control process exited with "
             f"error code.", "fail"),
            (f"See \"systemctl status {svc}.service\" and \"journalctl -xeu {svc}.service\" "
             f"for details.", "fail"),
            (f"● {svc}.service - {svc.capitalize()} service", "neutral"),
            ("     Active: failed (Result: exit-code) since "
             f"{_num(rng, 1, 28)} {_num(rng, 0, 23):02d}:{_num(rng, 0, 59):02d}:"
             f"{_num(rng, 0, 59):02d} UTC; {_num(rng, 1, 30)}s ago", "fail"),
            (f"    Process: {_num(rng, 400, 9000)} ExecStart=/usr/bin/{svc} "
             f"(code=exited, status=1/FAILURE)", "fail"),
            (f"   Main PID: {_num(rng, 400, 9000)} (code=exited, status=1/FAILURE)", "fail"),
            _footer(rng, "linux", 1),
        ]
        return _join(head, body), 1
    if status == "block":
        body = [
            (f"Failed to start {svc}.service: Access denied", "block"),
            ("Interactive authentication required.", "block"),
            (f"See \"systemctl status {svc}.service\" for details.", "neutral"),
            ("polkit: Operation not permitted", "block"),
            _footer(rng, "linux", 1),
        ]
        return _join(head, body), 1
    body = [
        (f"● {svc}.service - {svc.capitalize()} service", "neutral"),
        (f"     Loaded: loaded (/lib/systemd/system/{svc}.service; enabled)", "neutral"),
        (f"     Active: inactive (dead) since {_num(rng, 1, 28)} "
         f"{_num(rng, 0, 23):02d}:{_num(rng, 0, 59):02d}:{_num(rng, 0, 59):02d}", "neutral"),
        (f"    Process: {_num(rng, 400, 9000)} ExecStart=/usr/bin/{svc} "
         f"(code=exited, status=0/SUCCESS)", "neutral"),
    ]
    return _join(head, body), None


def _gen_network(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    host = _p(rng, DOMAINS)
    ip = _p(rng, IPS)
    if platform == "windows":
        cmd = _p(rng, [f"ping {host}", f"Test-NetConnection {host} -Port 443",
                       f"Invoke-WebRequest https://{host}"])
    else:
        cmd = _p(rng, [f"ping -c 4 {host}", f"curl -Iv https://{host}",
                       f"ssh {_p(rng, USERS)}@{ip}"])
    head = [_prompt(rng, platform, "~" if platform == "linux" else "C:\\Users\\dev", cmd)]
    if status == "success":
        if cmd.startswith("ping"):
            body = [
                f"PING {host} ({ip}) 56(84) bytes of data.",
                f"64 bytes from {ip}: icmp_seq=1 ttl=57 time={rng.uniform(1, 60):.1f} ms",
                f"64 bytes from {ip}: icmp_seq=2 ttl=57 time={rng.uniform(1, 60):.1f} ms",
                f"64 bytes from {ip}: icmp_seq=3 ttl=57 time={rng.uniform(1, 60):.1f} ms",
                "",
                f"--- {host} ping statistics ---",
                f"4 packets transmitted, 4 received, 0% packet loss, time {_num(rng, 3000, 5000)}ms",
                f"rtt min/avg/max/mdev = {rng.uniform(1, 20):.3f}/{rng.uniform(2, 30):.3f}/"
                f"{rng.uniform(30, 60):.3f}/{rng.uniform(0.1, 2):.3f} ms",
            ]
        elif cmd.startswith("curl"):
            body = [
                f"*   Trying {ip}:443...",
                f"* Connected to {host} ({ip}) port 443",
                f"* TLS 1.3 connection using TLS_AES_256_GCM_SHA384",
                f"> GET / HTTP/2",
                f"< HTTP/2 200",
                f"< content-type: text/html; charset=utf-8",
                f"< content-length: {_num(rng, 100, 90000)}",
                f"< server: nginx",
            ]
        else:
            body = [
                f"The authenticity of host '{ip} ({ip})' can't be established.",
                f"ED25519 key fingerprint is SHA256:{_p(rng, ['aBcD', 'xYz1', 'QwEr'])}...",
                f"Connection established.",
                f"Welcome to Ubuntu 22.04.4 LTS (GNU/Linux 5.15.0-105-generic x86_64)",
                f"Last login: {_p(rng, ['Mon', 'Tue', 'Thu'])} {_num(rng, 1, 28)} "
                f"{_num(rng, 0, 23):02d}:{_num(rng, 0, 59):02d} from {_p(rng, IPS)}",
            ]
        return _join(head, [(x, "neutral") for x in body],
                     [("Connection successful.", "success"), _footer(rng, platform, 0)]), 0
    if status == "fail":
        errs = _p(rng, [
            [(f"ping: {host}: Name or service not known", "fail"),
             ("4 packets transmitted, 0 received, 100% packet loss, time 3004ms", "fail")],
            [(f"curl: (7) Failed to connect to {host} port 443 after "
              f"{_num(rng, 100, 30000)} ms: Connection refused", "fail")],
            [(f"ssh: connect to host {ip} port 22: Connection refused", "fail")],
            [(f"curl: (6) Could not resolve host: {host}", "fail")],
            [("Test-NetConnection : TCP connect to " + host + ":443 failed", "fail"),
             ("WARNING: Ping to " + host + " failed with status: TimedOut", "fail")],
            [("Invoke-WebRequest : The remote name could not be resolved: '" + host + "'", "fail")],
        ])
        return _join(head, errs, [_footer(rng, platform, 1)]), 1
    if status == "block":
        body = _p(rng, [
            [(f"curl: (22) The requested URL returned error: 403", "block"),
             ("Access blocked by network policy (proxy CONNECT denied)", "block")],
            [("ssh: connect to host " + ip + " port 22: Permission denied", "block"),
             ("Connection blocked by firewall.", "block")],
            [(f"Invoke-WebRequest : The remote server returned an error: (403) Forbidden.", "block"),
             ("Request blocked by conditional access policy.", "block")],
            [("PING " + host + " (" + ip + "): 56 data bytes", "neutral"),
             ("Request timeout for icmp_seq 0", "neutral"),
             ("ping: sendmsg: Operation not permitted", "block")],
        ])
        return _join(head, body, [_footer(rng, platform, 126)]), 126
    body = _fillers(rng, _num(rng, 1, 4), [
        f"*   Trying {ip}:443...",
        f"PING {host} ({ip}) 56(84) bytes of data.",
        f"64 bytes from {ip}: icmp_seq=1 ttl=57 time={rng.uniform(1, 40):.1f} ms",
        f"64 bytes from {ip}: icmp_seq=2 ttl=57 time={rng.uniform(1, 40):.1f} ms",
    ])
    return _join(head, body), None


def _gen_filesystem(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    d = _p(rng, DIRS)
    f = _p(rng, FILES)
    if platform == "windows":
        cmd = _p(rng, [f"copy {f} D:\\backup\\", f"robocopy D:\\builds C:\\backup /MIR",
                       f"icacls {d}", f"del {d}\\{f}"])
    else:
        cmd = _p(rng, [f"cp {f} {d}/", f"rsync -av {d}/ backup:{d}/",
                       f"chmod 600 {f}", f"tar -czf backup.tar.gz {d}"])
    head = [_prompt(rng, platform, "~/proj" if platform == "linux" else "C:\\proj", cmd)]
    if status == "success":
        body = _fillers(rng, _num(rng, 2, 12), [
            f"'{f}' -> '{d}/{f}'",
            f"sending incremental file list",
            f"{f}",
            f"{_num(rng, 100, 9000)} bytes/sec",
            f"total size is {_num(rng, 100, 900000)}  speedup is 1.00",
            f"sent {_num(rng, 1, 900)} bytes  received {_num(rng, 10, 9000)} bytes  "
            f"{rng.uniform(0.1, 9):.2f} bytes/sec",
        ])
        return _join(head, body, [("1 file copied.", "success"), _footer(rng, platform, 0)]), 0
    if status == "fail":
        errs = _p(rng, [
            [(f"cp: cannot stat '{f}': No such file or directory", "fail")],
            [(f"cp: error writing '{d}/{f}': No space left on device", "fail")],
            [(f"rsync: [sender] change_dir \"{d}\" failed: No such file or directory (2)", "fail"),
             (f"rsync error: some files/attrs were not transferred (see previous errors) (code 23)", "fail")],
            [(f"The system cannot find the file specified.", "fail"),
             ("ERROR 2 (0x00000002) accessing " + d, "fail")],
            [(f"robocopy exited with code 8 (some files could not be copied)", "fail")],
        ])
        return _join(head, errs, [_footer(rng, platform, 1)]), 1
    if status == "block":
        body = _p(rng, [
            [(f"chmod: changing permissions of '{f}': Operation not permitted", "block"),
             ("Permission denied", "block")],
            [(f"cp: cannot create regular file '{d}/{f}': Permission denied", "block")],
            [(f"Access is denied.", "block"),
             ("0 files copied.", "block")],
            [(f"icacls: Access to the path '{d}' is denied.", "block")],
            [(f"rsync: [receiver] mkdir \"{d}\" failed: Permission denied (13)", "block")],
        ])
        return _join(head, body, [_footer(rng, platform, 126)]), 126
    body = _fillers(rng, _num(rng, 1, 5), [
        f"'{f}' -> '{d}/{f}'",
        f"sending incremental file list",
        f"{f}",
        f"{_num(rng, 100, 9000)} bytes/sec",
    ])
    return _join(head, body), None


def _gen_git(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    repo = _p(rng, PROJECTS)
    branch = _p(rng, BRANCHES)
    cmd = _p(rng, [f"git clone git@github.com:org/{repo}.git",
                   f"git pull origin {branch}", f"git push origin {branch}",
                   "git status", "git commit -am 'fix parser'"])
    head = [_prompt(rng, platform, "~/proj" if platform == "linux" else "C:\\proj", cmd)]
    if status == "success":
        if "clone" in cmd:
            body = [
                f"Cloning into '{repo}'...",
                f"remote: Enumerating objects: {_num(rng, 100, 9000)}, done.",
                f"remote: Counting objects: 100% ({_num(rng, 100, 9000)}/{_num(rng, 100, 9000)}), done.",
                f"remote: Compressing objects: 100% ({_num(rng, 20, 900)}/{_num(rng, 20, 900)}), done.",
                f"Receiving objects: 100% ({_num(rng, 100, 9000)}/{_num(rng, 100, 9000)}), "
                f"{rng.uniform(0.1, 30):.2f} MiB | {rng.uniform(0.1, 9):.2f} MiB/s, done.",
                f"Resolving deltas: 100% ({_num(rng, 10, 900)}/{_num(rng, 10, 900)}), done.",
            ]
        elif "push" in cmd:
            body = [
                f"Enumerating objects: {_num(rng, 1, 40)}, done.",
                f"Counting objects: 100% ({_num(rng, 1, 40)}/{_num(rng, 1, 40)}), done.",
                f"Delta compression using up to {_num(rng, 2, 16)} threads",
                f"Compressing objects: 100% ({_num(rng, 1, 40)}/{_num(rng, 1, 40)}), done.",
                f"Writing objects: 100% ({_num(rng, 1, 40)}/{_num(rng, 1, 40)}), "
                f"{rng.uniform(0.1, 9):.2f} KiB | {rng.uniform(0.1, 9):.2f} MiB/s, done.",
                f"To github.com:org/{repo}.git",
                f"   {branch} -> {branch}",
            ]
        else:
            body = [
                f"On branch {branch}",
                "Your branch is up to date with 'origin/" + branch + "'.",
                "nothing to commit, working tree clean" if _maybe(rng, 0.5)
                else "Changes not staged for commit:",
            ]
        return _join(head, [(x, "neutral") for x in body],
                     [("Everything up-to-date.", "success"), _footer(rng, platform, 0)]), 0
    if status == "fail":
        errs = _p(rng, [
            [(f"fatal: repository 'https://github.com/org/{repo}.git/' not found", "fail")],
            [(f"fatal: couldn't find remote ref {branch}", "fail")],
            [("error: Your local changes to the following files would be overwritten by merge:", "fail"),
             (f"  {_p(rng, FILES)}", "fail"),
             ("Please commit your changes or stash them before you merge.", "fail")],
            [("CONFLICT (content): Merge conflict in " + _p(rng, FILES), "fail"),
             ("Automatic merge failed; fix conflicts and then commit the result.", "fail")],
            [("! [rejected]        " + branch + " -> " + branch + " (non-fast-forward)", "fail"),
             ("error: failed to push some refs to 'github.com:org/" + repo + ".git'", "fail")],
            [(f"ERROR: Repository not found.", "fail")],
        ])
        return _join(head, errs, [_footer(rng, platform, 1)]), 1
    if status == "block":
        body = _p(rng, [
            [("git@github.com: Permission denied (publickey).", "block"),
             (f"fatal: Could not read from remote repository.", "block")],
            [("remote: HTTP Basic: Access denied", "block"),
             ("fatal: Authentication failed for 'https://github.com/org/" + repo + ".git/'", "block")],
            [("fatal: unable to access 'https://github.com/org/" + repo + ".git/': "
              "The requested URL returned error: 403", "block")],
            [("error: insufficient permission for adding an object to repository database .git/objects", "block"),
             ("fatal: failed to write object", "block")],
        ])
        return _join(head, body, [_footer(rng, platform, 126)]), 126
    body = _fillers(rng, _num(rng, 1, 4), [
        f"Cloning into '{repo}'...",
        f"remote: Enumerating objects: {_num(rng, 100, 9000)}, done.",
        f"Receiving objects: {_num(rng, 1, 99)}% ({_num(rng, 100, 900)}/{_num(rng, 900, 9000)})",
    ])
    return _join(head, body), None


def _gen_docker(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    img = _p(rng, ["nginx:latest", "python:3.10-slim", "postgres:16", "myapp:v2"])
    cmd = _p(rng, [f"docker pull {img}", "docker compose up -d", f"docker build -t myapp:v2 ."])
    head = [_prompt(rng, platform, "~/proj" if platform == "linux" else "C:\\proj", cmd)]
    if status == "success":
        body = [
            f"latest: Pulling from library/{img.split(':')[0]}",
            f"{_p(rng, ['a2abf6c4d29d', 'b7e6c9f2a1d4', 'e9d1c7a3b5f2'])}: Pull complete",
            f"{_p(rng, ['c3f4a1b2d6e7', 'f1e2d3c4b5a6'])}: Pull complete",
            f"Digest: sha256:{_p(rng, ['9f2c1a', 'ab34cd', '77e1aa'])}"
            f"{_num(rng, 100000, 999999):06d}",
            f"Status: Downloaded newer image for {img}",
            f"Creating network \"{_p(rng, PROJECTS)}_default\" with the default driver",
            f"Creating {_p(rng, PROJECTS)}_app_1 ... done",
        ]
        return _join(head, [(x, "neutral") for x in body],
                     [("Deployment completed successfully.", "success"),
                      _footer(rng, platform, 0)]), 0
    if status == "fail":
        errs = _p(rng, [
            [("Cannot connect to the Docker daemon at unix:///var/run/docker.sock. "
              "Is the docker daemon running?", "fail")],
            [("Error response from daemon: driver failed programming external connectivity "
              "on endpoint app: Bind for 0.0.0.0:8080 failed: port is already allocated", "fail")],
            [("Error response from daemon: no space left on device", "fail")],
            [("ERROR: Service 'app' failed to build: The command '/bin/sh -c pip install -r "
              "requirements.txt' returned a non-zero code: 1", "fail")],
            [("Error: No such container: myapp", "fail")],
        ])
        return _join(head, errs, [_footer(rng, platform, 1)]), 1
    if status == "block":
        body = [
            ("Error response from daemon: pull access denied for " + img + ", "
             "repository does not exist or may require 'docker login'", "block"),
            ("unauthorized: authentication required", "block"),
            ("Access denied by registry policy.", "block"),
            _footer(rng, platform, 126),
        ]
        return _join(head, body), 126
    body = _fillers(rng, _num(rng, 1, 5), [
        f"latest: Pulling from library/{img.split(':')[0]}",
        "a2abf6c4d29d: Downloading [==>   ]  4.2MB/27.1MB",
        "c3f4a1b2d6e7: Downloading [=>    ]  1.1MB/9.8MB",
        "Building 45.2s (12/18)",
    ])
    return _join(head, body), None


def _gen_kernel(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    cmd = _p(rng, ["dmesg -T", "dmesg | tail -n 40", "journalctl -k -n 60 --no-pager"])
    head = [_prompt(rng, "linux", "~", cmd)]
    if status == "success":
        body = [
            f"[{rng.uniform(0, 100):8.5f}] Linux version 6.8.0-45-generic "
            f"(buildd@lcy02-amd64-045) (gcc 12.3.0)",
            f"[{rng.uniform(0, 100):8.5f}] Command line: BOOT_IMAGE=/vmlinuz-6.8.0-45-generic "
            f"root=UUID={_p(rng, ['4a1b2c3d', 'e5f6a7b8'])}-... ro quiet splash",
            f"[{rng.uniform(0, 100):8.5f}] Memory: {_num(rng, 4, 64)}GiB available",
            f"[{rng.uniform(0, 100):8.5f}] EXT4-fs (nvme0n1p2): mounted filesystem with "
            f"ordered data mode",
            f"[{rng.uniform(0, 100):8.5f}] usb 1-2: new high-speed USB device number "
            f"{_num(rng, 1, 12)} using xhci_hcd",
            f"[{rng.uniform(0, 100):8.5f}] IPv6: ADDRCONF(NETDEV_CHANGE): enp5s0: link becomes ready",
        ]
        return _join(head, [(x, "neutral") for x in body],
                     [("dmesg: read kernel buffer successfully.", "success"),
                      _footer(rng, "linux", 0)]), 0
    if status == "fail":
        errs = _p(rng, [
            [(f"[{rng.uniform(0, 999):8.5f}] Out of memory: Killed process {_num(rng, 1000, 90000)} "
              f"(python3) total-vm:{_num(rng, 400000, 9000000)}kB, anon-rss:"
              f"{_num(rng, 100000, 9000000)}kB", "fail")],
            [(f"[{rng.uniform(0, 999):8.5f}] EXT4-fs error (device nvme0n1p2): "
              f"ext4_find_entry:1455: inode #{_num(rng, 100, 999999)}: comm \"updatedb\": "
              f"reading directory lblock 0", "fail")],
            [(f"[{rng.uniform(0, 999):8.5f}] nvme nvme0: I/O {_num(rng, 1, 99)} QID 3 timeout, "
              f"aborting", "fail"),
             (f"[{rng.uniform(0, 999):8.5f}] I/O error, dev nvme0n1, sector {_num(rng, 1000, 9000000)}", "fail")],
            [(f"[{rng.uniform(0, 999):8.5f}] python3[{_num(rng, 1000, 90000)}]: "
              f"segfault at 0 ip 00007f{_num(rng, 10000000, 99999999):08x} sp 00007ffd... "
              f"error 4 in libc.so.6", "fail"),
             (f"[{rng.uniform(0, 999):8.5f}] traps: python3[{_num(rng, 1000, 90000)}] "
              f"general protection fault ip:...", "fail")],
        ])
        return _join(head, _maybe_blank(rng), errs, [_footer(rng, "linux", 1)]), 1
    if status == "block":
        body = [
            (f"[{rng.uniform(0, 999):8.5f}] audit: type=1400 audit({rng.uniform(1, 9):.3f}:"
             f"{_num(rng, 10, 999)}): apparmor=\"DENIED\" operation=\"open\" "
             f"profile=\"/usr/bin/python3\" name=\"{_p(rng, DIRS)}/{_p(rng, FILES)}\" "
             f"pid={_num(rng, 100, 9000)} comm=\"python3\"", "block"),
            (f"[{rng.uniform(0, 999):8.5f}] audit: type=1326 audit({rng.uniform(1, 9):.3f}:"
             f"{_num(rng, 10, 999)}): auid=1000 uid=1000 gid=1000 ses=2 "
             f"subj=unconfined pid={_num(rng, 100, 9000)} comm=\"python3\" "
             f"exe=\"/usr/bin/python3\" sig=31 arch=c000003e syscall=59 "
             f"compat=0 ip=... code=0x0", "block"),
            ("SELinux: denied { execute } for pid=" + str(_num(rng, 100, 9000)) +
             " comm=\"python3\"", "block"),
            _footer(rng, "linux", 126),
        ]
        return _join(head, body), 126
    body = _fillers(rng, _num(rng, 3, 10), [
        f"[{rng.uniform(0, 999):8.5f}] usb 1-2: new high-speed USB device number "
        f"{_num(rng, 1, 12)} using xhci_hcd",
        f"[{rng.uniform(0, 999):8.5f}] IPv6: ADDRCONF(NETDEV_CHANGE): enp5s0: link becomes ready",
        f"[{rng.uniform(0, 999):8.5f}] EXT4-fs (nvme0n1p2): mounted filesystem",
        f"[{rng.uniform(0, 999):8.5f}] Bluetooth: hci0: RTL: examining hci_ver=0b",
    ])
    return _join(head, body), None


def _gen_journal(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    svc = _p(rng, SERVICES)
    cmd = _p(rng, [f"journalctl -u {svc} -n 50 --no-pager", "journalctl -xe --no-pager",
                   "journalctl -p err -b --no-pager"])
    head = [_prompt(rng, "linux", "~", cmd)]
    stamp = lambda: (f"{_p(rng, ['Nov', 'Oct', 'Sep'])} {_num(rng, 1, 28):02d} "
                     f"{_num(rng, 0, 23):02d}:{_num(rng, 0, 59):02d}:{_num(rng, 0, 59):02d} "
                     f"{_p(rng, HOSTS)} {_p(rng, [svc + '[1234]', 'kernel', 'systemd[1]'])}: ")
    if status == "success":
        body = [
            stamp() + f"Started {svc}.service - {svc.capitalize()}.",
            stamp() + f"{svc}[{_num(rng, 100, 9000)}]: configuration loaded from "
                      f"/etc/{svc}/{svc}.conf",
            stamp() + f"{svc}[{_num(rng, 100, 9000)}]: listening on port {_p(rng, PORTS)}",
            stamp() + f"{svc}[{_num(rng, 100, 9000)}]: worker {_num(rng, 1, 8)} started",
            stamp() + f"{svc}.service: Succeeded.",
        ]
        return _join(head, [(x, "neutral") for x in body[:4]],
                     [(body[4], "success"), _footer(rng, "linux", 0)]), 0
    if status == "fail":
        errs = [
            (stamp() + f"{svc}.service: Main process exited, code=exited, status=1/FAILURE", "fail"),
            (stamp() + f"{svc}.service: Failed with result 'exit-code'.", "fail"),
            (stamp() + f"{svc}[{_num(rng, 100, 9000)}]: fatal: cannot bind port "
             f"{_p(rng, PORTS)}: Address already in use", "fail"),
            (stamp() + f"{svc}.service: Scheduled restart job, restart counter is at "
             f"{_num(rng, 1, 9)}.", "fail"),
        ]
        return _join(head, _maybe_blank(rng), errs, [_footer(rng, "linux", 1)]), 1
    if status == "block":
        body = [
            (stamp() + f"audit: avc: denied {{ read }} for pid={_num(rng, 100, 9000)} "
             f"comm=\"{svc}\" path=\"{_p(rng, DIRS)}\"", "block"),
            (stamp() + f"{svc}.service: Failed to open configuration: Permission denied", "block"),
            (stamp() + "policykit: Operator not authorized to start unit", "block"),
            _footer(rng, "linux", 1),
        ]
        return _join(head, body), 1
    body = [(stamp() + f"{svc}[{_num(rng, 100, 9000)}]: request handled in "
             f"{rng.uniform(1, 900):.1f} ms", "neutral") for _ in range(_num(rng, 3, 8))]
    return _join(head, body), None


def _gen_build(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    tool = _p(rng, ["make", "cmake --build build", "cargo build --release", "go build ./...", "gcc -O2 main.c"])
    head = [_prompt(rng, platform, "~/proj" if platform == "linux" else "C:\\proj", tool)]
    if status == "success":
        body = _fillers(rng, _num(rng, 2, 8), [
            f"[ 25%] Building CXX object CMakeFiles/app.dir/src/main.cpp.o",
            f"[ 50%] Building CXX object CMakeFiles/app.dir/src/parser.cpp.o",
            f"[100%] Linking CXX executable app",
            f"   Compiling serde v1.0.{_num(rng, 10, 200)}",
            f"    Finished release [optimized] target(s) in {rng.uniform(1, 200):.2f}s",
            f"main.c: In function 'main':",
        ])
        return _join(head, body, [("Build succeeded.", "success"), _footer(rng, platform, 0)]), 0
    if status == "fail":
        errs = _p(rng, [
            [(f"src/{_p(rng, ['main', 'parser', 'io'])}.c:{_num(rng, 10, 400)}:{_num(rng, 1, 40)}: "
              f"error: expected ';' before '}}' token", "fail"),
             ("make: *** [Makefile:24: all] Error 1", "fail")],
            [("error[E0308]: mismatched types", "fail"),
             ("  --> src/main.rs:" + str(_num(rng, 10, 300)) + ":18", "fail"),
             ("error: could not compile `" + _p(rng, PROJECTS) + "` due to 2 previous errors", "fail")],
            [("undefined reference to `" + _p(rng, FUNCS) + "'", "fail"),
             ("collect2: error: ld returned 1 exit status", "fail")],
            [("FAILED: build/CMakeFiles/app.dir/src/main.cpp.o", "fail"),
             ("ninja: build stopped: subcommand failed.", "fail")],
            [("# github.com/org/" + _p(rng, PROJECTS), "fail"),
             ("./server.go:" + str(_num(rng, 10, 300)) + ": undefined: " + _p(rng, FUNCS), "fail")],
        ])
        return _join(head, errs, [_footer(rng, platform, 1)]), 1
    if status == "block":
        body = _p(rng, [
            [("/usr/bin/ld: cannot open output file app: Permission denied", "block"),
             ("collect2: error: ld returned 1 exit status", "fail")],
            [("error: failed to write to /usr/local/lib/libapp.so: Permission denied", "block")],
            [("CMake Error: could not create directory /opt/build: Permission denied", "block")],
        ])
        return _join(head, body, [_footer(rng, platform, 126)]), 126
    body = _fillers(rng, _num(rng, 1, 5), [
        "[ 10%] Building CXX object CMakeFiles/app.dir/src/main.cpp.o",
        "[ 20%] Building CXX object CMakeFiles/app.dir/src/parser.cpp.o",
        "   Compiling libc v0.2.155",
        "   Compiling proc-macro2 v1.0.86",
    ])
    return _join(head, body, [("[ 30%] Building CXX object CMakeFiles/app.dir/src/io.cpp.o", "neutral")]), None


def _gen_windows_cmd(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    cmd = _p(rng, ["dir C:\\proj", "ipconfig /all", "chkdsk C: /f", "sfc /scannow",
                   "tasklist", "reg query HKLM\\SOFTWARE", "copy setup.exe D:\\dist\\",
                   "net use Z: \\\\fileserver\\share"])
    head = [_prompt(rng, "windows", "C:\\proj", cmd)]
    if status == "success":
        body = _p(rng, [
            [" Volume in drive C has no label.", " Volume Serial Number is 4A2B-9C1D", "",
             " Directory of C:\\proj", "", "11/06/2026  10:14 AM    <DIR>          .",
             "11/06/2026  10:14 AM             1,204 main.py", "               1 File(s)          1,204 bytes"],
            ["Windows IP Configuration", "", "   Host Name . . . . . . . . . . . . : workstation",
             "   Primary Dns Suffix  . . . . . . . : corp.local",
             "   IPv4 Address. . . . . . . . . . . : 10.0.0.14",
             "   Default Gateway . . . . . . . . . : 10.0.0.1"],
            ["Beginning verification phase of system scan.",
             "Verification 100% complete.", "Windows Resource Protection did not find any integrity violations."],
            ["Image Name                     PID Session Name        Session#    Mem Usage",
             "========================= ======== ================ =========== ============",
             "System Idle Process              0 Services                   0          8 K",
             "python.exe                    4120 Console                    1     82,144 K"],
        ])
        return _join(head, [(x, "neutral") for x in body],
                     [("The operation completed successfully.", "success"),
                      _footer(rng, "windows", 0)]), 0
    if status == "fail":
        errs = _p(rng, [
            [("The system cannot find the file specified.", "fail")],
            [("The system cannot find the path specified.", "fail")],
            [("Access is denied.", "block")],
            [("Cannot create a file when that file already exists.", "fail")],
            [("The network path was not found.", "fail"),
             ("System error 53 has occurred.", "fail")],
            [("ERROR: The process \"python.exe\" not found.", "fail")],
        ])
        return _join(head, errs, [_footer(rng, "windows", 1)]), 1
    if status == "block":
        body = _p(rng, [
            [("Access is denied.", "block")],
            [("The requested operation requires elevation.", "block")],
            [("You do not have sufficient permissions to access this object.", "block"),
             ("Access is denied.", "block")],
            [("System error 5 has occurred.", "block"), ("Access is denied.", "block")],
        ])
        return _join(head, body, [_footer(rng, "windows", 5)]), 5
    body = _fillers(rng, _num(rng, 1, 5), [
        " Volume in drive C has no label.",
        " Directory of C:\\proj",
        "11/06/2026  10:14 AM    <DIR>          .",
        "11/06/2026  10:14 AM             1,204 main.py",
        "Windows IP Configuration",
        "Verification 30% complete.",
    ])
    return _join(head, body), None


def _gen_powershell(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    cmd = _p(rng, ["Get-ChildItem C:\\proj", "Get-Service | Where-Object Status -eq 'Stopped'",
                   "Install-Module PSReadLine -Force", "Invoke-WebRequest https://example.com",
                   "Get-Process python"])
    head = [_prompt(rng, "windows", "C:\\proj", cmd)]
    if status == "success":
        body = _p(rng, [
            ["    Directory: C:\\proj", "", "Mode                 LastWriteTime         Length Name",
             "----                 -------------         ------ ----", "-a----         11/6/2026  10:14 AM           1204 main.py",
             "-a----         11/6/2026  10:14 AM            842 config.yaml"],
            ["Status   Name               DisplayName", "------   ----               -----------",
             "Stopped  Spooler            Print Spooler", "Stopped  WSearch            Windows Search"],
            ["Name                            Version   Source", "----                            -------   ------",
             "PSReadLine                      2.2.6     PSGallery"],
            ["StatusCode        : 200", "StatusDescription : OK",
             "Content           : <!doctype html>", "Headers           : {[Content-Type, text/html]}"],
        ])
        return _join(head, [(x, "neutral") for x in body],
                     [("Command completed successfully.", "success"),
                      _footer(rng, "windows", 0)]), 0
    if status == "fail":
        errs = _p(rng, [
            [("Get-Item : Cannot find path 'C:\\proj\\missing.txt' because it does not exist.", "fail"),
             ("    + CategoryInfo          : ObjectNotFound: (C:\\proj\\missing.txt:String) [Get-Item], ItemNotFoundException", "fail")],
            [("Install-Module : No match was found for the specified search criteria and module name 'PSReadLine'.", "fail")],
            [("Invoke-WebRequest : The remote name could not be resolved: 'example.com'", "fail")],
            [("Start-Process : This command cannot be run due to the error: The system cannot find the file specified.", "fail")],
            [("Exception calling \"Open\" with \"2\" argument(s): \"Could not load file or assembly\"", "fail")],
        ])
        return _join(head, errs, [_footer(rng, "windows", 1)]), 1
    if status == "block":
        body = _p(rng, [
            [("Install-Module : Access to the path 'C:\\Program Files\\WindowsPowerShell\\Modules' is denied.", "block"),
             ("    + CategoryInfo          : PermissionDenied", "block")],
            [("Invoke-WebRequest : The remote server returned an error: (403) Forbidden.", "block")],
            [("Access to the registry key 'HKEY_LOCAL_MACHINE\\SOFTWARE' is denied.", "block")],
            [("Start-Process : This command cannot be run due to the error: Access is denied.", "block"),
             ("The requested operation requires elevation.", "block")],
            [("AuthorizationManager check failed.", "block"),
             ("File C:\\scripts\\deploy.ps1 cannot be loaded. The file is not digitally signed.", "block")],
        ])
        return _join(head, body, [_footer(rng, "windows", 5)]), 5
    body = _fillers(rng, _num(rng, 1, 5), [
        "    Directory: C:\\proj",
        "Mode                 LastWriteTime         Length Name",
        "----                 -------------         ------ ----",
        "-a----         11/6/2026  10:14 AM           1204 main.py",
        "Status   Name               DisplayName",
        "------   ----               -----------",
    ])
    return _join(head, body), None


def _gen_windows_update(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    pkg = _p(rng, ["Python.Python.3.10", "Microsoft.VisualStudioCode", "Git.Git",
                   "Docker.DockerDesktop", "7zip.7zip", "Mozilla.Firefox"])
    cmd = _p(rng, [f"winget install {pkg}", "winget upgrade --all"])
    head = [_prompt(rng, "windows", "C:\\Users\\dev", cmd)]
    if status == "success":
        body = [
            f"Found {pkg} [{'Python.Python' if 'Python' in pkg else pkg}]",
            "This application is licensed to you by its owner.",
            "Downloading https://github.com/.../installer.exe",
            f"  ██████████████████████████████  {rng.uniform(10, 200):.1f} MB / "
            f"{rng.uniform(10, 200):.1f} MB",
            "Successfully verified installer hash",
            "Starting package install...",
            "Successfully installed",
        ]
        return _join(head, [(x, "neutral") for x in body],
                     [("Installation completed.", "success"),
                      _footer(rng, "windows", 0)]), 0
    if status == "fail":
        errs = [
            _p(rng, [
                ("No package found matching input criteria.", "fail"),
                ("Installer failed with exit code: 1603", "fail"),
                ("Download failed: 404 Not Found", "fail"),
                ("Failed when searching source: msstore", "fail"),
                ("Package installation failed: The system cannot find the file specified.", "fail"),
            ])
        ]
        return _join(head, errs, [_footer(rng, "windows", 1)]), 1
    if status == "block":
        body = [
            ("Installing package requires administrator privileges.", "block"),
            ("Access is denied.", "block"),
            ("The installer was blocked by Windows Defender SmartScreen.", "block"),
            _footer(rng, "windows", 5),
        ]
        return _join(head, body), 5
    body = _fillers(rng, _num(rng, 1, 5), [
        f"Found {pkg}",
        "Downloading https://github.com/.../installer.exe",
        "  ████████████░░░░░░░░░░░░░░░░░░  42.1 MB / 118.0 MB",
        "Starting package install...",
    ])
    return _join(head, body), None


def _gen_security(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    app = _p(rng, ["python3", "node", "docker", "nginx", "backup-agent"])
    cmd = _p(rng, ["sudo ./deploy.sh", "auditctl -l", "tail -n 40 /var/log/auth.log",
                   "Get-WinEvent -LogName Security -MaxEvents 20"])
    head = [_prompt(rng, platform, "~" if platform == "linux" else "C:\\proj", cmd)]
    if status == "success":
        body = [
            f"type=SYSCALL msg=audit({rng.uniform(1, 9):.3f}:{_num(rng, 10, 999)}): "
            f"arch=c000003e syscall=59 success=yes exit=0 a0=... comm=\"{app}\"",
            f"type=EXECVE msg=audit({rng.uniform(1, 9):.3f}:{_num(rng, 10, 999)}): "
            f"argc=2 a0=\"/usr/bin/{app}\" a1=\"--config\"",
            f"type=PROCTITLE msg=audit({rng.uniform(1, 9):.3f}:{_num(rng, 10, 999)}): "
            f"proctitle=2F7573722F62696E2F{app}",
            f"Policy check passed for {app}; no violations detected.",
        ]
        return _join(head, [(x, "neutral") for x in body[:3]],
                     [(body[3], "success"),
                      ("Authorization successful.", "success"),
                      _footer(rng, platform, 0)]), 0
    if status == "fail":
        errs = _p(rng, [
            [("authentication failure; logname= uid=0 euid=0 tty=pts/0 ruser= rhost=10.0.0.14", "fail"),
             ("Failed password for invalid user admin from 10.0.0.14 port 51234 ssh2", "fail")],
            [("sudo: 3 incorrect password attempts", "fail"),
             ("sudo: unable to execute ./deploy.sh: Operation not permitted", "block")],
            [("Access violation: policy engine rejected process start", "fail"),
             ("EventID 4688: process creation blocked", "block")],
            [("type=AVC msg=audit(...): avc: denied { write } for pid=4120", "block"),
             ("Operation not permitted", "block")],
        ])
        return _join(head, errs, [_footer(rng, platform, 1)]), 1
    if status == "block":
        body = _p(rng, [
            [("Access denied: user is not in the sudoers file. This incident will be reported.", "block"),
             ("sudo: a password is required", "block")],
            [("Conditional access policy blocked the sign-in. Reason: device not compliant.", "block"),
             ("ERROR: request blocked by firewall rule 4721", "block")],
            [("Windows Defender: Threat quarantined, operation blocked.", "block"),
             ("Access to the object is denied by policy.", "block")],
            [("seccomp: audit: type=1326 ... syscall=257 code=0x0", "block"),
             ("Container process blocked by seccomp profile.", "block")],
        ])
        return _join(head, body, [_footer(rng, platform, 126)]), 126
    body = _fillers(rng, _num(rng, 1, 5), [
        f"type=SYSCALL msg=audit({rng.uniform(1, 9):.3f}:{_num(rng, 10, 999)}): "
        f"syscall=59 success=yes exit=0",
        "type=EXECVE msg=audit(...): argc=2",
        "tail: /var/log/auth.log: file truncated",
    ])
    return _join(head, body), None


def _gen_shell(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    cmd = _p(rng, ["ls -la", "df -h", "free -h", "ps aux", "env", "uname -a",
                   "find . -name '*.log'", "du -sh *", "top -b -n 1"])
    head = [_prompt(rng, platform, "~/proj" if platform == "linux" else "C:\\proj", cmd)]
    if status == "success":
        body = _fillers(rng, _num(rng, 4, 16), [
            f"total {_num(rng, 1, 100)}",
            f"drwxr-xr-x {_num(rng, 1, 9)} {_p(rng, USERS)} {_p(rng, USERS)} 4096 Nov  6 10:14 .",
            f"-rw-r--r-- 1 {_p(rng, USERS)} {_p(rng, USERS)} {_num(rng, 100, 9999)} Nov  6 10:14 {_p(rng, FILES)}",
            f"/dev/nvme0n1p2  {_num(rng, 100, 900)}G  {_num(rng, 50, 800)}G  "
            f"{_num(rng, 10, 100)}G  {_num(rng, 10, 99)}% /",
            f"               total        used        free      shared  buff/cache   available",
            f"Mem:           {_num(rng, 8, 128)}Gi        {_num(rng, 1, 60)}Gi        "
            f"{_num(rng, 1, 40)}Gi        {_num(rng, 1, 9)}Gi        {_num(rng, 1, 40)}Gi        {_num(rng, 1, 60)}Gi",
            f"Linux {_p(rng, HOSTS)} 6.8.0-45-generic #45-Ubuntu SMP x86_64 GNU/Linux",
            f"{_p(rng, USERS)} {_num(rng, 1000, 20000)}  0.0  0.1  {_num(rng, 10000, 99999)}  "
            f"{_num(rng, 1000, 20000)} ?  Ss  10:14   0:00 {_p(rng, SERVICES)}",
        ])
        return _join(head, body, [_footer(rng, platform, 0)]), 0
    if status == "fail":
        errs = _p(rng, [
            [(f"ls: cannot access '{_p(rng, FILES)}': No such file or directory", "fail")],
            [(f"find: ‘{_p(rng, DIRS)}’: No such file or directory", "fail")],
            [("df: /mnt/data: Stale file handle", "fail")],
            [("free: cannot read /proc/meminfo", "fail")],
            [(f"ps: error: unknown option -- z", "fail")],
            [("top: failed tty get", "fail")],
        ])
        return _join(head, errs, [_footer(rng, platform, 1)]), 1
    if status == "block":
        body = _p(rng, [
            [(f"ls: cannot open directory '{_p(rng, DIRS)}': Permission denied", "block")],
            [(f"find: ‘{_p(rng, DIRS)}’: Permission denied", "block")],
            [("Operation not permitted", "block")],
            [("Access is denied.", "block")],
        ])
        return _join(head, body, [_footer(rng, platform, 126)]), 126
    body = _fillers(rng, _num(rng, 1, 6), [
        f"total {_num(rng, 1, 100)}",
        f"-rw-r--r-- 1 {_p(rng, USERS)} {_p(rng, USERS)} {_num(rng, 100, 9999)} {_p(rng, FILES)}",
        f"/dev/nvme0n1p2  {_num(rng, 100, 900)}G  {_num(rng, 50, 800)}G  {_num(rng, 10, 100)}G  {_num(rng, 10, 99)}% /",
        f"Mem:           {_num(rng, 8, 128)}Gi        {_num(rng, 1, 60)}Gi        {_num(rng, 1, 40)}Gi",
    ])
    return _join(head, body), None


_GENERATORS = {
    "python": _gen_python,
    "pip": _gen_pip,
    "pytest": _gen_pytest,
    "conda": _gen_conda,
    "shell": _gen_shell,
    "package": _gen_package,
    "systemd": _gen_systemd,
    "network": _gen_network,
    "filesystem": _gen_filesystem,
    "git": _gen_git,
    "docker": _gen_docker,
    "kernel": _gen_kernel,
    "journal": _gen_journal,
    "build": _gen_build,
    "windows_cmd": _gen_windows_cmd,
    "powershell": _gen_powershell,
    "windows_update": _gen_windows_update,
    "security": _gen_security,
}

# Domains implemented in synth_extra.py (package updates, compilation,
# media/data processing, archives, backups, databases, cloud infrastructure).
from .synth_extra import EXTRA_GENERATORS, context_lines  # noqa: E402
from .synth_commands import COMMAND_GENERATORS  # noqa: E402
from .synth_errors import ERROR_GENERATORS  # noqa: E402
from .synth_real_errors import REAL_ERROR_GENERATORS  # noqa: E402
from .synth_multi import MULTI_GENERATORS  # noqa: E402
from .synth_modern import MODERN_GENERATORS  # noqa: E402

_GENERATORS.update(EXTRA_GENERATORS)
_GENERATORS.update(COMMAND_GENERATORS)
_GENERATORS.update(ERROR_GENERATORS)
_GENERATORS.update(REAL_ERROR_GENERATORS)
_GENERATORS.update(MULTI_GENERATORS)
_GENERATORS.update(MODERN_GENERATORS)

_WINDOWS_ONLY = {"windows_cmd", "powershell", "windows_update"}
_LINUX_ONLY = {
    "shell", "package", "systemd", "kernel", "journal", "docker", "git",
    "kubectl", "helm", "terraform", "ansible", "github_actions", "gitlab_ci",
    "jenkins", "redis", "postgres", "mysql", "nginx",
}


def _postprocess(
    rng: random.Random,
    lines: List[Line],
    platform: str,
    status: str,
    completed: bool,
    cfg: SynthConfig,
) -> List[Line]:
    out: List[Line] = []
    ts_rate = cfg.timestamp_rate
    for i, (text, label) in enumerate(lines):
        if not text:
            out.append(("", label))
            continue
        # Timestamp / PID decoration on neutral output only.
        if label == "neutral" and _maybe(rng, ts_rate) and platform == "linux":
            text = _stamp(rng, 1.0, "linux") + text
        elif label == "neutral" and _maybe(rng, ts_rate) and platform == "windows":
            text = _stamp(rng, 1.0, "windows") + text
        if label == "fail" and _maybe(rng, cfg.ansi_rate):
            text = f"\x1b[31m{text}\x1b[0m"
        elif label == "block" and _maybe(rng, cfg.ansi_rate):
            text = f"\x1b[33m{text}\x1b[0m"
        elif label == "success" and _maybe(rng, cfg.ansi_rate):
            text = f"\x1b[32m{text}\x1b[0m"
        out.append((text, label))

    # Truncate a session in the middle of the body, never the prompt line.
    if not completed and status == "unknown" and len(out) > 2:
        cut = rng.randint(2, len(out) - 1)
        out = out[:cut]
        if _maybe(rng, 0.5) and out:
            text, label = out[-1]
            if text:
                out[-1] = (text[: max(4, len(text) // 2)], label)
    return out


_SPLIT_PROFILES = {
    "train": {"seed": 0, "context": (1, 9), "complex": False, "truncation": 1.0},
    "val": {"seed": 100003, "context": (3, 14), "complex": True, "truncation": 0.9},
    "test": {"seed": 200003, "context": (5, 20), "complex": True, "truncation": 0.8},
}


def generate_document(
    rng: random.Random, cfg: SynthConfig, index: int, split: str = "train"
) -> Dict:
    profile = _SPLIT_PROFILES.get(split, _SPLIT_PROFILES["train"])
    cfg._split = split
    domains = list(cfg.domains) if cfg.domains else list(_GENERATORS)
    domain = _p(rng, domains)
    if domain not in _GENERATORS:
        domain = _p(rng, list(_GENERATORS))
    if domain in _WINDOWS_ONLY:
        platform = "windows"
    elif domain in _LINUX_ONLY:
        platform = "linux"
    else:
        platform = _p(rng, cfg.platforms or ["linux", "windows"])

    statuses = list(cfg.status_weights.keys())
    weights = [cfg.status_weights[s] for s in statuses]
    status = rng.choices(statuses, weights=weights, k=1)[0]
    if domain in ("errors", "real_errors") and status not in ("fail", "block"):
        status = rng.choice(["fail", "block"])
    if domain in ("multi", "multi_chain") and _maybe(rng, 0.15):
        # occasional truncated multi-command session
        status = "unknown" if status != "block" else status

    lines, exit_code = _GENERATORS[domain](rng, platform, status, cfg)

    # Longer, noisier transcripts: insert neutral startup/status lines around
    # the actual operation output.  This gives the chunk model more context
    # without inventing success/failure markers.
    if _maybe(rng, 0.75) and len(lines) > 2:
        extra = context_lines(rng, _num(rng, *profile["context"]))
        pos = rng.randint(1, max(1, len(lines) - 1))
        lines = lines[:pos] + extra + lines[pos:]

    completed = status in ("success", "fail")
    if status == "unknown" and _maybe(rng, 0.35):
        completed = True  # finished but no recognisable status marker
    if status == "unknown" and cfg.truncation_rate > 0:
        completed = _maybe(rng, 1.0 - cfg.truncation_rate)

    lines = _postprocess(rng, lines, platform, status, completed, cfg)

    # Hard negatives: successful runs that mention errors without failing.
    if status == "success" and _maybe(rng, cfg.hard_negative_rate):
        pos = rng.randrange(max(1, len(lines) - 1))
        decoys = [
            ("warning: previous error was handled, continuing", "neutral"),
            ("0 errors, 0 warnings", "neutral"),
            ("error handling initialized (strict=false)", "neutral"),
            ("INFO: ignore the word error in this informational message", "neutral"),
            ("retry 1/3 after transient failure: recovered", "neutral"),
        ]
        lines.insert(pos, _p(rng, decoys))

    lines = _postprocess(rng, lines, platform, status, completed, cfg) if _maybe(rng, 0.0) else lines

    command = ""
    for text, _ in lines:
        clean = text.replace("\x1b", "")
        if "$ " in clean:
            command = clean.split("$ ", 1)[1]
            break
        if "> " in clean and platform == "windows" and not command.startswith("PS"):
            command = clean.split("> ", 1)[1]
            break
    return {
        "id": f"{status[:3]}-{index:06d}",
        "split": split,
        "platform": platform,
        "domain": domain,
        "command": command,
        "status": status,
        "completed": bool(completed),
        "lines": [{"text": t, "label": l} for t, l in lines],
    }


def generate_corpus(cfg: SynthConfig) -> Dict[str, List[Dict]]:
    splits = {}
    for name, n in (
        ("train", cfg.n_train),
        ("val", cfg.n_val),
        ("test", cfg.n_test),
    ):
        profile = _SPLIT_PROFILES[name]
        rng = random.Random(cfg.seed + profile["seed"])
        docs = []
        for i in range(n):
            docs.append(generate_document(rng, cfg, i, split=name))
        splits[name] = docs
    return splits


def write_jsonl(docs: Sequence[Dict], path: str) -> None:
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict]:
    docs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def corpus_stats(docs: Sequence[Dict]) -> Dict[str, object]:
    stats: Dict[str, object] = {
        "documents": len(docs),
        "status": {},
        "platform": {},
        "domain": {},
        "completed": {},
        "lines": 0,
        "avg_lines": 0.0,
        "marks": {"success": 0, "fail": 0, "block": 0, "neutral": 0},
    }
    for doc in docs:
        stats["status"][doc["status"]] = stats["status"].get(doc["status"], 0) + 1
        stats["platform"][doc["platform"]] = stats["platform"].get(doc["platform"], 0) + 1
        stats["domain"][doc["domain"]] = stats["domain"].get(doc["domain"], 0) + 1
        stats["completed"][str(bool(doc["completed"]))] = (
            stats["completed"].get(str(bool(doc["completed"])), 0) + 1
        )
        stats["lines"] += len(doc["lines"])
        for line in doc["lines"]:
            stats["marks"][line["label"]] = stats["marks"].get(line["label"], 0) + 1
    stats["avg_lines"] = round(stats["lines"] / max(1, len(docs)), 2)
    return stats


def write_corpus(cfg: SynthConfig, out_dir: str, verbose: bool = True) -> Dict[str, object]:
    splits = generate_corpus(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"config": asdict(cfg), "splits": {}}
    for name, docs in splits.items():
        write_jsonl(docs, out / f"{name}.jsonl")
        meta["splits"][name] = corpus_stats(docs)
        if verbose:
            print(f"[synth] {name:5s} {len(docs):5d} docs -> {out / (name + '.jsonl')}")
    meta["total"] = sum(len(v) for v in splits.values())
    with open(out / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    with open(out / "README.txt", "w", encoding="utf-8") as fh:
        fh.write(
            "Synthetic console corpus generated by console_nn.synth.\n"
            "JSONL schema: id, split, platform, domain, command, status, "
            "completed, lines[{text,label}].\n"
            f"Total documents: {meta['total']}\n"
        )
    if verbose:
        print(f"[synth] total {meta['total']} documents")
        print(json.dumps(meta["splits"]["train"], indent=2))
    return meta
