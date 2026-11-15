"""Twenty real-world console error families.

These are deliberately short, realistic sessions.  They exercise the model on
errors that are not tied to one package manager or one operating system.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from .synth import (
    FILES,
    PACKAGES,
    PROJECTS,
    USERS,
    Line,
    _num,
    _p,
    _prompt,
)

_SPECS: List[Dict] = [
    {
        "name": "pep668",
        "status": "block",
        "cmd": "python3 -m pip install {pkg}",
        "body": [
            ("error: externally-managed-environment", "block"),
            ("", "neutral"),
            ("× This environment is externally managed", "block"),
            ("╰─> To install Python packages system-wide, try apt install", "block"),
            ("    python3-xyz, where xyz is the package you are trying to", "block"),
            ("    install.", "block"),
            ("", "neutral"),
            ("    If you wish to install a non-Debian-packaged Python package,", "block"),
            ("    create a virtual environment using python3 -m venv path/to/venv.", "block"),
            ("    Then use path/to/venv/bin/python and path/to/venv/bin/pip.", "block"),
            ("", "neutral"),
            ("note: If you believe this is a mistake, please contact your Python", "block"),
            ("installation or OS distribution provider. You can override this, at the", "block"),
            ("risk of breaking your Python installation or OS, by passing", "block"),
            ("--break-system-packages.", "block"),
            ("hint: See PEP 668 for the detailed specification.", "block"),
        ],
    },
    {
        "name": "cuda_oom",
        "status": "fail",
        "cmd": "python train.py --config config.yaml",
        "body": [
            ("Traceback (most recent call last):", "fail"),
            ('  File "train.py", line 42, in <module>', "fail"),
            ("    trainer.fit(model, data)", "fail"),
            ('  File "utils/trainer.py", line 118, in fit', "fail"),
            ("RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB", "fail"),
            ("GPU 0 has a total capacity of 24.00 GiB of which 112.50 MiB is free", "fail"),
        ],
    },
    {
        "name": "module_not_found",
        "status": "fail",
        "cmd": "python app.py",
        "body": [
            ("Traceback (most recent call last):", "fail"),
            ('  File "app.py", line 3, in <module>', "fail"),
            ("    import yaml", "fail"),
            ("ModuleNotFoundError: No module named 'yaml'", "fail"),
        ],
    },
    {
        "name": "permission_denied",
        "status": "block",
        "cmd": "cp model.bin /opt/app/models/",
        "body": [
            ("cp: cannot create regular file '/opt/app/models/model.bin': Permission denied", "block"),
        ],
    },
    {
        "name": "disk_full",
        "status": "fail",
        "cmd": "python export.py --out /mnt/data/export.parquet",
        "body": [
            ("  0%|          | 0/100 [00:00<?, ?it/s]", "neutral"),
            (" 99%|█████████▉| 99/100 [00:12<00:00,  8.12it/s]", "neutral"),
            ("OSError: [Errno 28] No space left on device", "fail"),
            ("  File \"/mnt/data/export.parquet\", line 1", "fail"),
        ],
    },
    {
        "name": "port_in_use",
        "status": "fail",
        "cmd": "python -m uvicorn app:api --port 8080",
        "body": [
            ("INFO:     Started server process [4120]", "neutral"),
            ("ERROR:    [Errno 98] error while attempting to bind on address ('0.0.0.0', 8080)", "fail"),
            ("ERROR:    address already in use", "fail"),
        ],
    },
    {
        "name": "connection_refused",
        "status": "fail",
        "cmd": "curl -v http://127.0.0.1:5432/health",
        "body": [
            ("*   Trying 127.0.0.1:5432...", "neutral"),
            ("* connect to 127.0.0.1 port 5432 failed: Connection refused", "fail"),
            ("curl: (7) Failed to connect to 127.0.0.1 port 5432: Connection refused", "fail"),
        ],
    },
    {
        "name": "dns_failure",
        "status": "fail",
        "cmd": "curl https://internal.corp.local/health",
        "body": [
            ("curl: (6) Could not resolve host: internal.corp.local", "fail"),
            (";; connection timed out; no servers could be reached", "fail"),
        ],
    },
    {
        "name": "tls_verify",
        "status": "fail",
        "cmd": "curl https://api.example.com/v1/status",
        "body": [
            ("curl: (60) SSL certificate problem: unable to get local issuer certificate", "fail"),
            ("More details here: https://curl.se/docs/sslcerts.html", "neutral"),
            ("curl failed to verify the legitimacy of the server", "fail"),
        ],
    },
    {
        "name": "pip_conflict",
        "status": "fail",
        "cmd": "python -m pip install -r requirements.txt",
        "body": [
            ("Collecting pydantic==1.10.0", "neutral"),
            ("Collecting fastapi==0.115.0", "neutral"),
            ("ERROR: Cannot install fastapi==0.115.0 and pydantic==1.10.0 because", "fail"),
            ("these package versions have conflicting dependencies.", "fail"),
            ("The conflict is caused by:", "fail"),
            ("    fastapi 0.115.0 depends on pydantic>=2.0.0", "fail"),
        ],
    },
    {
        "name": "dpkg_lock",
        "status": "block",
        "cmd": "sudo apt-get install -y nginx",
        "body": [
            ("E: Could not open lock file /var/lib/dpkg/lock-frontend - open (13: Permission denied)", "block"),
            ("E: Unable to acquire the dpkg frontend lock, are you root?", "block"),
            ("sudo: a password is required", "block"),
        ],
    },
    {
        "name": "docker_daemon",
        "status": "fail",
        "cmd": "docker ps",
        "body": [
            ("Cannot connect to the Docker daemon at unix:///var/run/docker.sock.", "fail"),
            ("Is the docker daemon running?", "fail"),
        ],
    },
    {
        "name": "systemd_failed",
        "status": "fail",
        "cmd": "sudo systemctl start app.service",
        "body": [
            ("Job for app.service failed because the control process exited with error code.", "fail"),
            ("See \"systemctl status app.service\" and \"journalctl -xeu app.service\" for details.", "fail"),
            ("● app.service - Application API", "neutral"),
            ("     Active: failed (Result: exit-code)", "fail"),
        ],
    },
    {
        "name": "oom_killer",
        "status": "fail",
        "cmd": "python worker.py --jobs 32",
        "body": [
            ("Killed", "fail"),
            ("[ 4218.442] Out of memory: Killed process 4120 (python3)", "fail"),
            ("[ 4218.443] oom_reaper: reaped process 4120 (python3)", "fail"),
        ],
    },
    {
        "name": "segfault",
        "status": "fail",
        "cmd": "./build/app --input data.bin",
        "body": [
            ("Segmentation fault (core dumped)", "fail"),
            ("[ 912.114] app[4120]: segfault at 0 ip 00007f42 sp 00007ffd error 4", "fail"),
        ],
    },
    {
        "name": "git_auth",
        "status": "block",
        "cmd": "git push origin main",
        "body": [
            ("git@github.com: Permission denied (publickey).", "block"),
            ("fatal: Could not read from remote repository.", "block"),
            ("Please make sure you have the correct access rights and the repository exists.", "block"),
        ],
    },
    {
        "name": "ssh_host_key",
        "status": "block",
        "cmd": "ssh deploy@10.0.0.14",
        "body": [
            ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", "block"),
            ("@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @", "block"),
            ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", "block"),
            ("Host key verification failed.", "block"),
        ],
    },
    {
        "name": "java_oom",
        "status": "fail",
        "cmd": "java -Xmx2g -jar app.jar",
        "body": [
            ("Exception in thread \"main\" java.lang.OutOfMemoryError: Java heap space", "fail"),
            ("    at java.base/java.util.Arrays.copyOf(Arrays.java:3537)", "fail"),
            ("    at com.example.App.load(App.java:42)", "fail"),
        ],
    },
    {
        "name": "npm_eresolve",
        "status": "fail",
        "cmd": "npm install",
        "body": [
            ("npm ERR! code ERESOLVE", "fail"),
            ("npm ERR! ERESOLVE unable to resolve dependency tree", "fail"),
            ("npm ERR! Found: react@18.3.1", "fail"),
            ("npm ERR! Could not resolve dependency: react-dom@17.0.2", "fail"),
        ],
    },
    {
        "name": "windows_access_denied",
        "status": "block",
        "cmd": "copy setup.exe C:\\Program Files\\App\\",
        "body": [
            ("Access is denied.", "block"),
            ("The requested operation requires elevation.", "block"),
            ("        0 file(s) copied.", "block"),
        ],
    },
]


def _ctx(rng: random.Random) -> Dict[str, str]:
    return {
        "pkg": _p(rng, PACKAGES),
        "file": _p(rng, FILES),
        "project": _p(rng, PROJECTS),
        "user": _p(rng, USERS),
        "port": str(_p(rng, [5000, 5432, 6379, 8000, 8080, 8081, 9000])),
    }


def gen_errors(rng: random.Random, platform: str, status: str, cfg):
    pool = [s for s in _SPECS if s["status"] == status]
    if not pool:
        pool = _SPECS
    spec = _p(rng, pool)
    ctx = _ctx(rng)
    cmd = spec["cmd"].format(**ctx)
    head = [(_prompt(rng, platform, "~/proj" if platform == "linux" else "C:\\proj", cmd), "neutral")]
    body: List[Line] = []
    for text, label in spec["body"]:
        try:
            text = text.format(**ctx)
        except (KeyError, IndexError, ValueError):
            pass
        body.append((text, label))
    if status == "unknown":
        body = [(text, "neutral") for text, _ in body[: max(1, len(body) // 2)]]
    return head + body, (0 if status == "success" else 1)


ERROR_GENERATORS = {"errors": gen_errors}
ERROR_SPECS = _SPECS
