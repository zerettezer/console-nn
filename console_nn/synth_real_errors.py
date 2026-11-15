"""Real Linux/Windows error sessions supplied as concrete error families."""
from __future__ import annotations

import random
from typing import Dict, List

from .synth import _p, _prompt

SPECS: List[Dict] = [
    {
        "name": "make_openssl_missing",
        "platform": "linux", "status": "fail",
        "cmd": "make",
        "body": [
            ("gcc -Wall -o main main.c -lssl", "neutral"),
            ("main.c:3:10: fatal error: openssl/ssl.h: No such file or directory", "fail"),
            ("    3 | #include <openssl/ssl.h>", "fail"),
            ("      |          ^~~~~~~~~~~~~~~", "fail"),
            ("compilation terminated.", "fail"),
            ("make: *** [Makefile:4: main] Error 1", "fail"),
        ],
    },
    {
        "name": "nvidia_nvml_mismatch",
        "platform": "linux", "status": "fail",
        "cmd": "watch nvidia-smi",
        "body": [
            ("Every 2.0s: nvidia-smi                          workstation: Sat Nov  7 12:00:01 2026", "neutral"),
            ("", "neutral"),
            ("Failed to initialize NVML: Driver/library version mismatch", "fail"),
            ("NVML library version: 560.35", "fail"),
        ],
    },
    {
        "name": "apt_lock",
        "platform": "linux", "status": "block",
        "cmd": "apt install nginx",
        "body": [
            ("E: Could not open lock file /var/lib/dpkg/lock-frontend - open (13: Permission denied)", "block"),
            ("E: Unable to acquire the dpkg frontend lock (/var/lib/dpkg/lock-frontend), are you root?", "block"),
        ],
    },
    {
        "name": "apt_unable_locate",
        "platform": "linux", "status": "fail",
        "cmd": "sudo apt install nginx",
        "body": [
            ("Reading package lists... Done", "neutral"),
            ("Building dependency tree... Done", "neutral"),
            ("E: Unable to locate package nginx", "fail"),
        ],
    },
    {
        "name": "docker_permission",
        "platform": "linux", "status": "block",
        "cmd": "docker run hello-world",
        "body": [
            ("docker: permission denied while trying to connect to the Docker daemon socket at", "block"),
            ("unix:///var/run/docker.sock: Post \"http://%2Fvar%2Frun%2Fdocker.sock/v1.47/containers/create\":", "block"),
            ("dial unix /var/run/docker.sock: connect: permission denied.", "block"),
            ("See 'docker run --help'.", "block"),
        ],
    },
    {
        "name": "git_push_rejected",
        "platform": "linux", "status": "fail",
        "cmd": "git push",
        "body": [
            ("To github.com:user/repo.git", "neutral"),
            (" ! [rejected]        main -> main (fetch first)", "fail"),
            ("error: failed to push some refs to 'github.com:user/repo.git'", "fail"),
            ("hint: Updates were rejected because the remote contains work that you do not", "fail"),
            ("hint: have locally. This is usually caused by another repository pushing to", "fail"),
            ("hint: the same ref. If you want to integrate the remote changes, use", "fail"),
            ("hint: 'git pull' before pushing again.", "fail"),
            ("hint: See the 'Note about fast-forwards' in 'git push --help' for details.", "fail"),
        ],
    },
    {
        "name": "npm_eresolve_react",
        "platform": "linux", "status": "fail",
        "cmd": "npm install",
        "body": [
            ("npm error code ERESOLVE", "fail"),
            ("npm error ERESOLVE could not resolve", "fail"),
            ("npm error While resolving: react-scripts@5.0.1", "fail"),
            ("npm error Found: typescript@5.6.3", "fail"),
            ("npm error node_modules/typescript", "fail"),
            ("npm error   peerOptional typescript@\"*\" from react-scripts@5.0.1", "fail"),
            ("npm error Could not resolve dependency:", "fail"),
            ("npm error peerOptional typescript@\"^3.2.1 || ^4\" from react-scripts@5.0.1", "fail"),
            ("npm error Fix the upstream dependency conflict, or retry", "fail"),
            ("npm error this command with --force or --legacy-peer-deps", "fail"),
        ],
    },
    {
        "name": "python_flask_missing",
        "platform": "linux", "status": "fail",
        "cmd": "python3 app.py",
        "body": [
            ("Traceback (most recent call last):", "fail"),
            ('  File "/home/dev/app.py", line 1, in <module>', "fail"),
            ("    from flask import Flask", "fail"),
            ("ModuleNotFoundError: No module named 'flask'", "fail"),
        ],
    },
    {
        "name": "pytest_assertion",
        "platform": "linux", "status": "fail",
        "cmd": "pytest",
        "body": [
            ("============================= test session starts ==============================", "neutral"),
            ("collected 3 items", "neutral"),
            ("tests/test_api.py .F.                                                    [100%]", "neutral"),
            ("=================================== FAILURES ===================================", "fail"),
            ("______________________________ test_get_user ___________________________________", "fail"),
            ("    def test_get_user():", "fail"),
            ('>       assert get_user(1)["name"] == "Bob"', "fail"),
            ("E       AssertionError: assert 'Alice' == 'Bob'", "fail"),
            ("E         - Bob", "fail"),
            ("E         + Alice", "fail"),
            ("tests/test_api.py:12: AssertionError", "fail"),
            ("FAILED tests/test_api.py::test_get_user - AssertionError: assert 'Alice' == 'Bob'", "fail"),
            ("========================= 1 failed, 2 passed in 0.14s ==========================", "fail"),
        ],
    },
    {
        "name": "cargo_borrow_moved",
        "platform": "linux", "status": "fail",
        "cmd": "cargo build",
        "body": [
            ("   Compiling myapp v0.1.0 (/home/dev/myapp)", "neutral"),
            ("error[E0382]: borrow of moved value: `name`", "fail"),
            (" --> src/main.rs:5:20", "fail"),
            ("  |", "fail"),
            ("3 |     let name = String::from(\"bob\");", "fail"),
            ("  |         ---- move occurs because `name` has type `String`", "fail"),
            ("5 |     println!(\"{}\", name);", "fail"),
            ("  |                    ^^^^ value borrowed here after move", "fail"),
            ("error: could not compile `myapp` (bin \"myapp\") due to 1 previous error", "fail"),
        ],
    },
    {
        "name": "ssh_host_key_changed",
        "platform": "linux", "status": "block",
        "cmd": "ssh user@192.168.1.10",
        "body": [
            ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", "block"),
            ("@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @", "block"),
            ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", "block"),
            ("IT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!", "block"),
            ("Someone could be eavesdropping on you right now (man-in-the-middle attack)!", "block"),
            ("Offending ECDSA key in /home/dev/.ssh/known_hosts:7", "block"),
            ("Host key verification failed.", "block"),
        ],
    },
    {
        "name": "systemctl_myapp_failed",
        "platform": "linux", "status": "fail",
        "cmd": "systemctl start myapp",
        "body": [
            ("Job for myapp.service failed because the control process exited with error code.", "fail"),
            ("See \"systemctl status myapp.service\" and \"journalctl -xeu myapp.service\" for details.", "fail"),
        ],
    },
    {
        "name": "kubectl_validation",
        "platform": "linux", "status": "fail",
        "cmd": "kubectl apply -f deploy.yaml",
        "body": [
            ("error: error validating \"deploy.yaml\": error validating data: ValidationError(Deployment.spec):", "fail"),
            ("missing required field \"selector\" in io.k8s.api.apps.v1.DeploymentSpec;", "fail"),
            ("if you choose to ignore these errors, turn validation off with --validate=false", "fail"),
        ],
    },
    {
        "name": "terraform_lock",
        "platform": "linux", "status": "fail",
        "cmd": "terraform apply",
        "body": [
            ("╷", "neutral"),
            ("│ Error: Error acquiring the state lock", "fail"),
            ("│ Error message: resource temporarily unavailable", "fail"),
            ("│ Lock Info:", "fail"),
            ("│   ID:        6f1c2a3e-8d4b-1c9a-7e55-0a4f3b2d9c11", "fail"),
            ("│   Path:      terraform.tfstate", "fail"),
            ("│   Operation: OperationTypeApply", "fail"),
            ("╵", "neutral"),
        ],
    },
    {
        "name": "curl_ssl_cert",
        "platform": "linux", "status": "fail",
        "cmd": "curl https://example.com",
        "body": [
            ("curl: (60) SSL certificate problem: unable to get local issuer certificate", "fail"),
            ("More details here: https://curl.se/docs/sslcerts.html", "neutral"),
            ("curl failed to verify the legitimacy of the server and therefore could not", "fail"),
            ("establish a secure connection to it.", "fail"),
        ],
    },
    {
        "name": "msbuild_cs0103",
        "platform": "windows", "status": "fail",
        "cmd": "msbuild App.sln",
        "body": [
            ("Microsoft (R) Build Engine version 17.11.9+a69bbaaf5 for .NET Framework", "neutral"),
            ("Build started 07.11.2026 12:00:01.", "neutral"),
            ("Program.cs(14,21): error CS0103: The name 'userName' does not exist in the current context", "fail"),
            ("Build FAILED.", "fail"),
            ("    0 Warning(s)", "fail"),
            ("    1 Error(s)", "fail"),
        ],
    },
    {
        "name": "powershell_execution_policy",
        "platform": "windows", "status": "block",
        "cmd": ".\\script.ps1",
        "body": [
            (".\\script.ps1 : File C:\\Users\\user\\script.ps1 cannot be loaded because running", "block"),
            ("scripts is disabled on this system. For more information, see about_Execution_Policies.", "block"),
            ("    + CategoryInfo          : SecurityError: (:) [], PSSecurityException", "block"),
            ("    + FullyQualifiedErrorId : UnauthorizedAccess", "block"),
        ],
    },
    {
        "name": "windows_pip_missing",
        "platform": "windows", "status": "fail",
        "cmd": "pip install numpy",
        "body": [
            ("'pip' is not recognized as an internal or external command,", "fail"),
            ("operable program or batch file.", "fail"),
        ],
    },
    {
        "name": "npm_openssl_unsupported",
        "platform": "windows", "status": "fail",
        "cmd": "npm run build",
        "body": [
            ("> app@1.0.0 build", "neutral"),
            ("> webpack --mode production", "neutral"),
            ("Error: error:0308010C:digital envelope routines::unsupported", "fail"),
            ("    at new Hash (node:internal/crypto/hash:79:19)", "fail"),
            ("  opensslErrorStack: [ 'error:03000086:digital envelope routines::initialization error' ],", "fail"),
            ("  code: 'ERR_OSSL_EVP_UNSUPPORTED'", "fail"),
        ],
    },
    {
        "name": "windows_access_denied_hosts",
        "platform": "windows", "status": "block",
        "cmd": "del C:\\Windows\\System32\\drivers\\etc\\hosts",
        "body": [
            ("C:\\Windows\\System32\\drivers\\etc\\hosts", "block"),
            ("Access is denied.", "block"),
        ],
    },
    {
        "name": "windows_net_start_access",
        "platform": "windows", "status": "block",
        "cmd": "net start MySQL",
        "body": [
            ("System error 5 has occurred.", "block"),
            ("Access is denied.", "block"),
        ],
    },
    {
        "name": "windows_get_content_missing",
        "platform": "windows", "status": "fail",
        "cmd": "Get-Content C:\\logs\\app.log",
        "body": [
            ("Get-Content : Cannot find path 'C:\\logs\\app.log' because it does not exist.", "fail"),
            ("At line:1 char:1", "fail"),
            ("    + CategoryInfo          : ObjectNotFound: (C:\\logs\\app.log:String)", "fail"),
            ("    + FullyQualifiedErrorId : PathNotFound", "fail"),
        ],
    },
    {
        "name": "winget_msstore_missing",
        "platform": "windows", "status": "fail",
        "cmd": "winget install Git.Git",
        "body": [
            ("Failed when searching source: msstore", "fail"),
            ("An unexpected error occurred while executing the command:", "fail"),
            ("0x8a15000f : Data required by the source is missing", "fail"),
        ],
    },
]


def _fmt(text: str) -> str:
    return text


def gen_real_errors(rng: random.Random, platform: str, status: str, cfg):
    pool = [s for s in SPECS if s["status"] == status]
    if not pool:
        pool = SPECS
    spec = _p(rng, pool)
    plat = spec["platform"]
    head = [(_prompt(rng, plat, "~/proj" if plat == "linux" else "C:\\src", spec["cmd"]), "neutral")]
    return head + [(text, label) for text, label in spec["body"]], (0 if status == "success" else 1)


REAL_ERROR_GENERATORS = {"real_errors": gen_real_errors}
REAL_ERROR_SPECS = SPECS
