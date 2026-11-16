"""Domain generators for modern language, orchestration and data services.

Kept separate from :mod:`console_nn.synth` so the core generator stays
readable.  Each function follows the same contract:
``(rng, platform, status, cfg) -> (lines, exit_code)``.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from .synth import (
    FILES,
    PROJECTS,
    Line,
    _footer,
    _join,
    _num,
    _p,
    _prompt,
)

_VERSIONS = [
    "20.11.1", "18.19.0", "1.77.2", "1.61.0", "3.11.4", "8.7.0", "2.0.3",
    "7.2.4", "16.2", "15.6", "24.0.1", "3.13.0",
]


def _cwd(platform: str, project: str = "app") -> str:
    return f"~/proj/{project}" if platform == "linux" else f"C:\\proj\\{project}"


def _lines(*items) -> List[Line]:
    out: List[Line] = []
    for item in items:
        if isinstance(item, tuple):
            out.append(item)
        else:
            out.append((str(item), "neutral"))
    return out


# --------------------------------------------------------------------------
# language ecosystems
# --------------------------------------------------------------------------
def _gen_node(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    cmd = f"node {_p(rng, ['app.js', 'server.js', 'index.js'])}"
    head = [_prompt(rng, platform, _cwd(platform), cmd)]
    if status == "success":
        body = _lines(
            f"Server listening on http://127.0.0.1:{_num(rng, 3000, 9999)}",
            f"connected to redis://127.0.0.1:6379",
            f"loaded {_num(rng, 3, 80)} routes",
            ("ready in " + f"{rng.uniform(0.1, 2.5):.2f}s", "success"),
        )
        return _join(head, body), 0
    if status == "fail":
        body = _lines(
            "        throw new Error('missing API token');",
            "        ^",
            ("SyntaxError: Unexpected token '}'", "fail"),
            "    at Module._compile (node:internal/modules/cjs/loader:1376:14)",
            f"    at Object.<anonymous> ({_p(rng, FILES)}:1:1)",
            ("Node.js v20.11.1", "fail"),
            _footer(rng, platform, 1),
        )
        return _join(head, body), 1
    if status == "block":
        body = _lines(
            "node:internal/fs/utils:347",
            ("EACCES: permission denied, open '/var/www/app/package.json'", "block"),
            "    at Object.openSync (node:fs:596:3)",
            ("Access denied by the process sandbox.", "block"),
            _footer(rng, platform, 126),
        )
        return _join(head, body), 126
    body = _lines(
        "compiling application...",
        f"bundled {_num(rng, 10, 900)} modules",
        "waiting for worker pool...",
        ("  0%|          | 0/100 [00:00<?, ?it/s]", "neutral"),
    )
    return _join(head, body), None


def _gen_npm(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    pkg = _p(rng, ["express", "typescript", "eslint", "vite", "react", "lodash"])
    cmd = _p(
        rng,
        [f"npm install {pkg}", "npm install", f"npm run build", f"npm test"],
    )
    head = [_prompt(rng, platform, _cwd(platform), cmd)]
    if status == "success":
        body = _lines(
            f"added {_num(rng, 20, 400)} packages, and audited {_num(rng, 20, 400)} packages in {_num(rng, 1, 20)}s",
            f"{_num(rng, 0, 20)} packages are looking for funding",
            "run `npm fund` for details",
            ("found 0 vulnerabilities", "success"),
        )
        return _join(head, body), 0
    if status == "fail":
        body = _lines(
            ("npm ERR! code ERESOLVE", "fail"),
            "npm ERR! ERESOLVE unable to resolve dependency tree",
            f"npm ERR! Found: react@{_p(rng, _VERSIONS)}",
            f"npm ERR! Could not resolve dependency: peer react@^18 from {pkg}@1.0.0",
            ("npm ERR! A complete log of this run can be found in ~/.npm/_logs/", "fail"),
            _footer(rng, platform, 1),
        )
        return _join(head, body), 1
    if status == "block":
        body = _lines(
            ("npm ERR! code EACCES", "block"),
            "npm ERR! syscall open",
            "npm ERR! path /usr/lib/node_modules/.package-lock.json",
            ("npm ERR! errno -13", "block"),
            ("npm ERR! Error: EACCES: permission denied", "block"),
            _footer(rng, platform, 126),
        )
        return _join(head, body), 126
    body = _lines(
        "npm WARN config global `--global` is deprecated",
        "⸨⠂⠂⠂⠂⠂⠂⠂⠂⸩ ⠧ idealTree:app: timing reifyNode",
        ("", "neutral"),
    )
    return _join(head, body), None


def _gen_cargo(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    cmd = _p(rng, ["cargo build", "cargo test", "cargo check", "cargo run --release"])
    head = [_prompt(rng, platform, _cwd(platform, "engine"), cmd)]
    if status == "success":
        body = _lines(
            f"   Compiling engine v0.1.0 (/home/dev/proj/engine)",
            f"   Compiling serde v{_p(rng, _VERSIONS)}",
            f"    Finished dev [unoptimized + debuginfo] target(s) in {rng.uniform(0.3, 8):.2f}s",
            ("test result: ok. 12 passed; 0 failed; 0 ignored", "success"),
        )
        return _join(head, body), 0
    if status == "fail":
        body = _lines(
            "error[E0382]: borrow of moved value: `data`",
            f"  --> src/{_p(rng, ['main.rs', 'lib.rs', 'worker.rs'])}:{_num(rng, 10, 200)}:{_num(rng, 1, 40)}",
            "   |",
            "   |     consume(data);",
            "   |             ^^^^ value moved here",
            ("error: could not compile `engine` due to previous error", "fail"),
            _footer(rng, platform, 101),
        )
        return _join(head, body), 101
    if status == "block":
        body = _lines(
            f"error: failed to write `target/debug/engine`: Permission denied (os error 13)",
            ("error: could not compile `engine`", "block"),
            ("The build directory is read-only.", "block"),
            _footer(rng, platform, 126),
        )
        return _join(head, body), 126
    body = _lines(
        "   Compiling engine v0.1.0",
        f"    Checking serde v{_p(rng, _VERSIONS)}",
        "    Building [========>            ] 8/20: engine",
    )
    return _join(head, body), None


def _gen_go(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    cmd = _p(rng, ["go test ./...", "go build ./...", "go vet ./...", "go run ."])
    head = [_prompt(rng, platform, _cwd(platform, "service"), cmd)]
    if status == "success":
        body = _lines(
            "ok   example.com/service/api    0.021s",
            "ok   example.com/service/store  0.014s",
            ("PASS", "success"),
        )
        return _join(head, body), 0
    if status == "fail":
        body = _lines(
            "# example.com/service/api",
            f"./handler.go:{_num(rng, 10, 200)}:{_num(rng, 1, 30)}: undefined: {_p(rng, ['parseRequest', 'newStore', 'handle'])}",
            ("FAIL\texample.com/service/api [build failed]", "fail"),
            _footer(rng, platform, 1),
        )
        return _join(head, body), 1
    if status == "block":
        body = _lines(
            ("go: creating work dir: mkdir /root/go: permission denied", "block"),
            ("go: could not create module cache", "block"),
            _footer(rng, platform, 126),
        )
        return _join(head, body), 126
    body = _lines(
        "?    example.com/service/api    [no test files]",
        "=== RUN   TestStore",
        "--- PASS: TestStore (0.00s)",
    )
    return _join(head, body), None


def _gen_maven(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    cmd = _p(rng, ["mvn test", "mvn package -DskipTests", "mvn clean install"])
    head = [_prompt(rng, platform, _cwd(platform, "shop"), cmd)]
    if status == "success":
        body = _lines(
            "[INFO] Scanning for projects...",
            "[INFO] Building shop 1.0.0",
            f"[INFO] Tests run: {_num(rng, 5, 80)}, Failures: 0, Errors: 0, Skipped: 0",
            ("[INFO] BUILD SUCCESS", "success"),
        )
        return _join(head, body), 0
    if status == "fail":
        body = _lines(
            "[INFO] Scanning for projects...",
            ("[ERROR] COMPILATION ERROR :", "fail"),
            f"[ERROR] /src/main/java/com/example/App.java:[{_num(rng, 10, 200)},{_num(rng, 1, 30)}] cannot find symbol",
            ("[INFO] BUILD FAILURE", "fail"),
            _footer(rng, platform, 1),
        )
        return _join(head, body), 1
    if status == "block":
        body = _lines(
            ("[ERROR] Could not create local repository at /root/.m2/repository", "block"),
            ("[ERROR] Permission denied", "block"),
            _footer(rng, platform, 126),
        )
        return _join(head, body), 126
    body = _lines(
        "[INFO] Scanning for projects...",
        "Downloading from central: https://repo.maven.apache.org/maven2/...",
        f"Progress ({_num(rng, 1, 9)}): {rng.uniform(0.1, 9):.1f}/{rng.uniform(10, 90):.1f} MB",
    )
    return _join(head, body), None


def _gen_gradle(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    cmd = _p(rng, ["./gradlew build", "./gradlew test", "./gradlew bootRun"])
    head = [_prompt(rng, platform, _cwd(platform, "payments"), cmd)]
    if status == "success":
        body = _lines(
            "> Task :compileJava",
            "> Task :test",
            ("BUILD SUCCESSFUL in 4s", "success"),
            "3 actionable tasks: 3 executed",
        )
        return _join(head, body), 0
    if status == "fail":
        body = _lines(
            "> Task :test FAILED",
            ("FAILURE: Build failed with an exception.", "fail"),
            "* What went wrong:",
            "Execution failed for task ':test'.",
            "> There were failing tests.",
            _footer(rng, platform, 1),
        )
        return _join(head, body), 1
    if status == "block":
        body = _lines(
            ("Could not write to /home/dev/.gradle/caches: Permission denied", "block"),
            ("> Permission denied", "block"),
            _footer(rng, platform, 126),
        )
        return _join(head, body), 126
    body = _lines(
        "> Task :compileJava",
        "> Task :processResources",
        "> Task :classes",
    )
    return _join(head, body), None


def _gen_dotnet(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    cmd = _p(rng, ["dotnet build", "dotnet test", "dotnet run"])
    head = [_prompt(rng, platform, _cwd(platform, "invoice"), cmd)]
    if status == "success":
        body = _lines(
            "Determining projects to restore...",
            "  All projects are up-to-date for restore.",
            ("Build succeeded.", "success"),
            "    0 Warning(s)",
            "    0 Error(s)",
        )
        return _join(head, body), 0
    if status == "fail":
        body = _lines(
            "Determining projects to restore...",
            ("/src/Program.cs(12,18): error CS1002: ; expected", "fail"),
            ("Build FAILED.", "fail"),
            _footer(rng, platform, 1),
        )
        return _join(head, body), 1
    if status == "block":
        body = _lines(
            ("MSB3021: Unable to copy file. Access is denied.", "block"),
            ("Build FAILED.", "block"),
            _footer(rng, platform, 126),
        )
        return _join(head, body), 126
    body = _lines(
        "Determining projects to restore...",
        "  Restored /src/invoice.csproj (in 212 ms).",
        "  invoice -> /src/bin/Debug/net8.0/invoice.dll",
    )
    return _join(head, body), None


def _gen_ruby(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    cmd = _p(rng, ["bundle exec rspec", "ruby app.rb", "rake test"])
    head = [_prompt(rng, platform, _cwd(platform, "api"), cmd)]
    if status == "success":
        body = _lines(
            "Randomized with seed 41233",
            f"Finished in {rng.uniform(0.1, 4):.2f} seconds",
            ("12 examples, 0 failures", "success"),
        )
        return _join(head, body), 0
    if status == "fail":
        body = _lines(
            "Failures:",
            "  1) API returns a token",
            "     Failure/Error: expect(response.status).to eq(200)",
            ("12 examples, 1 failure", "fail"),
            _footer(rng, platform, 1),
        )
        return _join(head, body), 1
    if status == "block":
        body = _lines(
            ("Permission denied @ rb_sysopen - /var/lib/app/state.json (Errno::EACCES)", "block"),
            _footer(rng, platform, 126),
        )
        return _join(head, body), 126
    body = _lines(
        "Run options: include {:focus=>true}",
        "Randomized with seed 41233",
        "..",
    )
    return _join(head, body), None


def _gen_php(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    cmd = _p(rng, ["php artisan test", "composer install", "php -l app.php"])
    head = [_prompt(rng, platform, _cwd(platform, "portal"), cmd)]
    if status == "success":
        body = _lines(
            "PHPUnit 10.5.0 by Sebastian Bergmann and contributors.",
            "............",
            ("OK (12 tests, 24 assertions)", "success"),
        )
        return _join(head, body), 0
    if status == "fail":
        body = _lines(
            ("PHP Parse error: syntax error, unexpected token \";\" in app.php on line 12", "fail"),
            "Errors parsing app.php",
            _footer(rng, platform, 255),
        )
        return _join(head, body), 255
    if status == "block":
        body = _lines(
            ("file_put_contents(/var/www/storage/logs/laravel.log): Failed to open stream: Permission denied", "block"),
            _footer(rng, platform, 126),
        )
        return _join(head, body), 126
    body = _lines(
        "Composer is operating in a temporary directory.",
        "Loading composer repositories with package information",
        "Updating dependencies",
    )
    return _join(head, body), None


# --------------------------------------------------------------------------
# orchestration and CI
# --------------------------------------------------------------------------
def _gen_kubectl(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    ns = _p(rng, ["default", "prod", "staging"])
    cmd = _p(rng, [f"kubectl get pods -n {ns}", f"kubectl apply -f deployment.yaml", f"kubectl rollout status deploy/app -n {ns}"])
    head = [_prompt(rng, platform, _cwd(platform, "infra"), cmd)]
    if status == "success":
        body = _lines(
            "NAME                     READY   STATUS    RESTARTS   AGE",
            f"app-{_num(rng, 10000, 99999)}-{_p(rng, ['abcde', 'x7k2p', 'q9m4n'])}   1/1     Running   0          2m",
            ("deployment.apps/app configured", "success"),
        )
        return _join(head, body), 0
    if status == "fail":
        body = _lines(
            ("Error from server (NotFound): deployments.apps \"app\" not found", "fail"),
            "The connection to the server 127.0.0.1:6443 was refused.",
            _footer(rng, platform, 1),
        )
        return _join(head, body), 1
    if status == "block":
        body = _lines(
            ("Error from server (Forbidden): pods is forbidden: User \"dev\" cannot list resource \"pods\"", "block"),
            ("RBAC: access denied by cluster policy.", "block"),
            _footer(rng, platform, 1),
        )
        return _join(head, body), 1
    body = _lines(
        "Please enter Username:",
        "Please enter Password:",
    )
    return _join(head, body), None


def _gen_helm(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    release = _p(rng, ["app", "api", "worker"])
    cmd = _p(rng, [f"helm upgrade --install {release} ./chart", f"helm list", f"helm status {release}"])
    head = [_prompt(rng, platform, _cwd(platform, "infra"), cmd)]
    if status == "success":
        body = _lines(
            f"Release \"{release}\" has been upgraded. Happy Helming!",
            "NAME: " + release,
            ("STATUS: deployed", "success"),
            "REVISION: 4",
        )
        return _join(head, body), 0
    if status == "fail":
        body = _lines(
            ("Error: UPGRADE FAILED: release: not found", "fail"),
            ("Error: INSTALLATION FAILED: unable to build kubernetes objects", "fail"),
            _footer(rng, platform, 1),
        )
        return _join(head, body), 1
    if status == "block":
        body = _lines(
            ("Error: query: failed to query with labels: secrets is forbidden", "block"),
            ("Kubernetes cluster unreachable: permission denied", "block"),
            _footer(rng, platform, 1),
        )
        return _join(head, body), 1
    body = _lines(
        "WARNING: Kubernetes configuration file is group-readable.",
        "WARNING: Kubernetes configuration file is world-readable.",
    )
    return _join(head, body), None


def _gen_terraform(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    cmd = _p(rng, ["terraform apply -auto-approve", "terraform plan", "terraform init"])
    head = [_prompt(rng, platform, _cwd(platform, "infra"), cmd)]
    if status == "success":
        body = _lines(
            "aws_instance.web: Creating...",
            "aws_instance.web: Creation complete after 12s",
            ("Apply complete! Resources: 1 added, 0 changed, 0 destroyed.", "success"),
            "Outputs:",
            "public_ip = \"203.0.113.7\"",
        )
        return _join(head, body), 0
    if status == "fail":
        body = _lines(
            ("Error: Invalid provider configuration", "fail"),
            "Provider \"aws\" requires a region.",
            ("Error: Failed to load state: state snapshot was created by a newer Terraform", "fail"),
            _footer(rng, platform, 1),
        )
        return _join(head, body), 1
    if status == "block":
        body = _lines(
            ("Error: Error acquiring the state lock", "block"),
            "Error message: ConditionalCheckFailedException: AccessDenied",
            ("Lock Info: ID: 8f2c... Operation: OperationTypeApply", "block"),
            _footer(rng, platform, 1),
        )
        return _join(head, body), 1
    body = _lines(
        "Terraform will perform the following actions:",
        "  # aws_instance.web will be created",
        "Plan: 1 to add, 0 to change, 0 to destroy.",
    )
    return _join(head, body), None


def _gen_ansible(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    cmd = _p(rng, ["ansible-playbook -i inventory site.yml", "ansible all -m ping", "ansible-playbook deploy.yml --check"])
    head = [_prompt(rng, platform, _cwd(platform, "infra"), cmd)]
    if status == "success":
        body = _lines(
            "PLAY [Configure application] *******************************************",
            "TASK [Gathering Facts] *************************************************",
            "ok: [web-01]",
            "PLAY RECAP *************************************************************",
            ("web-01 : ok=4 changed=1 unreachable=0 failed=0 skipped=0", "success"),
        )
        return _join(head, body), 0
    if status == "fail":
        body = _lines(
            "TASK [Deploy release] **************************************************",
            ("fatal: [web-01]: FAILED! => {\"msg\": \"Destination /opt/app not writable\"}", "fail"),
            "PLAY RECAP *************************************************************",
            ("web-01 : ok=3 changed=0 unreachable=0 failed=1 skipped=0", "fail"),
            _footer(rng, platform, 2),
        )
        return _join(head, body), 2
    if status == "block":
        body = _lines(
            "TASK [Install packages] ************************************************",
            ("fatal: [web-01]: FAILED! => {\"msg\": \"Missing sudo password\"}", "block"),
            ("web-01 : ok=1 changed=0 unreachable=0 failed=1 skipped=0", "block"),
            _footer(rng, platform, 2),
        )
        return _join(head, body), 2
    body = _lines(
        "PLAY [Configure application] *******************************************",
        "TASK [Gathering Facts] *************************************************",
        "ok: [web-01]",
        "TASK [Install packages] ************************************************",
    )
    return _join(head, body), None


def _gen_github_actions(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    cmd = _p(rng, ["act -j test", "gh workflow run ci.yml", "gh run view --log"])
    head = [_prompt(rng, platform, _cwd(platform, "repo"), cmd)]
    if status == "success":
        body = _lines(
            "[CI/test] ⭐ Run Set up job",
            "[CI/test] 🚀 Start image=catthehacker/ubuntu:act-latest",
            "[CI/test] ✅ Success - Main Run tests",
            ("🏁 Job succeeded", "success"),
        )
        return _join(head, body), 0
    if status == "fail":
        body = _lines(
            "[CI/test] ⭐ Run Main Run tests",
            ("[CI/test]   ❌  Failure - Main Run tests", "fail"),
            "[CI/test] 🏁 Job failed",
            _footer(rng, platform, 1),
        )
        return _join(head, body), 1
    if status == "block":
        body = _lines(
            ("Error: Resource not accessible by integration", "block"),
            ("gh: HTTP 403: Resource not accessible by integration", "block"),
            _footer(rng, platform, 1),
        )
        return _join(head, body), 1
    body = _lines(
        "Waiting for a runner to pick up this job...",
        "Job is queued: test",
    )
    return _join(head, body), None


def _gen_gitlab_ci(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    cmd = _p(rng, ["gitlab-runner exec shell test", "gitlab-ci-local test", "glab ci run"])
    head = [_prompt(rng, platform, _cwd(platform, "repo"), cmd)]
    if status == "success":
        body = _lines(
            "Running with gitlab-runner 16.2.0",
            "Preparing the shell executor",
            ("Job succeeded", "success"),
            "Cleaning up project directory and file based variables",
        )
        return _join(head, body), 0
    if status == "fail":
        body = _lines(
            "Running with gitlab-runner 16.2.0",
            ("ERROR: Job failed: exit status 1", "fail"),
            _footer(rng, platform, 1),
        )
        return _join(head, body), 1
    if status == "block":
        body = _lines(
            ("ERROR: Permission denied (publickey).", "block"),
            ("ERROR: Job failed: exit status 1", "block"),
            _footer(rng, platform, 1),
        )
        return _join(head, body), 1
    body = _lines(
        "Running with gitlab-runner 16.2.0",
        "Using shell executor...",
        "Waiting for another job to finish...",
    )
    return _join(head, body), None


def _gen_jenkins(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    cmd = _p(rng, ["jenkins-cli build app", "java -jar jenkins-cli.jar build app", "curl -s http://localhost:8080/job/app/lastBuild/api/json"])
    head = [_prompt(rng, platform, _cwd(platform, "repo"), cmd)]
    if status == "success":
        body = _lines(
            "Started by user dev",
            "[Pipeline] { (Build)",
            "[Pipeline] sh",
            ("Finished: SUCCESS", "success"),
        )
        return _join(head, body), 0
    if status == "fail":
        body = _lines(
            "Started by user dev",
            "[Pipeline] { (Test)",
            ("ERROR: script returned exit code 1", "fail"),
            ("Finished: FAILURE", "fail"),
            _footer(rng, platform, 1),
        )
        return _join(head, body), 1
    if status == "block":
        body = _lines(
            ("ERROR: Authentication required", "block"),
            ("Jenkins: access denied", "block"),
            _footer(rng, platform, 1),
        )
        return _join(head, body), 1
    body = _lines(
        "Started by user dev",
        "[Pipeline] { (Build)",
        "Waiting for next executor...",
    )
    return _join(head, body), None


# --------------------------------------------------------------------------
# data services and web server
# --------------------------------------------------------------------------
def _gen_redis(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    cmd = _p(rng, ["redis-cli ping", "redis-cli INFO", "redis-cli --scan --pattern 'session:*'"])
    head = [_prompt(rng, platform, _cwd(platform), cmd)]
    if status == "success":
        body = _lines(
            "PONG",
            "# Server",
            f"redis_version:{_p(rng, _VERSIONS)}",
            "connected_clients:4",
            f"used_memory_human:{rng.uniform(0.5, 80):.1f}M",
            ("# Keyspace", "success"),
            "db0:keys=12,expires=4,avg_ttl=3600000",
        )
        return _join(head, body), 0
    if status == "fail":
        body = _lines(
            ("Could not connect to Redis at 127.0.0.1:6379: Connection refused", "fail"),
            _footer(rng, platform, 1),
        )
        return _join(head, body), 1
    if status == "block":
        body = _lines(
            ("NOAUTH Authentication required.", "block"),
            ("Redis access denied by requirepass policy.", "block"),
            _footer(rng, platform, 1),
        )
        return _join(head, body), 1
    body = _lines(
        "Waiting for Redis to load the dataset...",
        "Loading RDB produced by version 7.2.4",
    )
    return _join(head, body), None


def _gen_postgres(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    cmd = _p(rng, ["psql -c 'select 1'", "psql -c 'select * from users limit 5'", "pg_isready"])
    head = [_prompt(rng, platform, _cwd(platform), cmd)]
    if status == "success":
        body = _lines(
            " ?column?",
            "----------",
            "        1",
            ("(1 row)", "success"),
            "Time: 1.204 ms",
        )
        return _join(head, body), 0
    if status == "fail":
        body = _lines(
            ("psql: error: connection to server at \"127.0.0.1\", port 5432 failed: Connection refused", "fail"),
            "Is the server running on that host and accepting TCP/IP connections?",
            _footer(rng, platform, 2),
        )
        return _join(head, body), 2
    if status == "block":
        body = _lines(
            ("psql: error: connection to server at \"127.0.0.1\", port 5432 failed: FATAL: password authentication failed for user \"dev\"", "block"),
            ("The database rejected the credentials.", "block"),
            _footer(rng, platform, 2),
        )
        return _join(head, body), 2
    body = _lines(
        "Password for user dev:",
        "psql: hint: use \\password to set a password",
    )
    return _join(head, body), None


def _gen_mysql(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    cmd = _p(rng, ["mysql -e 'select 1'", "mysql -e 'show tables'", "mysqladmin ping"])
    head = [_prompt(rng, platform, _cwd(platform), cmd)]
    if status == "success":
        body = _lines(
            "+---+",
            "| 1 |",
            "+---+",
            "| 1 |",
            "+---+",
            ("1 row in set (0.00 sec)", "success"),
        )
        return _join(head, body), 0
    if status == "fail":
        body = _lines(
            ("ERROR 1045 (28000): Access denied for user 'dev'@'localhost' (using password: YES)", "fail"),
            _footer(rng, platform, 1),
        )
        return _join(head, body), 1
    if status == "block":
        body = _lines(
            ("ERROR 1044 (42000): Access denied for user 'dev'@'%' to database 'app'", "block"),
            ("Database policy denied the operation.", "block"),
            _footer(rng, platform, 1),
        )
        return _join(head, body), 1
    body = _lines(
        "Enter password:",
        "mysql: [Warning] Using a password on the command line interface can be insecure.",
    )
    return _join(head, body), None


def _gen_nginx(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    cmd = _p(rng, ["nginx -t", "nginx -s reload", "systemctl reload nginx"])
    head = [_prompt(rng, platform, _cwd(platform), cmd)]
    if status == "success":
        body = _lines(
            "nginx: the configuration file /etc/nginx/nginx.conf syntax is ok",
            ("nginx: configuration file /etc/nginx/nginx.conf test is successful", "success"),
        )
        return _join(head, body), 0
    if status == "fail":
        body = _lines(
            ("nginx: [emerg] unknown directive \"serve\" in /etc/nginx/nginx.conf:12", "fail"),
            ("nginx: configuration file /etc/nginx/nginx.conf test failed", "fail"),
            _footer(rng, platform, 1),
        )
        return _join(head, body), 1
    if status == "block":
        body = _lines(
            ("nginx: [emerg] open() \"/var/log/nginx/error.log\" failed (13: Permission denied)", "block"),
            ("nginx: configuration file /etc/nginx/nginx.conf test failed", "block"),
            _footer(rng, platform, 1),
        )
        return _join(head, body), 1
    body = _lines(
        "nginx: the configuration file /etc/nginx/nginx.conf syntax is ok",
        "nginx: [warn] conflicting server name \"app\" on 0.0.0.0:80",
    )
    return _join(head, body), None


MODERN_GENERATORS = {
    "node": _gen_node,
    "npm": _gen_npm,
    "cargo": _gen_cargo,
    "go": _gen_go,
    "maven": _gen_maven,
    "gradle": _gen_gradle,
    "dotnet": _gen_dotnet,
    "ruby": _gen_ruby,
    "php": _gen_php,
    "kubectl": _gen_kubectl,
    "helm": _gen_helm,
    "terraform": _gen_terraform,
    "ansible": _gen_ansible,
    "github_actions": _gen_github_actions,
    "gitlab_ci": _gen_gitlab_ci,
    "jenkins": _gen_jenkins,
    "redis": _gen_redis,
    "postgres": _gen_postgres,
    "mysql": _gen_mysql,
    "nginx": _gen_nginx,
}
