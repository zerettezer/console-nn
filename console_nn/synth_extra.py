"""Extra domain generators (package updates, compilation, processing, ...).

Kept in a separate module to keep ``synth.py`` readable.  Every generator has
the same contract: ``(rng, platform, status, cfg) -> (lines, exit_code)`` where
``lines`` is a list of ``(text, label)`` pairs and ``label`` is one of
``success`` / ``fail`` / ``block`` / ``neutral``.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

from .synth import (
    BRANCHES,
    COLORS,
    DIRS,
    DOMAINS,
    FILES,
    FUNCS,
    IPS,
    MODULES,
    PACKAGES,
    PROJECTS,
    SERVICES,
    USERS,
    VERSIONS,
    Line,
    _footer,
    _host,
    _join,
    _maybe,
    _maybe_blank,
    _num,
    _p,
    _prompt,
    _stamp,
)

_WEB = ["nginx", "postgres", "redis", "app", "worker", "api", "frontend"]


# --------------------------------------------------------------------------
# package updates
# --------------------------------------------------------------------------
def _gen_package_update(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    if platform == "windows":
        tool = _p(rng, ["winget", "choco", "pip", "powershell"])
    else:
        tool = _p(rng, ["apt", "apt", "dnf", "pacman", "pip", "conda", "snap", "flatpak"])
    pkg = _p(rng, PACKAGES)
    version = _p(rng, VERSIONS)
    cmd = {
        "winget": "winget upgrade --all",
        "choco": "choco upgrade all -y",
        "pip": f"python -m pip install --upgrade {pkg}",
        "powershell": "Update-Module -Force",
        "apt": "sudo apt update && sudo apt upgrade -y",
        "dnf": "sudo dnf upgrade --refresh -y",
        "pacman": "sudo pacman -Syu --noconfirm",
        "conda": "conda update --all -y",
        "snap": "sudo snap refresh",
        "flatpak": "flatpak update -y",
    }[tool]
    head = [_prompt(rng, platform, "~" if platform == "linux" else "C:\\proj", cmd)]
    success_body = {
        "apt": [
            "Hit:1 http://deb.debian.org/debian bookworm InRelease",
            "Get:2 http://security.debian.org/debian-security bookworm-security InRelease [48.0 kB]",
            "Reading package lists... Done",
            "Building dependency tree... Done",
            "Calculating upgrade... Done",
            "The following packages will be upgraded:",
            f"  {pkg} {version}",
            f"Preparing to unpack .../{pkg}_{version}_amd64.deb ...",
            f"Setting up {pkg} ({version}) ...",
            "Processing triggers for man-db ...",
            "0 upgraded, 0 newly installed, 0 to remove and 0 not upgraded.",
        ],
        "dnf": [
            "Last metadata expiration check: 0:12:41 ago.",
            "Dependencies resolved.",
            "====================================================================",
            f" Package        Arch     Version          Repository        Size",
            "====================================================================",
            f" {pkg}          x86_64   {version}        updates           {_num(rng, 100, 9000)} k",
            "Transaction Summary",
            f"Upgrade  {_num(rng, 1, 40)} Packages",
            "Downloading Packages:",
            f"[{_num(rng, 1, 40)}/40] {pkg}-{version}.x86_64.rpm   {_num(rng, 1, 40)} MB/s | "
            f"{_num(rng, 1, 40)} MB  00:0{_num(rng, 1, 9)}",
            "Complete!",
        ],
        "pacman": [
            ":: Synchronising package databases...",
            " core is up to date",
            " extra is up to date",
            ":: Starting full system upgrade...",
            f"resolving dependencies...",
            f"looking for conflicting packages...",
            f"Packages ({_num(rng, 1, 30)}): {pkg}-{version}",
            f"Total Download Size:   {rng.uniform(0.1, 90):.2f} MiB",
            f"Total Installed Size:  {rng.uniform(0.1, 300):.2f} MiB",
            ":: Proceed with installation? [Y/n] y",
            f"(1/1) upgrading {pkg}",
        ],
        "pip": [
            f"Requirement already satisfied: {pkg} in /usr/lib/python3.10/site-packages",
            f"Collecting {pkg}>={version}",
            f"  Downloading {pkg}-{version}-py3-none-any.whl ({rng.uniform(0.1, 20):.1f} MB)",
            f"Installing collected packages: {pkg}",
            f"  Attempting uninstall: {pkg}",
            f"    Found existing installation: {pkg} {version}",
            f"    Uninstalling {pkg}-{version}:",
            f"      Successfully uninstalled {pkg}-{version}",
            f"Successfully installed {pkg}-{version}",
        ],
        "conda": [
            "Collecting package metadata (current_repodata.json): done",
            "Solving environment: done",
            "## Package Plan ##",
            f"  environment location: /home/{_p(rng, USERS)}/miniconda3",
            f"  updated specs: {pkg}",
            "The following packages will be UPDATED:",
            f"  {pkg}  {version} --> {_p(rng, VERSIONS)}",
            "Preparing transaction: done",
            "Verifying transaction: done",
            "Executing transaction: done",
        ],
        "snap": [
            "Refresh all snaps: no updates in progress",
            f"snap \"{pkg}\" has no updates available",
            "All snaps up to date.",
        ],
        "flatpak": [
            "Looking for updates…",
            "Nothing to do.",
            "All flatpaks are up to date.",
        ],
        "winget": [
            f"Found {pkg} [{'Python.Python' if 'python' in pkg.lower() else pkg}]",
            "Downloading https://github.com/.../installer.exe",
            f"  ████████████████████████  {rng.uniform(10, 200):.1f} MB / "
            f"{rng.uniform(10, 200):.1f} MB",
            "Successfully verified installer hash",
            "Starting package install...",
            "Successfully installed",
        ],
        "choco": [
            "Chocolatey v6.2.1",
            "Upgrading the following packages:",
            f"  {pkg}",
            f"Upgrading {pkg} from {version} to {_p(rng, VERSIONS)}",
            "Downloading package from source 'https://community.chocolatey.org/api/v2/'",
            "Installing package...",
            f"The upgrade of {pkg} was successful.",
            "The operation completed successfully.",
        ],
        "powershell": [
            "Updating PowerShell modules from PSGallery",
            "PSReadLine 2.3.4 downloaded",
            "Successfully installed PSReadLine",
            "Module update complete.",
        ],
    }[tool]
    if status == "success":
        lines = _join(
            head,
            [(x, "neutral") for x in success_body[:-2]],
            [(success_body[-2], "success"), (success_body[-1], "success")],
            [_footer(rng, platform, 0)],
        )
        return lines, 0
    if status == "fail":
        errs = _p(rng, [
            [(f"E: Failed to fetch http://deb.debian.org/debian/pool/main/{pkg}.deb  "
              f"404  Not Found [IP: 151.101.2.132 80]", "fail"),
             ("E: Some index files failed to download. They have been ignored, or old ones used instead.", "fail")],
            [(f"E: Unable to locate package {pkg}", "fail")],
            [("Error: Failed to download metadata for repo 'updates'", "fail"),
             ("Error: GPG check FAILED", "fail")],
            [("error: failed retrieving file '" + pkg + "-" + version + "' from mirror", "fail"),
             ("error: target not found: " + pkg, "fail")],
            [(f"ERROR: Could not find a version that satisfies the requirement {pkg}>={version}", "fail"),
             (f"ERROR: No matching distribution found for {pkg}>={version}", "fail")],
            [("CondaHTTPError: HTTP 000 CONNECTION FAILED", "fail"),
             ("An HTTP error occurred when trying to retrieve this URL.", "fail")],
            [("Could not find a part of the path 'C:\\ProgramData\\Package Cache'.", "fail")],
            [("Installer failed with exit code: 1603", "fail"),
             ("The installation failed. Fatal error during installation.", "fail")],
            [("The upgrade of " + pkg + " failed.", "fail"),
             ("The operation did not complete successfully.", "fail")],
        ])
        return _join(head, errs, [_footer(rng, platform, 1)]), 1
    if status == "block":
        body = _p(rng, [
            [("E: Could not open lock file /var/lib/dpkg/lock-frontend - open (13: Permission denied)", "block"),
             ("E: Unable to acquire the dpkg frontend lock, are you root?", "block")],
            [("The requested operation requires elevation.", "block"),
             ("Access is denied.", "block")],
            [("ERROR: Could not install packages due to an OSError: [Errno 13] Permission denied: "
              "'/usr/lib/python3/dist-packages'", "block")],
            [("error: snap \"" + pkg + "\" has \"refresh\" change in progress, but it is blocked by policy", "block")],
            [("Installing package requires administrator privileges.", "block"),
             ("The installer was blocked by Windows Defender SmartScreen.", "block")],
            [("error: unable to lock database", "block")],
        ])
        return _join(head, body, [_footer(rng, platform, 126)]), 126
    body = _fillers_from(rng, _num(rng, 2, 8), [
        "Reading package lists... Done",
        "Building dependency tree... Done",
        "Calculating upgrade...",
        "Downloading packages...",
        f"[{_num(rng, 1, 20)}/{_num(rng, 20, 80)}] fetching {pkg}",
        "Transaction progress 42%",
    ])
    return _join(head, body), None


# --------------------------------------------------------------------------
# compilation / builds
# --------------------------------------------------------------------------
def _gen_compile(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    if platform == "windows":
        tool = _p(rng, ["msbuild", "cl", "dotnet", "tsc", "cmake"])
    else:
        tool = _p(rng, ["gcc", "g++", "clang", "make", "cmake", "ninja", "cargo",
                        "go", "mvn", "gradle"])
    target = _p(rng, PROJECTS)
    src = _p(rng, ["main.c", "src/main.cpp", "src/lib.rs", "cmd/server/main.go"])
    cmd = {
        "gcc": f"gcc -O2 -Wall -Wextra {src} -o build/{target}",
        "g++": f"g++ -std=c++17 -O2 {src} -o build/{target}",
        "clang": f"clang -O3 -Wall {src} -o build/{target}",
        "make": "make -j8",
        "cmake": "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8",
        "ninja": "ninja -C build -j 8",
        "cargo": "cargo build --release",
        "go": "go build -o bin/app ./...",
        "mvn": "mvn -q clean package",
        "gradle": "./gradlew build --no-daemon",
        "msbuild": f"msbuild {target}.sln /p:Configuration=Release /m",
        "cl": f"cl /O2 /W4 {src} /Fe:build\\{target}.exe",
        "dotnet": "dotnet build -c Release",
        "tsc": "npx tsc --build",
    }[tool]
    head = [_prompt(rng, platform, "~/proj" if platform == "linux" else "C:\\proj", cmd)]

    if status == "success":
        body = {
            "gcc": [f"{src}: In function 'main':",
                    f"{src}:{_num(rng, 10, 300)}:{_num(rng, 1, 20)}: warning: unused variable "
                    f"'tmp' [-Wunused-variable]",
                    f"{_num(rng, 1, 40)} |     int tmp = 0;",
                    "      |         ^~~"],
            "g++": [f"[ 25%] Building CXX object CMakeFiles/app.dir/{src}.o",
                    f"[ 50%] Building CXX object CMakeFiles/app.dir/src/parser.cpp.o",
                    f"[ 75%] Building CXX object CMakeFiles/app.dir/src/io.cpp.o",
                    f"[100%] Linking CXX executable {target}"],
            "clang": [f"{src}:{_num(rng, 10, 300)}:{_num(rng, 1, 20)}: warning: variable 'rc' "
                      "set but not used [-Wunused-but-set-variable]"],
            "make": [f"[ 10%] Building C object CMakeFiles/app.dir/{src}.o",
                     f"[ 20%] Building C object CMakeFiles/app.dir/src/io.c.o",
                     f"[ 40%] Linking C executable {target}",
                     f"[100%] Built target {target}"],
            "cmake": ["-- The C compiler identification is GNU 12.3.0",
                      "-- Check for working C compiler: /usr/bin/cc - works",
                      "-- Configuring done",
                      "-- Generating done",
                      "-- Build files have been written to: /home/dev/proj/build",
                      f"[ 50%] Building C object CMakeFiles/{target}.dir/{src}.o",
                      f"[100%] Linking C executable {target}",
                      f"[100%] Built target {target}"],
            "ninja": [f"[1/{_num(rng, 5, 40)}] Building CXX object CMakeFiles/app.dir/{src}.o",
                      f"[{_num(rng, 5, 40)}/{_num(rng, 20, 60)}] Linking CXX executable {target}",
                      "ninja: build stopped: no work to do."],
            "cargo": ["   Compiling proc-macro2 v1.0.86",
                      "   Compiling quote v1.0.36",
                      "   Compiling syn v2.0.68",
                      f"   Compiling {target} v0.1.0 (/home/dev/proj)",
                      f"    Finished release [optimized] target(s) in {rng.uniform(1, 90):.2f}s"],
            "go": ["go: downloading github.com/pkg/errors v0.9.1",
                   "go: downloading golang.org/x/sync v0.7.0",
                   f"go: building {target}",
                   f"built ./cmd/server -> bin/app"],
            "mvn": ["[INFO] Scanning for projects...",
                    "[INFO] Building " + target + " 1.0.0",
                    "[INFO] --- maven-compiler-plugin:3.11.0:compile (default-compile) @ app ---",
                    "[INFO] Compiling 42 source files to /home/dev/proj/target/classes",
                    "[INFO] BUILD SUCCESS",
                    f"[INFO] Total time:  {rng.uniform(1, 60):.2f} s"],
            "gradle": ["> Task :compileJava", "> Task :processResources", "> Task :classes",
                       "> Task :jar", "> Task :build",
                       "BUILD SUCCESSFUL in " + str(_num(rng, 2, 60)) + "s",
                       "3 actionable tasks: 3 executed"],
            "msbuild": ["Microsoft (R) Build Engine version 17.9.8",
                        "Build started 11/6/2026 10:14:22 AM.",
                        f"  {target} -> C:\\proj\\bin\\Release\\{target}.exe",
                        "    0 Warning(s)", "    0 Error(s)"],
            "cl": [f"{src}", "Microsoft (R) C/C++ Optimizing Compiler Version 19.39.33523",
                   f"/out:build\\{target}.exe", "Generating code", "Finished generating code"],
            "dotnet": ["  Determining projects to restore...",
                       "  Restored C:\\proj\\app.csproj (in 412 ms).",
                       f"  {target} -> C:\\proj\\bin\\Release\\net8.0\\{target}.dll",
                       "Build succeeded.", "    0 Warning(s)", "    0 Error(s)"],
            "tsc": [f"src/index.ts -> dist/index.js", f"src/server.ts -> dist/server.js",
                    "Compilation complete."],
        }[tool]
        lines = _join(
            head,
            [(x, "neutral") for x in body],
            [("Build succeeded.", "success"), _footer(rng, platform, 0)],
        )
        return lines, 0

    if status == "fail":
        errs = _p(rng, [
            [(f"{src}:{_num(rng, 10, 300)}:{_num(rng, 1, 40)}: error: expected ';' before '}}' token", "fail"),
             (f"{src}:{_num(rng, 10, 300)}:{_num(rng, 1, 40)}: error: 'NULL' undeclared "
              f"(first use in this function)", "fail"),
             ("make: *** [Makefile:24: all] Error 1", "fail")],
            [("undefined reference to `" + _p(rng, FUNCS) + "'", "fail"),
             ("collect2: error: ld returned 1 exit status", "fail")],
            [("error[E0308]: mismatched types", "fail"),
             ("  --> src/main.rs:" + str(_num(rng, 10, 300)) + ":18:5", "fail"),
             ("error: could not compile `" + target + "` due to 2 previous errors", "fail")],
            [("./server.go:" + str(_num(rng, 10, 300)) + ": undefined: " + _p(rng, FUNCS), "fail"),
             ("make: *** [Makefile:12: build] Error 1", "fail")],
            [("[ERROR] COMPILATION ERROR :", "fail"),
             ("[ERROR] /home/dev/proj/src/Main.java:[42,18] cannot find symbol", "fail"),
             ("[INFO] BUILD FAILURE", "fail")],
            [("FAILURE: Build failed with an exception.", "fail"),
             ("* What went wrong:", "fail"),
             ("Execution failed for task ':compileJava'.", "fail")],
            [("CMake Error at CMakeLists.txt:12 (find_package):", "fail"),
             ("  Could not find a package configuration file provided by \"OpenCV\"", "fail")],
            [("cl : Command line error D8021 : invalid numeric argument '/Werror'", "fail")],
            [("error CS0246: The type or namespace name 'App' could not be found", "fail"),
             ("Build FAILED.", "fail")],
            [("error TS2307: Cannot find module './types' or its corresponding type declarations.", "fail")],
        ])
        return _join(head, errs, [_footer(rng, platform, 1)]), 1

    if status == "block":
        body = _p(rng, [
            [("/usr/bin/ld: cannot open output file " + _p(rng, DIRS) + "/app: Permission denied", "block"),
             ("collect2: error: ld returned 1 exit status", "fail")],
            [("CMake Error: could not create directory /opt/build: Permission denied", "block")],
            [("error: failed to write to /usr/local/lib/libapp.so: Permission denied", "block")],
            [("C:\\proj\\bin\\Release\\app.exe : error: Access is denied.", "block")],
        ])
        return _join(head, body, [_footer(rng, platform, 126)]), 126

    body = _fillers_from(rng, _num(rng, 2, 8), [
        f"[{_num(rng, 5, 60)}%] Building C object CMakeFiles/app.dir/{src}.o",
        "[ 42%] Building CXX object CMakeFiles/app.dir/src/parser.cpp.o",
        "   Compiling serde v1.0.203",
        "   Compiling tokio v1.38.0",
        "go: downloading golang.org/x/net v0.26.0",
        "Downloading https://repo.maven.apache.org/maven2/...",
    ])
    return _join(head, body), None


# --------------------------------------------------------------------------
# media / processing
# --------------------------------------------------------------------------
def _gen_media(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    tool = _p(rng, ["ffmpeg", "convert", "sox"]) if platform == "linux" else _p(rng, ["ffmpeg", "magick"])
    infile = _p(rng, ["input.mp4", "recording.mkv", "frame.png", "audio.wav", "clip.mov"])
    outfile = _p(rng, ["output.mp4", "out.webm", "frame.jpg", "audio.ogg", "clip_720p.mp4"])
    cmd = {
        "ffmpeg": f"ffmpeg -i {infile} -c:v libx264 -crf 23 -preset medium {outfile}",
        "convert": f"convert {infile} -resize 50% -quality 85 {outfile}",
        "magick": f"magick {infile} -resize 50% -quality 85 {outfile}",
        "sox": f"sox {infile} -r 44100 -c 2 {outfile}",
    }[tool]
    head = [_prompt(rng, platform, "~/media" if platform == "linux" else "C:\\media", cmd)]
    if status == "success":
        if tool == "ffmpeg":
            body = [
                "ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2024 the FFmpeg developers",
                "  built with gcc 13 (Ubuntu 13.2.0-23ubuntu4)",
                f"Input #0, mov,mp4,m4a,3gp,3g2,mj2, from '{infile}':",
                f"  Duration: 00:{_num(rng, 1, 59):02d}:{_num(rng, 1, 59):02d}.{_num(rng, 10, 99)}, "
                f"start: 0.000000, bitrate: {_num(rng, 500, 9000)} kb/s",
                f"  Stream #0:0: Video: h264 (High), yuv420p, {_p(rng, ['1280x720', '1920x1080', '720x480'])}, "
                f"{_num(rng, 24, 60)} fps",
                f"  Stream #0:1: Audio: aac (LC), 48000 Hz, stereo, fltp, {_num(rng, 90, 320)} kb/s",
                "Stream mapping:",
                "  Stream #0:0 -> #0:0 (h264 (native) -> h264 (libx264))",
                "  Stream #0:1 -> #0:1 (aac (native) -> aac (native))",
                f"frame={_num(rng, 100, 9000)} fps={_num(rng, 20, 120)} q={_num(rng, 18, 30)}.0 "
                f"size={_num(rng, 1, 900)}kB time=00:00:{_num(rng, 10, 59):02d}.00 "
                f"bitrate={_num(rng, 100, 9000)}kbits/s speed={rng.uniform(0.5, 8):.2f}x",
                f"video:{_num(rng, 100, 90000)}kB audio:{_num(rng, 10, 9000)}kB "
                f"subtitle:0kB other streams:0kB global headers:0kB muxing overhead: "
                f"{rng.uniform(0.01, 2):.4f}%",
            ]
        elif tool == "sox":
            body = [
                f"sox WARN formats: can't set sample rate 44100; using 48000",
                f"Input File     : '{infile}'",
                f"Channels       : 1",
                f"Sample Rate    : 48000",
                f"Precision      : 16-bit",
                f"Duration       : 00:00:{_num(rng, 10, 59):02d}.00",
                f"Output File    : '{outfile}'",
                "Processing audio...",
            ]
        else:
            body = [
                f"{infile} PNG {_p(rng, ['1280x720', '1920x1080'])} "
                f"{_num(rng, 8, 32)}-bit sRGB {rng.uniform(0.1, 9):.2f}MB",
                f"{infile} => {outfile}",
                f"  resize: 50% -> {_p(rng, ['640x360', '960x540'])}",
                "  quality: 85",
                "  writing output ...",
            ]
        return _join(head, [(x, "neutral") for x in body],
                     [("Processing completed successfully.", "success"),
                      _footer(rng, platform, 0)]), 0
    if status == "fail":
        errs = _p(rng, [
            [(f"{infile}: No such file or directory", "fail")],
            [(f"{infile}: Invalid data found when processing input", "fail"),
             ("Conversion failed!", "fail")],
            [("Error initializing output stream 0:0 -- Error while opening encoder", "fail"),
             ("Conversion failed!", "fail")],
            [("Unknown encoder 'libx265'", "fail")],
            [("convert-im6.q16: unable to open image '" + infile + "': No such file or directory", "fail")],
            [("sox FAIL formats: can't open input file `" + infile + "': No such file or directory", "fail")],
            [("Output file #0 does not contain any stream", "fail")],
        ])
        return _join(head, errs, [_footer(rng, platform, 1)]), 1
    if status == "block":
        body = _p(rng, [
            [(f"Could not open file : {outfile}", "block"),
             ("Permission denied", "block")],
            [("Error opening output file " + outfile + ": Permission denied", "block")],
            [("magick: unable to open image '" + outfile + "': Permission denied @ error/blob.c/OpenBlob/3596.", "block")],
            [("Access is denied.", "block")],
        ])
        return _join(head, body, [_footer(rng, platform, 126)]), 126
    body = _fillers_from(rng, _num(rng, 2, 7), [
        "ffmpeg version 6.1.1 Copyright (c) 2000-2024 the FFmpeg developers",
        f"Input #0, mov,mp4,m4a,3gp,3g2,mj2, from '{infile}':",
        f"  Duration: 00:0{_num(rng, 1, 9)}:{_num(rng, 10, 59)}.00, bitrate: {_num(rng, 500, 9000)} kb/s",
        "Stream mapping:",
        f"frame={_num(rng, 100, 9000)} fps={_num(rng, 20, 120)} size={_num(rng, 1, 900)}kB "
        f"time=00:00:{_num(rng, 10, 59):02d}.00 bitrate={_num(rng, 100, 9000)}kbits/s",
    ])
    return _join(head, body), None


# --------------------------------------------------------------------------
# data processing
# --------------------------------------------------------------------------
def _gen_data_process(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    tool = _p(rng, ["python", "jq", "awk", "duckdb", "mlr"])
    infile = _p(rng, ["sales.csv", "events.json", "users.parquet", "access.log", "metrics.csv"])
    outfile = _p(rng, ["out.parquet", "report.json", "summary.csv", "clean.parquet"])
    cmd = {
        "python": f"python process.py --input {infile} --output {outfile}",
        "jq": f"jq '.items[] | select(.active == true) | {{id, name}}' {infile}",
        "awk": f"awk -F, 'NR>1 {{sum+=$3; n++}} END {{print sum/n}}' {infile}",
        "duckdb": f"duckdb -c \"COPY (SELECT * FROM '{infile}') TO '{outfile}' (FORMAT PARQUET)\"",
        "mlr": f"mlr --csv cut -f id,name,value {infile} > {outfile}",
    }[tool]
    head = [_prompt(rng, platform, "~/data" if platform == "linux" else "C:\\data", cmd)]
    if status == "success":
        body = {
            "python": [
                f"[INFO] loading {infile}",
                f"[INFO] {_num(rng, 1000, 900000)} rows x {_num(rng, 4, 40)} columns",
                "[INFO] dtype inference complete",
                f"[INFO] stage 1/4: cleaning ({_num(rng, 1, 9)}.{_num(rng, 10, 99)}s)",
                f"[INFO] stage 2/4: normalising",
                f"[INFO] stage 3/4: joining lookup tables",
                f"[INFO] stage 4/4: writing {outfile}",
                f"[INFO] wrote {outfile} ({rng.uniform(0.1, 900):.1f} MB)",
                "Done.",
            ],
            "jq": [
                "{",
                '  "id": 101,',
                '  "name": "alpha"',
                "}",
                "{",
                '  "id": 204,',
                '  "name": "beta"',
                "}",
                f"{_num(rng, 2, 900)} records emitted.",
            ],
            "awk": [f"{rng.uniform(1, 9000):.4f}", f"{_num(rng, 1000, 99999)} records processed."],
            "duckdb": [
                "┌─────────┬──────────┬────────┐",
                "│ id      │ name     │ value  │",
                "├─────────┼──────────┼────────┤",
                f"│ {_num(rng, 1, 999):7d} │ alpha    │ {rng.uniform(0, 9):6.2f} │",
                f"│ {_num(rng, 1, 999):7d} │ beta     │ {rng.uniform(0, 9):6.2f} │",
                "└─────────┴──────────┴────────┘",
                f"Copied {_num(rng, 100, 90000)} rows to {outfile}.",
            ],
            "mlr": [
                "id,name,value",
                f"1,alpha,{rng.uniform(0, 9):.2f}",
                f"2,beta,{rng.uniform(0, 9):.2f}",
                f"{_num(rng, 2, 900)} records written.",
            ],
        }[tool]
        return _join(head, [(x, "neutral") for x in body[:-1]],
                     [(body[-1], "success"), _footer(rng, platform, 0)]), 0
    if status == "fail":
        errs = _p(rng, [
            [(f"KeyError: 'price'", "fail"),
             ('  File "process.py", line ' + str(_num(rng, 10, 300)) + ", in transform", "fail"),
             ("    df['total'] = df['price'] * df['qty']", "fail")],
            [("pandas.errors.ParserError: Error tokenizing data. C error: Expected "
              f"{_num(rng, 3, 20)} fields in line {_num(rng, 10, 9000)}, saw {_num(rng, 21, 40)}", "fail")],
            [("jq: error (at " + infile + ":" + str(_num(rng, 1, 900)) + "): Cannot index string with \"items\"", "fail")],
            [("awk: cmd. line:1: fatal: cannot open file `" + infile + "' for reading (No such file or directory)", "fail")],
            [("MemoryError: Unable to allocate " + str(_num(rng, 1, 64)) + ".0 GiB for an array", "fail")],
            [("IOException: No space left on device", "fail")],
            [("duckdb.duckdb.IOException: Could not write to file \"" + outfile + "\"", "fail")],
            [("mlr: CSV parse error at line " + str(_num(rng, 2, 9000)) + ": expected "
              f"{_num(rng, 3, 20)} fields, got {_num(rng, 1, 30)}", "fail")],
        ])
        return _join(head, errs, [_footer(rng, platform, 1)]), 1
    if status == "block":
        body = _p(rng, [
            [("PermissionError: [Errno 13] Permission denied: '" + infile + "'", "block")],
            [("jq: error: Could not open " + infile + ": Permission denied", "block")],
            [("awk: cmd. line:1: fatal: cannot open file `" + infile + "' for reading (Permission denied)", "block")],
            [("ERROR: could not write " + outfile + ": Access is denied.", "block")],
        ])
        return _join(head, body, [_footer(rng, platform, 126)]), 126
    body = _fillers_from(rng, _num(rng, 2, 7), [
        f"[INFO] loading {infile}",
        f"[INFO] {_num(rng, 1000, 900000)} rows x {_num(rng, 4, 40)} columns",
        f"[INFO] stage {_num(rng, 1, 3)}/4: processing",
        f"processed {_num(rng, 1000, 900000)} records",
        f"elapsed {rng.uniform(0.1, 90):.2f}s",
    ])
    return _join(head, body), None


# --------------------------------------------------------------------------
# archives / backups
# --------------------------------------------------------------------------
def _gen_archive(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    tool = _p(rng, ["tar", "zip", "unzip", "7z"]) if platform == "linux" else _p(rng, ["tar", "7z", "Expand-Archive"])
    src = _p(rng, ["build", "logs", "dataset", "backup"])
    out = _p(rng, ["backup.tar.gz", "logs.zip", "bundle.7z", "release.tar.zst"])
    cmd = {
        "tar": f"tar -czf {out} {src}/",
        "zip": f"zip -r {out} {src}/",
        "unzip": f"unzip -o {out} -d {src}/",
        "7z": f"7z a -mx=9 {out} {src}",
        "Expand-Archive": f"Expand-Archive -Path {out} -DestinationPath {src}",
    }[tool]
    head = [_prompt(rng, platform, "~/proj" if platform == "linux" else "C:\\proj", cmd)]
    if status == "success":
        if tool == "tar":
            body = [
                f"{src}/",
                f"{src}/main.py",
                f"{src}/config.yaml",
                f"{src}/logs/app.log",
                f"{src}/data/input.csv",
                f"Compressed {_num(rng, 100, 9000)} files, {rng.uniform(0.1, 900):.1f} MB -> "
                f"{rng.uniform(0.1, 300):.1f} MB",
            ]
        elif tool == "zip":
            body = [
                f"  adding: {src}/ (stored 0%)",
                f"  adding: {src}/main.py (deflated {_num(rng, 10, 90)}%)",
                f"  adding: {src}/config.yaml (deflated {_num(rng, 10, 90)}%)",
                f"  adding: {src}/logs/app.log (deflated {_num(rng, 10, 90)}%)",
                f"total bytes={_num(rng, 1000, 900000)}",
            ]
        elif tool == "unzip":
            body = [
                f"Archive:  {out}",
                f"  inflating: {src}/main.py",
                f"  inflating: {src}/config.yaml",
                f"{_num(rng, 2, 900)} files extracted.",
            ]
        elif tool == "7z":
            body = [
                "7-Zip 23.01 (x64) : Copyright (c) 1999-2023 Igor Pavlov",
                "Scanning the drive:",
                f"{_num(rng, 10, 9000)} files, {_num(rng, 100, 900000)} bytes",
                "Creating archive: " + out,
                f"Compressing: {src}/main.py",
                f"Compressing: {src}/config.yaml",
                "Everything is Ok",
                f"Size: {_num(rng, 100, 90000)}",
                f"Compressed: {_num(rng, 10, 9000)}",
            ]
        else:
            body = [
                f"Extracting {out} to {src}",
                f"  {src}/main.py",
                f"  {src}/config.yaml",
                "Archive expanded successfully.",
            ]
        return _join(head, [(x, "neutral") for x in body],
                     [("Archive operation completed successfully.", "success"),
                      _footer(rng, platform, 0)]), 0
    if status == "fail":
        errs = _p(rng, [
            [(f"tar: {src}: Cannot stat: No such file or directory", "fail"),
             ("tar: Exiting with failure status due to previous errors", "fail")],
            [("zip warning: name not matched: " + src, "fail"),
             ("zip error: Nothing to do! (" + out + ")", "fail")],
            [("unzip:  cannot find or open " + out + ", " + out + ".zip or " + out + ".ZIP.", "fail")],
            [("ERROR: " + out + " : CRC Failed", "fail"),
             ("Sub items Errors: 3", "fail")],
            [("7z: command not found", "fail")],
            [("Expand-Archive : The path '" + out + "' either does not exist or is not a valid file system path.", "fail")],
            [("gzip: stdin: unexpected end of file", "fail"),
             ("tar: Unexpected EOF in archive", "fail")],
        ])
        return _join(head, errs, [_footer(rng, platform, 1)]), 1
    if status == "block":
        body = _p(rng, [
            [(f"tar: {out}: Cannot open: Permission denied", "block")],
            [(f"zip I/O error: Permission denied", "block"),
             (f"zip error: Output file creation failed", "block")],
            [("Expand-Archive : Access to the path '" + src + "' is denied.", "block")],
            [("7z: ERROR: Can not open output file " + out + " : Permission denied", "block")],
        ])
        return _join(head, body, [_footer(rng, platform, 126)]), 126
    body = _fillers_from(rng, _num(rng, 2, 6), [
        f"  adding: {src}/main.py",
        f"  adding: {src}/config.yaml",
        f"Compressing {_num(rng, 10, 9000)} files...",
        f"progress: {_num(rng, 10, 99)}%",
    ])
    return _join(head, body), None


def _gen_backup(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    src = _p(rng, DIRS)
    dst = _p(rng, ["backup-server:/backups", "D:\\backup", "s3://bucket/backups", "/mnt/backup"])
    tool = _p(rng, ["rsync", "rclone", "robocopy"]) if platform == "linux" else _p(rng, ["robocopy", "rclone"])
    cmd = {
        "rsync": f"rsync -avz --delete {src}/ {dst}/",
        "rclone": f"rclone sync {src} {dst} --progress",
        "robocopy": f"robocopy {src} {dst} /MIR /R:2 /W:5",
    }[tool]
    head = [_prompt(rng, platform, "~" if platform == "linux" else "C:\\", cmd)]
    if status == "success":
        if tool == "rsync":
            body = [
                "sending incremental file list",
                f"{src}/main.py",
                f"{src}/config.yaml",
                f"{src}/data/input.csv",
                "",
                f"sent {_num(rng, 100, 90000)} bytes  received {_num(rng, 100, 9000)} bytes  "
                f"{rng.uniform(0.1, 9):.2f} bytes/sec",
                f"total size is {_num(rng, 1000, 9000000)}  speedup is {rng.uniform(1, 8):.2f}",
            ]
        elif tool == "rclone":
            body = [
                "Transferred:   " + str(_num(rng, 10, 900)) + " MiB / "
                + str(_num(rng, 10, 900)) + " MiB, 100%, " + str(_num(rng, 1, 90)) + " MiB/s",
                "Checks:        " + str(_num(rng, 10, 900)) + " / " + str(_num(rng, 10, 900))
                + ", 100%",
                "Deleted:       " + str(_num(rng, 0, 50)),
                "Elapsed time:  " + str(_num(rng, 1, 900)) + "s",
            ]
        else:
            body = [
                f"   ROBOCOPY     ::     Robust File Copy for Windows",
                "",
                f"   Started : 11/6/2026 10:14:22 AM",
                f"   Source : {src}",
                f"   Dest : {dst}",
                "",
                f"     Files : {_num(rng, 10, 900)}",
                f"     Bytes : {_num(rng, 1000, 90000000)}",
                f"   Copied : {_num(rng, 10, 900)}",
                "    Ended : 11/6/2026 10:15:01 AM",
            ]
        return _join(head, [(x, "neutral") for x in body],
                     [("Backup completed successfully.", "success"),
                      _footer(rng, platform, 0)]), 0
    if status == "fail":
        errs = _p(rng, [
            [(f"rsync: [receiver] write failed on \"{dst}/{_p(rng, FILES)}\": "
              f"No space left on device (28)", "fail"),
             ("rsync error: error in file IO (code 11) at receiver.c(378)", "fail")],
            [(f"rsync: [sender] change_dir \"{src}\" failed: No such file or directory (2)", "fail"),
             ("rsync error: some files/attrs were not transferred (code 23)", "fail")],
            [("rclone: error reading source directory: directory not found", "fail"),
             ("Failed to sync: 1 error(s) occurred", "fail")],
            [("ERROR 5 (0x00000005) Copying File " + src + "\\main.py", "fail"),
             ("Access is denied.", "block")],
            [("ERROR 112 (0x00000070) Creating Destination Directory " + dst, "fail"),
             ("There is not enough space on the disk.", "fail")],
            [("Couldn't read source: connection reset by peer", "fail")],
        ])
        return _join(head, errs, [_footer(rng, platform, 1)]), 1
    if status == "block":
        body = _p(rng, [
            [(f"rsync: [receiver] mkdir \"{dst}\" failed: Permission denied (13)", "block")],
            [("rclone: failed to copy: 403 Forbidden: access denied", "block")],
            [("ERROR 5 (0x00000005) Copying File " + src, "block"),
             ("Access is denied.", "block")],
            [("The request was blocked by the storage policy.", "block")],
        ])
        return _join(head, body, [_footer(rng, platform, 126)]), 126
    body = _fillers_from(rng, _num(rng, 2, 7), [
        "sending incremental file list",
        f"{src}/main.py",
        f"{src}/config.yaml",
        f"{_num(rng, 10, 900)} files transferred",
        f"{_num(rng, 10, 900)} MiB / {_num(rng, 10, 900)} MiB, 100%",
    ])
    return _join(head, body), None


# --------------------------------------------------------------------------
# databases
# --------------------------------------------------------------------------
def _gen_database(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    tool = _p(rng, ["psql", "mysql", "sqlite3", "redis-cli", "alembic"])
    db = _p(rng, ["app", "analytics", "users", "orders", "cache"])
    cmd = {
        "psql": f"psql -h {_p(rng, IPS)} -U {_p(rng, USERS)} -d {db} -c 'SELECT count(*) FROM orders;'",
        "mysql": f"mysql -h {_p(rng, IPS)} -u {_p(rng, USERS)} -p {db} < migration.sql",
        "sqlite3": f"sqlite3 data/{db}.db '.tables'",
        "redis-cli": f"redis-cli -h {_p(rng, IPS)} INFO",
        "alembic": "alembic upgrade head",
    }[tool]
    head = [_prompt(rng, platform, "~/proj" if platform == "linux" else "C:\\proj", cmd)]
    if status == "success":
        body = {
            "psql": [
                f" count ",
                "-------",
                f" {_num(rng, 10, 9000000)}",
                "(1 row)",
                f"Time: {rng.uniform(0.1, 900):.2f} ms",
            ],
            "mysql": [
                f"Database changed",
                f"Query OK, {_num(rng, 1, 9000)} rows affected ({rng.uniform(0.01, 9):.2f} sec)",
                f"Records: {_num(rng, 1, 9000)}  Duplicates: 0  Warnings: 0",
            ],
            "sqlite3": [
                "orders        users         products      migrations",
                f"cache_hits    sessions      audit_log",
                f"{_num(rng, 5, 40)} tables listed.",
            ],
            "redis-cli": [
                "# Server",
                "redis_version:" + _p(rng, ["7.2.4", "7.0.15", "6.2.14"]),
                f"redis_mode:standalone",
                f"os:Linux {_num(rng, 5, 6)}.{_num(rng, 1, 15)}.0-{_num(rng, 10, 99)}-generic x86_64",
                f"connected_clients:{_num(rng, 1, 900)}",
                f"used_memory_human:{rng.uniform(0.1, 900):.2f}M",
                f"keyspace_hits:{_num(rng, 100, 9000000)}",
                "role:master",
            ],
            "alembic": [
                "INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.",
                "INFO  [alembic.runtime.migration] Will assume transactional DDL.",
                f"INFO  [alembic.runtime.migration] Running upgrade  -> a1b2c3d4e5f6, initial schema",
                f"INFO  [alembic.runtime.migration] Running upgrade a1b2c3d4e5f6 -> b2c3d4e5f6a7, add index",
            ],
        }[tool]
        return _join(head, [(x, "neutral") for x in body[:-1]],
                     [(body[-1], "success"), _footer(rng, platform, 0)]), 0
    if status == "fail":
        errs = _p(rng, [
            [(f"psql: error: connection to server at \"{_p(rng, IPS)}\", port 5432 failed: "
              f"Connection refused", "fail"),
             ("\tIs the server running on that host and accepting TCP/IP connections?", "fail")],
            [(f"ERROR:  relation \"orders\" does not exist", "fail"),
             ("LINE 1: SELECT count(*) FROM orders;", "fail")],
            [(f"ERROR 1045 (28000): Access denied for user '{_p(rng, USERS)}'@'localhost' "
              f"(using password: YES)", "block")],
            [("ERROR: deadlock detected", "fail"),
             ("DETAIL: Process 4120 waits for ShareLock on transaction 8891; blocked by process 4121.", "fail"),
             ("HINT: See server log for query details.", "fail")],
            [("Error: database is locked", "fail")],
            [("sqlite3.OperationalError: no such table: users", "fail")],
            [("Could not connect to Redis at " + _p(rng, IPS) + ":6379: Connection refused", "fail")],
            [("sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) could not connect to server", "fail")],
            [("alembic.util.exc.CommandError: Can't locate revision identified by 'c3d4e5f6a7b8'", "fail")],
        ])
        return _join(head, errs, [_footer(rng, platform, 1)]), 1
    if status == "block":
        body = _p(rng, [
            [("psql: error: permission denied for table orders", "block"),
             ("ERROR:  permission denied for schema public", "block")],
            [("ERROR 1142 (42000): SELECT command denied to user 'app'@'localhost' for table 'orders'", "block")],
            [("(error) NOAUTH Authentication required.", "block")],
            [("ERROR: must be owner of table orders", "block")],
            [("Access denied; you need (at least one of) the SUPER privilege(s) for this operation", "block")],
            [("SSL required: connection blocked by server policy", "block")],
        ])
        return _join(head, body, [_footer(rng, platform, 126)]), 126
    body = _fillers_from(rng, _num(rng, 2, 7), [
        "Connecting to database...",
        f"server version: {_num(rng, 12, 16)}.{_num(rng, 1, 9)}",
        "executing query...",
        f"processed {_num(rng, 100, 90000)} rows",
        "receiving data...",
    ])
    return _join(head, body), None


# --------------------------------------------------------------------------
# cloud / infrastructure
# --------------------------------------------------------------------------
def _gen_cloud(rng, platform, status, cfg) -> Tuple[List[Line], Optional[int]]:
    tool = _p(rng, ["kubectl", "terraform", "ansible", "aws", "gcloud", "docker"])
    ns = _p(rng, ["default", "prod", "staging", "monitoring"])
    cmd = {
        "kubectl": f"kubectl apply -f deploy/{_p(rng, ['app.yaml', 'web.yaml', 'worker.yaml'])} -n {ns}",
        "terraform": "terraform apply -auto-approve",
        "ansible": f"ansible-playbook -i inventory deploy.yml --limit {_p(rng, ['web', 'db', 'all'])}",
        "aws": f"aws s3 sync ./dist s3://{_p(rng, ['app-bucket', 'static-assets'])} --delete",
        "gcloud": f"gcloud app deploy app.yaml --quiet --project {_p(rng, ['prod-01', 'demo-42'])}",
        "docker": f"docker compose -f compose.prod.yml up -d --build",
    }[tool]
    head = [_prompt(rng, platform, "~/proj" if platform == "linux" else "C:\\proj", cmd)]
    if status == "success":
        body = {
            "kubectl": [
                f"deployment.apps/{_p(rng, ['app', 'web', 'worker'])} configured",
                f"service/{_p(rng, ['app', 'web'])} unchanged",
                f"configmap/app-config created",
                f"deployment \"app\" successfully rolled out",
            ],
            "terraform": [
                "Terraform used the selected providers to generate the following execution plan.",
                "  # aws_instance.web will be created",
                f"  + resource \"aws_instance\" \"web\" {{",
                f"      + ami           = \"ami-0{_num(rng, 100000, 999999)}\"",
                f"      + instance_type = \"t3.medium\"",
                "    }",
                "Plan: 3 to add, 0 to change, 0 to destroy.",
                f"aws_instance.web: Creating...",
                f"aws_instance.web: Creation complete after {_num(rng, 1, 90)}s [id=i-0{_num(rng, 1000000, 9999999)}]",
                "Apply complete! Resources: 3 added, 0 changed, 0 destroyed.",
            ],
            "ansible": [
                "PLAY [Deploy application] ******************************************************",
                "TASK [Gathering Facts] *********************************************************",
                f"ok: [{_p(rng, ['web-01', 'web-02', 'db-01'])}]",
                "TASK [Install packages] ********************************************************",
                f"changed: [{_p(rng, ['web-01', 'web-02'])}]",
                "TASK [Copy application] ********************************************************",
                f"changed: [{_p(rng, ['web-01', 'web-02'])}]",
                "PLAY RECAP *********************************************************************",
                f"web-01                     : ok={_num(rng, 5, 20)} changed={_num(rng, 1, 10)} "
                f"unreachable=0 failed=0 skipped={_num(rng, 0, 5)}",
            ],
            "aws": [
                "upload: dist/index.html to s3://app-bucket/index.html",
                f"upload: dist/assets/main.{_p(rng, ['js', 'css'])} to s3://app-bucket/assets/main.js",
                f"delete: s3://app-bucket/assets/old-{_num(rng, 1, 99)}.js",
                f"Completed {_num(rng, 3, 900)} file(s) with {_num(rng, 0, 10)} file(s) remaining.",
            ],
            "gcloud": [
                "Services to deploy:",
                f"descriptor:      [/home/dev/proj/app.yaml]",
                f"source:          [/home/dev/proj]",
                f"target project:  [prod-01]",
                f"target service:  [default]",
                f"target version:  [v{_num(rng, 1, 99)}]",
                f"target url:      [https://prod-01.appspot.com]",
                f"Beginning deployment of service [default]...",
                f"Updating service [default]...done.",
                f"Deployed service [default] to [https://prod-01.appspot.com]",
            ],
            "docker": [
                f"[+] Building {_num(rng, 10, 90)}.{_num(rng, 1, 9)}s (12/12) FINISHED",
                " => [internal] load build definition from Dockerfile",
                " => [1/5] FROM docker.io/library/python:3.10-slim",
                " => [3/5] COPY . .",
                " => [4/5] RUN pip install -r requirements.txt",
                " => exporting to image",
                " => => naming to docker.io/library/app:latest",
                f"[+] Running 3/3",
                f" ✔ Network {ns}_default       Created",
                f" ✔ Container {ns}-app-1       Started",
            ],
        }[tool]
        return _join(head, [(x, "neutral") for x in body[:-1]],
                     [(body[-1], "success"), _footer(rng, platform, 0)]), 0
    if status == "fail":
        errs = _p(rng, [
            [("Error from server (NotFound): error when creating \"deploy/app.yaml\": "
              "deployments.apps \"app\" not found", "fail")],
            [("Error from server (BadRequest): container \"app\" in pod \"app-1\" is waiting "
              "to start: ImagePullBackOff", "fail")],
            [("Error: Error acquiring the state lock", "fail"),
             ("Lock Info:", "fail"), ("  ID:       " + str(_num(rng, 1000, 9999)), "fail")],
            [("Error: Failed to install provider", "fail")],
            [("fatal: [web-01]: FAILED! => {\"msg\": \"Failed to connect to the host via ssh: "
              "Connection timed out\"}", "fail")],
            [("fatal: [db-01]: FAILED! => {\"msg\": \"apt lock is held by another process\"}", "fail")],
            [("upload failed: ./dist/main.js to s3://app-bucket/main.js An error occurred "
              "(AccessDenied) when calling the PutObject operation: Access Denied", "block")],
            [("ERROR: (gcloud.app.deploy) PERMISSION_DENIED: The caller does not have permission", "block")],
            [("Error response from daemon: pull access denied for registry/app, repository does not "
              "exist or may require 'docker login'", "block")],
            [("ERROR: Service 'app' failed to build: process \"/bin/sh -c pip install\" did not "
              "complete successfully: exit code: 1", "fail")],
        ])
        return _join(head, errs, [_footer(rng, platform, 1)]), 1
    if status == "block":
        body = _p(rng, [
            [("Error from server (Forbidden): deployments.apps \"app\" is forbidden: "
              "User \"dev\" cannot patch resource \"deployments\" in API group \"apps\"", "block")],
            [("Error: Insufficient permissions to create resources", "block")],
            [("fatal: [web-01]: FAILED! => {\"msg\": \"sudo: a password is required\"}", "block")],
            [("An error occurred (AccessDenied) when calling the PutObject operation: Access Denied", "block")],
            [("ERROR: (gcloud.app.deploy) PERMISSION_DENIED", "block")],
            [("ERROR: unauthorized: authentication required", "block")],
        ])
        return _join(head, body, [_footer(rng, platform, 126)]), 126
    body = _fillers_from(rng, _num(rng, 2, 8), [
        "Refreshing state...",
        f"module.app.aws_instance.web: Still creating... [{_num(rng, 1, 90)}s elapsed]",
        "Waiting for rollout to finish: 1 of 3 updated replicas are available...",
        "PLAY [Deploy application] ******************************************************",
        "TASK [Gathering Facts] *********************************************************",
        " => [internal] load build definition from Dockerfile",
    ])
    return _join(head, body), None


# --------------------------------------------------------------------------
# helper: neutral context lines (used by synth.generate_document)
# --------------------------------------------------------------------------
_CONTEXT = [
    "[INFO] session={hex} request={hex}",
    "[DEBUG] worker={n} queue={n} retry={n}",
    "[INFO] endpoint=10.0.{n}.{n}:{port}",
    "[INFO] package={pkg} version={n}.{n}.{n}",
    "[WARN] slow operation: {f} ms",
    "[INFO] cache hit ratio={f}",
    "[DEBUG] thread={n} state=RUNNABLE",
    "[INFO] opening /var/log/app-{n}.log",
    "[INFO] loaded {n} records from {pkg} cache",
    "[INFO] retry budget remaining={n}",
    "[DEBUG] lock wait={f} ms",
    "[INFO] loading shared libraries",
    "[INFO] configuration file: /etc/app/config.yaml",
    "[INFO] using {n} worker threads",
    "[INFO] log level: INFO",
    "[INFO] reading environment variables",
    "[INFO] cache directory: /home/{user}/.cache/app",
    "[INFO] startup complete in {f}s",
    "[INFO] processing queue: {n} items",
    "monitor: cpu={n}% mem={f}GiB",
    "open file descriptors: {n}",
    "locale: en_US.UTF-8",
    "transport: tcp",
    "session id: {n}",
    "loading plugins from /usr/lib/app/plugins",
    "waiting for lock ... acquired",
    "handshake complete",
]


def _fillers_from(rng: random.Random, n: int, pool) -> List[Line]:
    return [(str(_p(rng, pool)), "neutral") for _ in range(max(0, n))]


def context_lines(rng: random.Random, n: int) -> List[Line]:
    out = []
    for _ in range(max(0, n)):
        text = _p(rng, _CONTEXT)
        text = text.format(
            n=_num(rng, 1, 900), f=rng.uniform(0.01, 9), user=_p(rng, USERS),
            port=_num(rng, 1024, 65535), pkg=_p(rng, PACKAGES),
            hex=f"{rng.randrange(16**8):08x}",
        )
        out.append((text, "neutral"))
    return out


EXTRA_GENERATORS = {
    "package_update": _gen_package_update,
    "compile": _gen_compile,
    "media": _gen_media,
    "data_process": _gen_data_process,
    "archive": _gen_archive,
    "backup": _gen_backup,
    "database": _gen_database,
    "cloud": _gen_cloud,
}
