"""A broad catalogue of everyday console commands and their output.

The other generators model complete workflows (pip, systemd, builds, ...).
This module covers the long tail of short sysadmin commands: file management,
networking, users, monitoring, disks, permissions, search, compression,
processes, shells and package inspection.

Each catalogue entry provides command variants and status-specific output.
The actual command is always echoed in the prompt, so the corpus contains the
call itself as well as its result.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from .synth import (
    DOMAINS,
    FILES,
    HOSTS,
    IPS,
    PACKAGES,
    PROJECTS,
    PORTS,
    SERVICES,
    USERS,
    Line,
    _footer,
    _join,
    _num,
    _p,
    _prompt,
)

_LINUX_DIRS = [
    "/var/log", "/etc/nginx", "/usr/local/bin", "/home/dev/proj", "/mnt/data",
    "/opt/app", "/var/lib/docker", "/tmp", "~/.cache/pip", "/srv/www",
]
_WIN_DIRS = [
    "C:\\proj", "C:\\Windows\\System32", "D:\\builds\\release",
    "C:\\Users\\dev\\AppData\\Local\\Temp", "C:\\Program Files\\App",
]
_LINUX_FILES = [
    "main.py", "config.yaml", "requirements.txt", "access.log", "app.log",
    "Dockerfile", "Makefile", "dataset.csv", "README.md",
]
_WIN_FILES = [
    "main.py", "setup.exe", "app.config", "System.log", "report.csv",
    "installer.msi", "notes.txt",
]
_GROUPS = ["sudo", "docker", "www-data", "developers", "admins", "video", "adm"]


def _ctx(rng: random.Random, platform: str) -> Dict[str, object]:
    dirs = _WIN_DIRS if platform == "windows" else _LINUX_DIRS
    files = _WIN_FILES if platform == "windows" else _LINUX_FILES
    return {
        "file": _p(rng, files),
        "file2": _p(rng, files),
        "dir": _p(rng, dirs),
        "dir2": _p(rng, dirs),
        "pkg": _p(rng, PACKAGES),
        "svc": _p(rng, SERVICES),
        "user": _p(rng, USERS),
        "user2": _p(rng, USERS),
        "group": _p(rng, _GROUPS),
        "host": _p(rng, HOSTS),
        "domain": _p(rng, DOMAINS),
        "ip": _p(rng, IPS),
        "port": _p(rng, PORTS),
        "project": _p(rng, PROJECTS),
        "n": _num(rng, 1, 9000),
        "n2": _num(rng, 100, 90000),
        "f": round(rng.uniform(0.1, 900.0), 2),
        "cpu": _num(rng, 2, 128),
    }


# --------------------------------------------------------------------------
# catalogue
# --------------------------------------------------------------------------
def _e(lin=None, win=None, ok=None, bad=None, deny=None, complex=False) -> Dict:
    return {
        "lin": lin or [],
        "win": win or [],
        "ok": ok or [],
        "bad": bad or [],
        "deny": deny or [],
        "complex": bool(complex),
    }


CATALOG: List[Dict] = [
    # ---------------- files and directories ----------------
    _e(
        lin=["ls -la", "ls -lh /var/log", "ls -l --color=auto"],
        win=["dir C:\\proj", "dir /a C:\\Windows\\System32"],
        ok=[
            "total 48", "drwxr-xr-x  8 dev dev 4096 Nov  6 10:14 .",
            "drwxr-xr-x 18 dev dev 4096 Nov  5 21:02 ..",
            "-rw-r--r--  1 dev dev 1204 Nov  6 10:14 {file}",
            "-rwxr-xr-x  1 dev dev 8421 Nov  6 09:51 build.sh",
            "Volume in drive C has no label.", "Directory of {dir}", "",
            "11/06/2026  10:14 AM    <DIR>          .",
            "11/06/2026  10:14 AM             1,204 {file}",
        ],
        bad=["ls: cannot access '{file}': No such file or directory"],
        deny=["ls: cannot open directory '{dir}': Permission denied"],
    ),
    _e(
        lin=["cd {dir} && pwd", "pwd"],
        win=["cd /d {dir} && cd", "cd"],
        ok=["{dir}"],
        bad=["bash: cd: {dir}: No such file or directory",
             "The system cannot find the path specified."],
        deny=["bash: cd: {dir}: Permission denied"],
    ),
    _e(
        lin=["mkdir -p build/release", "mkdir -p {dir}/cache"],
        win=["mkdir C:\\proj\\build", "md {dir}\\cache"],
        ok=["created directory '{dir}/cache'"],
        bad=["mkdir: cannot create directory '{dir}': File exists"],
        deny=["mkdir: cannot create directory '{dir}': Permission denied"],
    ),
    _e(
        lin=["cp -v {file} {dir}/", "cp -r src/ {dir}/"],
        win=["copy {file} {dir}\\", "xcopy /E /I src {dir}\\src"],
        ok=["'{file}' -> '{dir}/{file}'", "1 file copied."],
        bad=["cp: cannot stat '{file}': No such file or directory"],
        deny=["cp: cannot create regular file '{dir}/{file}': Permission denied"],
    ),
    _e(
        lin=["mv -v {file} {dir}/", "mv old.log {dir}/"],
        win=["move {file} {dir}\\", "move /Y old.log {dir}\\"],
        ok=["renamed '{file}' -> '{dir}/{file}'"],
        bad=["mv: cannot stat '{file}': No such file or directory"],
        deny=["mv: cannot move '{file}' to '{dir}/': Permission denied"],
    ),
    _e(
        lin=["rm -rf build/tmp", "rm -f old.log"],
        win=["del /Q {file}", "rmdir /S /Q build\\tmp"],
        ok=["removed '{file}'"],
        bad=["rm: cannot remove '{file}': No such file or directory",
             "The system cannot find the file specified."],
        deny=["rm: cannot remove '{dir}/{file}': Permission denied",
              "Access is denied."],
    ),
    _e(
        lin=["touch {file}", "touch -c {dir}/.keep"],
        win=["type nul > {file}", "echo. > {dir}\\.keep"],
        ok=["touched '{file}'"],
        bad=["touch: cannot touch '{dir}/{file}': No such file or directory"],
        deny=["touch: cannot touch '{file}': Permission denied"],
    ),
    _e(
        lin=["file {file}", "stat {file}"],
        win=["where {file}", "dir {file}"],
        ok=[
            "{file}: Python script, ASCII text executable",
            "  File: {file}",
            "  Size: 1204       Blocks: 8          IO Block: 4096 regular file",
            "Access: (0644/-rw-r--r--)  Uid: ( 1000/     dev)   Gid: ( 1000/     dev)",
            "Modify: 2026-11-06 10:14:22.000000000 +0000",
        ],
        bad=["stat: cannot statx '{file}': No such file or directory"],
        deny=["stat: cannot statx '{file}': Permission denied"],
    ),
    _e(
        lin=["tree -L 2 src", "tree -a -I '.git|__pycache__'"],
        ok=[
            "src",
            "├── main.py",
            "├── parser.py",
            "├── io",
            "│   ├── reader.py",
            "│   └── writer.py",
            "└── utils.py",
            "",
            "3 directories, 8 files",
        ],
        bad=["tree: {dir}: No such file or directory"],
        deny=["tree: {dir}: Permission denied"],
    ),
    _e(
        lin=["ln -sf /usr/bin/python3 /usr/local/bin/python", "ln -s {file} link"],
        ok=["created symlink 'link' -> '{file}'"],
        bad=["ln: failed to create symbolic link 'link': File exists"],
        deny=["ln: failed to create symbolic link '/usr/local/bin/python': Permission denied"],
    ),
    _e(
        lin=["cat /etc/os-release", "cat {file}"],
        win=["type {file}"],
        ok=[
            'PRETTY_NAME="Ubuntu 22.04.4 LTS"',
            'NAME="Ubuntu"',
            'VERSION_ID="22.04"',
            'VERSION="22.04.4 LTS (Jammy Jellyfish)"',
            'ID=ubuntu',
        ],
        bad=["cat: {file}: No such file or directory"],
        deny=["cat: {file}: Permission denied"],
    ),
    _e(
        lin=["echo $PATH", "echo 'build complete'"],
        win=["echo %PATH%", "echo build complete"],
        ok=["/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"],
        bad=["bash: echo: write error: Broken pipe"],
    ),
    _e(
        lin=["head -n 5 app.log", "tail -n 20 app.log"],
        win=["powershell -c \"Get-Content app.log -TotalCount 5\""],
        ok=[
            "2026-11-06 10:14:20 INFO  starting worker",
            "2026-11-06 10:14:21 INFO  connected to database",
            "2026-11-06 10:14:22 WARN  retry 1/3",
            "2026-11-06 10:14:23 INFO  request handled in 42 ms",
            "2026-11-06 10:14:24 INFO  idle",
        ],
        bad=["head: cannot open '{file}' for reading: No such file or directory"],
        deny=["tail: cannot open '{file}' for reading: Permission denied"],
    ),
    _e(
        lin=["wc -l app.log", "sort -u names.txt | wc -l", "uniq -c ips.txt"],
        ok=["  1284 app.log", "  42", "     7 10.0.0.14"],
        bad=["wc: {file}: No such file or directory"],
        deny=["wc: {file}: Permission denied"],
    ),
    _e(
        lin=["grep -Rn 'ERROR' logs/", "grep -n 'timeout' app.log"],
        win=["findstr /S /I \"error\" *.log", "findstr /N \"timeout\" app.log"],
        ok=[
            "logs/app.log:42:ERROR connection reset by peer",
            "logs/worker.log:17:ERROR retry limit exceeded",
            "42:timeout while waiting for lock",
        ],
        bad=["grep: {file}: No such file or directory"],
        deny=["grep: {dir}: Permission denied"],
    ),
    _e(
        lin=["awk -F: '{print $1}' /etc/passwd", "awk '{sum+=$2} END {print sum}' data.txt"],
        ok=["root", "daemon", "bin", "sys", "dev", "42"],
        bad=["awk: cannot open {file} for reading (No such file or directory)"],
        deny=["awk: cannot open {file} for reading (Permission denied)"],
    ),
    _e(
        lin=["sed -i 's/localhost/127.0.0.1/g' config.yaml", "sed -n '1,20p' app.log"],
        ok=["updated 3 occurrences in config.yaml", "2026-11-06 10:14:20 INFO starting worker"],
        bad=["sed: can't read {file}: No such file or directory"],
        deny=["sed: couldn't open temporary file: Permission denied"],
    ),
    _e(
        lin=["find . -name '*.log' -mtime -1", "locate nginx.conf"],
        win=["where /R C:\\proj *.log", "dir /S /B *.log"],
        ok=["./logs/app.log", "./logs/worker.log", "/etc/nginx/nginx.conf"],
        bad=["find: '{dir}': No such file or directory"],
        deny=["find: '{dir}': Permission denied"],
    ),
    _e(
        lin=["rg -n 'TODO|FIXME' src/"],
        ok=["src/main.py:18:# TODO: retry policy", "src/io.py:64:# FIXME: handle EOF"],
        bad=["rg: {dir}: No such file or directory"],
        deny=["rg: {dir}: Permission denied"],
    ),
    _e(
        lin=["find . -name '*.tmp' | xargs rm -f", "cat files.txt | xargs -n1 wc -l"],
        ok=["removed 12 temporary files", "42 {file}"],
        bad=["xargs: rm: No such file or directory"],
        deny=["xargs: cannot remove '{dir}': Permission denied"],
    ),

    # ---------------- networking ----------------
    _e(
        lin=["ping -c 4 {domain}", "ping -c 2 {ip}"],
        win=["ping {domain}", "ping -n 4 {ip}"],
        ok=[
            "PING {domain} ({ip}) 56(84) bytes of data.",
            "64 bytes from {ip}: icmp_seq=1 ttl=57 time=12.4 ms",
            "64 bytes from {ip}: icmp_seq=2 ttl=57 time=11.8 ms",
            "--- {domain} ping statistics ---",
            "4 packets transmitted, 4 received, 0% packet loss",
        ],
        bad=["ping: {domain}: Name or service not known",
             "4 packets transmitted, 0 received, 100% packet loss"],
        deny=["ping: socket: Operation not permitted"],
    ),
    _e(
        lin=["traceroute -n {domain}", "traceroute -I {ip}"],
        win=["tracert {domain}"],
        ok=[
            "traceroute to {domain} ({ip}), 30 hops max, 60 byte packets",
            " 1  192.168.1.1  1.024 ms  0.981 ms  0.944 ms",
            " 2  10.0.0.1  4.211 ms  4.180 ms  4.150 ms",
            " 3  {ip}  12.442 ms  12.390 ms  12.355 ms",
        ],
        bad=["traceroute: unknown host {domain}"],
        deny=["traceroute: socket: Operation not permitted"],
    ),
    _e(
        lin=["nmcli device status", "nmcli connection show"],
        ok=[
            "DEVICE  TYPE      STATE         CONNECTION",
            "enp5s0  ethernet  connected     Wired connection 1",
            "lo      loopback  unmanaged     --",
            "wlp3s0  wifi      disconnected  --",
        ],
        bad=["Error: NetworkManager is not running."],
        deny=["Error: Could not create NMClient object: Access denied."],
    ),
    _e(
        lin=["ip -br addr", "ip route", "ss -tulpn"],
        ok=[
            "lo               UNKNOWN        127.0.0.1/8 ::1/128",
            "enp5s0           UP             10.0.0.14/24 fe80::1c2b/64",
            "default via 10.0.0.1 dev enp5s0 proto dhcp metric 100",
            "Netid State  Recv-Q Send-Q Local Address:Port Peer Address:Port",
            "tcp   LISTEN 0      128    0.0.0.0:22         0.0.0.0:*",
            "tcp   LISTEN 0      511    0.0.0.0:80         0.0.0.0:*",
        ],
        bad=["Cannot open netlink socket: No such file or directory"],
        deny=["RTNETLINK answers: Operation not permitted"],
    ),
    _e(
        lin=["netstat -tulpn", "netstat -rn"],
        win=["netstat -ano", "netstat -rn"],
        ok=[
            "Proto Recv-Q Send-Q Local Address           Foreign Address         State",
            "tcp        0      0 0.0.0.0:22              0.0.0.0:*               LISTEN",
            "tcp        0      0 127.0.0.1:5432          0.0.0.0:*               LISTEN",
            "udp        0      0 0.0.0.0:68              0.0.0.0:*",
        ],
        bad=["netstat: command not found"],
        deny=["netstat: no support for `AF INET (tcp)' on this system."],
    ),
    _e(
        lin=["dig +short {domain}", "nslookup {domain}"],
        win=["nslookup {domain}"],
        ok=[
            "Server:         127.0.0.53",
            "Address:        127.0.0.53#53",
            "Name:   {domain}",
            "Address: {ip}",
        ],
        bad=[";; connection timed out; no servers could be reached",
             "** server can't find {domain}: NXDOMAIN"],
    ),
    _e(
        lin=["curl -sS -I https://{domain}", "wget -q --spider https://{domain}"],
        win=["curl.exe -I https://{domain}"],
        ok=[
            "HTTP/2 200",
            "content-type: text/html; charset=utf-8",
            "content-length: {n2}",
            "server: nginx",
            "date: Fri, 06 Nov 2026 10:14:22 GMT",
        ],
        bad=["curl: (6) Could not resolve host: {domain}",
             "curl: (7) Failed to connect to {domain} port 443: Connection refused"],
        deny=["curl: (22) The requested URL returned error: 403",
              "Access blocked by network policy"],
    ),
    _e(
        lin=["ifconfig", "route -n"],
        win=["ipconfig /all", "route print"],
        ok=[
            "enp5s0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500",
            "        inet 10.0.0.14  netmask 255.255.255.0  broadcast 10.0.0.255",
            "        inet6 fe80::1c2b:3dff:fe4a:12ab  prefixlen 64  scopeid 0x20<link>",
            "Kernel IP routing table",
            "Destination     Gateway         Genmask         Flags Metric Ref    Use Iface",
            "0.0.0.0         10.0.0.1        0.0.0.0         UG    100    0        0 enp5s0",
        ],
        bad=["ifconfig: command not found"],
        deny=["ifconfig: /proc/net/dev: Permission denied"],
    ),

    # ---------------- users and groups ----------------
    _e(
        lin=["id", "whoami", "groups"],
        win=["whoami", "whoami /groups"],
        ok=[
            "uid=1000(dev) gid=1000(dev) groups=1000(dev),27(sudo),999(docker)",
            "dev",
            "dev sudo docker",
        ],
        bad=["id: cannot find name for user ID 4242"],
    ),
    _e(
        lin=["sudo useradd -m -s /bin/bash {user}", "sudo usermod -aG {group} {user}"],
        ok=["user '{user}' created", "user '{user}' added to group '{group}'"],
        bad=["useradd: user '{user}' already exists"],
        deny=["useradd: Permission denied.", "sudo: a password is required"],
    ),
    _e(
        lin=["sudo groupadd {group}", "groups {user}"],
        ok=["group '{group}' created", "{user} : {user} {group}"],
        bad=["groupadd: group '{group}' already exists"],
        deny=["groupadd: Permission denied."],
    ),
    _e(
        lin=["sudo passwd {user}", "passwd"],
        ok=["passwd: password updated successfully"],
        bad=["passwd: Authentication token manipulation error"],
        deny=["passwd: You may not view or modify password information for {user}."],
    ),
    _e(
        lin=["who", "w", "last -n 5"],
        ok=[
            "dev      pts/0        2026-11-06 10:01 (10.0.0.14)",
            "USER     TTY      FROM             LOGIN@   IDLE   JCPU   PCPU WHAT",
            "dev      pts/0    10.0.0.14        10:01    2.00s  0.12s  0.01s w",
            "dev      pts/0        10.0.0.14    Fri Nov  6 10:01   still logged in",
            "reboot   system boot  6.8.0-45-generic Fri Nov  6 09:58   still running",
        ],
        bad=["last: cannot open /var/log/wtmp: No such file or directory"],
        deny=["last: cannot open /var/log/wtmp: Permission denied"],
    ),

    # ---------------- system information ----------------
    _e(
        lin=["lscpu", "cat /proc/cpuinfo | head -n 20"],
        win=["wmic cpu get name,numberofcores,numberoflogicalprocessors"],
        ok=[
            "Architecture:            x86_64",
            "  CPU op-mode(s):        32-bit, 64-bit",
            "  Address sizes:         46 bits physical, 48 bits virtual",
            "  CPU(s):                {cpu}",
            "  Model name:            Intel(R) Xeon(R) CPU E5-2670 v3 @ 2.30GHz",
            "  Thread(s) per core:    2",
            "  Core(s) per socket:    8",
            "  Socket(s):             2",
            "  L3 cache:              60 MiB",
        ],
        bad=["lscpu: command not found"],
        deny=["lscpu: cannot read /sys/devices/system/cpu: Permission denied"],
    ),
    _e(
        lin=["free -h", "free -m"],
        win=["systeminfo | findstr /C:\"Total Physical Memory\" /C:\"Available Physical Memory\""],
        ok=[
            "               total        used        free      shared  buff/cache   available",
            "Mem:            62Gi        18Gi       2.1Gi       1.4Gi        42Gi        41Gi",
            "Swap:          8.0Gi       512Mi       7.5Gi",
        ],
        bad=["free: cannot read /proc/meminfo"],
        deny=["free: /proc/meminfo: Permission denied"],
    ),
    _e(
        lin=["top -b -n 1", "top -b -n 1 -o %CPU | head -n 20"],
        ok=[
            "top - 10:14:22 up 2 days,  3:41,  2 users,  load average: 1.42, 1.18, 0.97",
            "Tasks: 412 total,   2 running, 410 sleeping,   0 stopped,   0 zombie",
            "%Cpu(s): 12.4 us,  3.1 sy,  0.0 ni, 84.1 id,  0.4 wa",
            "MiB Mem :  63998.0 total,   2184.1 free,  18442.6 used,  43371.3 buff/cache",
            "  PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND",
            " 4120 dev       20   0  8421m   1.2g   128m R  42.1   1.9   3:12.44 python3",
        ],
        bad=["top: failed tty get"],
        deny=["top: /proc/4120/stat: Permission denied"],
    ),
    _e(
        lin=["du -sh *", "du -ah . | sort -h | tail -n 10"],
        win=["powershell -c \"Get-ChildItem | Measure-Object -Property Length -Sum\""],
        ok=[
            "1.2G    dataset",
            "420M    checkpoints",
            "84M     logs",
            "12M     src",
            "4.0K    README.md",
        ],
        bad=["du: cannot access '{dir}': No such file or directory"],
        deny=["du: cannot read directory '{dir}': Permission denied"],
    ),
    _e(
        lin=["df -hT", "df -i"],
        win=["wmic logicaldisk get caption,freespace,size"],
        ok=[
            "Filesystem     Type      Size  Used Avail Use% Mounted on",
            "/dev/nvme0n1p2 ext4      468G  389G   55G  88% /",
            "tmpfs          tmpfs      32G   12M   32G   1% /dev/shm",
            "/dev/nvme0n1p1 vfat      511M   34M  478M   7% /boot/efi",
        ],
        bad=["df: /mnt/data: Stale file handle"],
        deny=["df: /root: Permission denied"],
    ),
    _e(
        lin=["uptime", "uptime -p"],
        ok=[" 10:14:22 up 2 days,  3:41,  2 users,  load average: 1.42, 1.18, 0.97",
            "up 2 days, 3 hours, 41 minutes"],
        bad=["uptime: /proc/uptime: No such file or directory"],
    ),
    _e(
        lin=["uname -a", "lsb_release -a", "hostnamectl"],
        win=["ver", "systeminfo"],
        ok=[
            "Linux {host} 6.8.0-45-generic #45-Ubuntu SMP PREEMPT_DYNAMIC x86_64 GNU/Linux",
            "Distributor ID: Ubuntu",
            "Description:    Ubuntu 22.04.4 LTS",
            "Release:        22.04",
            "Codename:       jammy",
            "Static hostname: {host}",
            "Operating System: Ubuntu 22.04.4 LTS",
            "Kernel: Linux 6.8.0-45-generic",
            "Architecture: x86-64",
        ],
        bad=["uname: command not found"],
    ),
    _e(
        lin=["vmstat 1 3", "iostat -x 1 2"],
        ok=[
            "procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----",
            " r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st",
            " 1  0 524288 224144  84212 431824   0    0    12    84  421 1204 12  3 84  1  0",
            "Linux 6.8.0-45-generic (host)  11/06/2026  _x86_64_  (16 CPU)",
            "avg-cpu:  %user   %nice %system %iowait  %steal   %idle",
            "           12.41    0.00    3.12    0.42    0.00   84.05",
        ],
        bad=["vmstat: command not found"],
        deny=["iostat: cannot open /proc/diskstats: Permission denied"],
    ),
    _e(
        lin=["lsof -i :{port}", "lsof -p 4120"],
        ok=[
            "COMMAND  PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME",
            "python3 4120  dev    7u  IPv4 184231      0t0  TCP *:{port} (LISTEN)",
            "python3 4120  dev    8u  IPv4 184232      0t0  TCP 10.0.0.14:55212->10.0.0.5:5432 (ESTABLISHED)",
        ],
        bad=["lsof: no process ID specified"],
        deny=["lsof: WARNING: can't stat() fuse.gvfsd-fuse file system"],
    ),
    _e(
        lin=["lsblk", "blkid", "lsblk -f"],
        ok=[
            "NAME        MAJ:MIN RM   SIZE RO TYPE MOUNTPOINTS",
            "nvme0n1     259:0    0 476.9G  0 disk",
            "├─nvme0n1p1 259:1    0   512M  0 part /boot/efi",
            "├─nvme0n1p2 259:2    0   468G  0 part /",
            "└─nvme0n1p3 259:3    0   8.4G  0 part [SWAP]",
            "/dev/nvme0n1p2: UUID=\"4a1b2c3d-...\" TYPE=\"ext4\" PARTUUID=\"...\"",
        ],
        bad=["lsblk: /dev/sdb: No such file or directory"],
        deny=["blkid: cannot open /dev/nvme0n1: Permission denied"],
    ),
    _e(
        lin=["mount | column -t", "sudo mount /dev/sdb1 /mnt/data", "sudo umount /mnt/data"],
        ok=[
            "/dev/nvme0n1p2 on / type ext4 (rw,relatime,errors=remount-ro)",
            "proc on /proc type proc (rw,nosuid,nodev,noexec,relatime)",
            "mounted /dev/sdb1 on /mnt/data",
        ],
        bad=["mount: /mnt/data: special device /dev/sdb1 does not exist.",
             "umount: /mnt/data: not mounted."],
        deny=["mount: /mnt/data: permission denied.", "umount: /mnt/data: must be superuser."],
    ),
    _e(
        lin=["sudo fdisk -l", "sudo parted -l"],
        ok=[
            "Disk /dev/nvme0n1: 476.94 GiB, 512110190592 bytes, 1000215216 sectors",
            "Disk model: Samsung SSD 970 EVO Plus 500GB",
            "Units: sectors of 1 * 512 = 512 bytes",
            "Device            Start        End    Sectors   Size Type",
            "/dev/nvme0n1p1     2048    1050623    1048576   512M EFI System",
            "/dev/nvme0n1p2  1050624 1000214527  999163904 476.4G Linux filesystem",
        ],
        bad=["fdisk: cannot open /dev/sdb: No such file or directory"],
        deny=["fdisk: cannot open /dev/nvme0n1: Permission denied"],
    ),

    # ---------------- permissions ----------------
    _e(
        lin=["chmod 600 {file}", "chmod +x deploy.sh", "chmod -R 755 src/"],
        win=["icacls {file} /grant dev:(R,W)"],
        ok=["mode of '{file}' changed to 0600 (rw-------)"],
        bad=["chmod: cannot access '{file}': No such file or directory"],
        deny=["chmod: changing permissions of '{file}': Operation not permitted"],
    ),
    _e(
        lin=["sudo chown {user}:{group} {file}", "sudo chgrp {group} {file}"],
        ok=["changed ownership of '{file}' to {user}:{group}"],
        bad=["chown: cannot access '{file}': No such file or directory"],
        deny=["chown: changing ownership of '{file}': Operation not permitted"],
    ),
    _e(
        lin=["umask", "getfacl {file}", "setfacl -m u:{user}:rw {file}"],
        ok=["0022", "# file: {file}", "# owner: dev", "# group: dev",
            "user::rw-", "group::r--", "other::r--", "ACL updated for '{file}'"],
        bad=["getfacl: {file}: No such file or directory"],
        deny=["setfacl: {file}: Operation not permitted"],
    ),

    # ---------------- processes ----------------
    _e(
        lin=["ps aux", "ps -ef | grep python"],
        win=["tasklist", "tasklist /FI \"IMAGENAME eq python.exe\""],
        ok=[
            "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND",
            "root         1  0.0  0.1 167432 11204 ?        Ss   Nov05   0:04 /sbin/init",
            "dev       4120 42.1  1.9 8621440 1245184 pts/0 Rl+ 10:14   3:12 python3 train.py",
            "dev       4188  0.0  0.0  12144  3840 pts/0    S+  10:14   0:00 grep --color=auto python",
        ],
        bad=["ps: error: unknown option -- z"],
        deny=["ps: /proc/1/task: Permission denied"],
    ),
    _e(
        lin=["kill -TERM 4120", "kill -9 4120"],
        win=["taskkill /PID 4120", "taskkill /F /IM python.exe"],
        ok=["process 4120 terminated"],
        bad=["kill: (4120): No such process", "ERROR: The process \"4120\" not found."],
        deny=["kill: (4120): Operation not permitted", "Access is denied."],
    ),
    _e(
        lin=["pgrep -af python", "killall -TERM worker"],
        ok=["4120 python3 train.py", "4188 python3 worker.py", "terminated 2 processes"],
        bad=["pgrep: no matching processes found", "worker: no process found"],
        deny=["killall: Operation not permitted"],
    ),
    _e(
        lin=["nice -n 10 python train.py", "renice -n 5 -p 4120"],
        ok=["4120 (process ID) old priority 0, new priority 5"],
        bad=["renice: failed to get priority for 9999 (No such process)"],
        deny=["renice: failed to set priority for 4120 (Operation not permitted)"],
    ),
    _e(
        lin=["nohup python worker.py > worker.log 2>&1 &", "jobs -l"],
        ok=["[1] 4120", "[1]+ 4120 Running                 python worker.py"],
        bad=["bash: nohup: command not found"],
        deny=["bash: worker.log: Permission denied"],
    ),

    # ---------------- terminal / shell ----------------
    _e(
        lin=["clear"], win=["cls"],
        ok=["terminal cleared"],
        bad=["clear: TERM environment variable not set."],
        deny=["clear: cannot write to terminal: Permission denied"],
    ),
    _e(
        lin=["tmux ls", "tmux new -s build -d", "tmux attach -t build"],
        ok=["build: 1 windows (created Fri Nov  6 10:14:22 2026)",
            "session 'build' created", "attached to session build"],
        bad=["no server running on /tmp/tmux-1000/default"],
        deny=["error connecting to /tmp/tmux-1000/default (Permission denied)"],
    ),
    _e(
        lin=["screen -ls", "screen -S build -dm"],
        ok=["There is a screen on:", "\t4120.build\t(Detached)",
            "1 Socket in /run/screen/S-dev."],
        bad=["No Sockets found in /run/screen/S-dev."],
        deny=["screen: /run/screen/S-dev: Permission denied"],
    ),
    _e(
        lin=["history | tail -n 5", "alias ll='ls -alF'", "export CUDA_VISIBLE_DEVICES=0"],
        ok=["  512  git status", "  513  pytest -q", "  514  history | tail -n 5",
            "alias ll='ls -alF'", "CUDA_VISIBLE_DEVICES=0"],
        bad=["bash: alias: invalid alias name"],
    ),
    _e(
        lin=["source venv/bin/activate", "source ~/.bashrc"],
        ok=["(venv) activated", "environment loaded"],
        bad=["bash: venv/bin/activate: No such file or directory"],
        deny=["bash: /home/dev/.bashrc: Permission denied"],
    ),
    _e(
        lin=["crontab -l", "sudo systemctl list-timers --all"],
        ok=["*/5 * * * * /usr/bin/python3 /opt/app/sync.py >> /var/log/sync.log 2>&1",
            "NEXT                         LEFT     LAST                         PASSED UNIT",
            "Fri 2026-11-06 10:15:00 UTC  38s left Fri 2026-11-06 10:10:00 UTC  4min ago logrotate.timer"],
        bad=["no crontab for dev"],
        deny=["crontab: permission denied"],
    ),
    _e(
        lin=["date", "hostname -f", "timedatectl"],
        ok=["Fri Nov  6 10:14:22 UTC 2026", "{host}.corp.local",
            "               Local time: Fri 2026-11-06 10:14:22 UTC",
            "           Universal time: Fri 2026-11-06 10:14:22 UTC",
            "                 Time zone: UTC (UTC, +0000)",
            "System clock synchronized: yes"],
        bad=["date: invalid date 'tomorrow'"],
    ),
    _e(
        lin=["man ls | head -n 20", "man -k network"],
        ok=["LS(1)                       User Commands                      LS(1)",
            "NAME", "       ls - list directory contents", "SYNOPSIS",
            "       ls [OPTION]... [FILE]...",
            "network (7)          - introduction to networking"],
        bad=["No manual entry for ls"],
    ),
    _e(
        lin=["watch -n 2 'df -h'", "time python train.py --epochs 1"],
        ok=["Every 2.0s: df -h                    host: Fri Nov  6 10:14:22 2026",
            "Filesystem      Size  Used Avail Use% Mounted on",
            "/dev/nvme0n1p2  468G  389G   55G  88% /",
            "real    0m12.421s", "user    0m18.114s", "sys     0m2.031s"],
        bad=["watch: command not found"],
        deny=["watch: cannot execute: Permission denied"],
    ),

    # ---------------- package inspection ----------------
    _e(
        lin=["apt list --upgradable", "apt-cache policy {pkg}", "dpkg -l | head -n 20"],
        ok=[
            "Listing... Done",
            "{pkg}/jammy-updates 1.2.3 amd64 [upgradable from: 1.2.2]",
            "{pkg}:",
            "  Installed: 1.2.2",
            "  Candidate: 1.2.3",
            "  Version table:",
            "     1.2.3 500",
            "        500 http://deb.debian.org/debian jammy-updates/main amd64 Packages",
            "ii  {pkg}  1.2.2  amd64  utility package",
        ],
        bad=["E: Unable to locate package {pkg}"],
        deny=["E: Could not open lock file /var/lib/dpkg/lock-frontend - open (13: Permission denied)"],
    ),
    _e(
        lin=["rpm -qa | head -n 20", "rpm -qi {pkg}"],
        ok=["bash-5.1.8-6.el9.x86_64", "openssl-3.0.7-27.el9.x86_64",
            "Name        : {pkg}", "Version     : 1.2.3", "Release     : 1.el9",
            "Architecture: x86_64", "Install Date: Fri 06 Nov 2026 10:14:22 AM UTC"],
        bad=["package {pkg} is not installed"],
        deny=["error: cannot open Packages database: Permission denied"],
    ),
    _e(
        lin=["snap list", "flatpak list"],
        ok=["Name               Version        Rev    Tracking       Publisher   Notes",
            "core22             20240918       1655   latest/stable  canonical✓  base",
            "firefox            131.0.2-1      4793   latest/stable  mozilla✓    -",
            "org.gnome.Platform 46             12     flathub        flathub     system"],
        bad=["error: no snaps found"],
        deny=["error: cannot access snapd socket: Permission denied"],
    ),
    _e(
        lin=["dpkg -S /usr/bin/python3", "dpkg -L {pkg} | head"],
        ok=["python3-minimal: /usr/bin/python3", "/usr/lib/python3/dist-packages/{pkg}",
            "/usr/share/doc/{pkg}/copyright"],
        bad=["dpkg-query: no path found matching pattern /usr/bin/nope"],
        deny=["dpkg-query: error: unable to access the dpkg database: Permission denied"],
    ),

    # ---------------- compression ----------------
    _e(
        lin=["gzip -k app.log", "bzip2 -k dataset.csv", "xz -9 -k dump.sql",
             "zstd -19 -T0 archive.tar"],
        ok=["app.log:  82.4% -- replaced with app.log.gz",
            "dataset.csv: 71.2% -- replaced with dataset.csv.bz2",
            "dump.sql: 64.8% -- replaced with dump.sql.xz",
            "archive.tar: 18.42%   (12.4 MiB => 2.3 MiB, archive.tar.zst)"],
        bad=["gzip: {file}: No such file or directory"],
        deny=["gzip: {file}: Permission denied"],
    ),
    _e(
        lin=["unzip -l bundle.zip", "zip -r bundle.zip src/"],
        ok=["Archive:  bundle.zip", "  Length      Date    Time    Name",
            "---------  ---------- -----   ----", "     1204  2026-11-06 10:14   src/main.py",
            "---------                     -------", "     1204                     1 file",
            "  adding: src/main.py (deflated 42%)", "  adding: src/utils.py (deflated 38%)"],
        bad=["unzip:  cannot find or open bundle.zip"],
        deny=["zip I/O error: Permission denied"],
    ),
    _e(
        lin=["vim --version | head -n 5", "nano --version", "less --version | head -n 2",
             "vi {file}"],
        ok=[
            "VIM - Vi IMproved 9.1 (2024 Jan 02, compiled May 04 2024 10:11:04)",
            "Included patches: 1-16",
            "Compiled by team+vim@tracker.debian.org",
            "Huge version without GUI.  Features included (+) or not (-):",
            " GNU nano, version 7.2",
            " (C) 2023 the GNU nano team",
            "less 590 (GNU regular expressions)",
            "Copyright (C) 1984-2021 Mark Nudelman",
        ],
        bad=["vim: command not found", "nano: command not found"],
        deny=["vim: cannot read {file}: Permission denied"],
    ),
]

# --------------------------------------------------------------------------
# held-out complex command families (val/test only)
# --------------------------------------------------------------------------
_COMPLEX = [
    _e(
        lin=["grep -RIn --include='*.py' 'TODO' src/ | xargs -r wc -l",
             "grep -RIn 'deprecated' app/ | sort | uniq -c | sort -rn"],
        ok=["src/main.py:18:# TODO: retry policy", "src/io.py:64:# TODO: handle EOF",
            "src/tasks.py:122:# TODO: metrics", "  3 total",
            "     12 app/api.py:88: deprecated call", "      4 app/db.py:31: deprecated"],
        bad=["grep: src/: No such file or directory"],
        deny=["grep: app/: Permission denied"],
        complex=True,
    ),
    _e(
        lin=["find /var/log -type f -name '*.log' -mtime +7 -print -delete",
             "find /srv/app -type f -size +100M -printf '%s %p\n' | sort -rn | head"],
        ok=["/var/log/app.log", "/var/log/worker.log", "/var/log/sync.log",
            "104857600 /srv/app/data/archive.bin", "268435456 /srv/app/data/dump.sql"],
        bad=["find: '/var/log': No such file or directory"],
        deny=["find: '/srv/app': Permission denied"],
        complex=True,
    ),
    _e(
        lin=["awk -F, 'NR>1 {sum+=$4; n++} END {printf \"rows=%d avg=%.2f\n\", n, sum/n}' report.csv",
             "awk '$9 ~ /^5[0-9][0-9]$/ {print $1, $7, $9}' access.log | sort | uniq -c"],
        ok=["rows=48213 avg=127.43", "    218 10.0.0.14 /api/orders 503",
            "     94 10.0.0.22 /api/search 502"],
        bad=["awk: cannot open report.csv for reading (No such file or directory)"],
        deny=["awk: cannot open access.log for reading (Permission denied)"],
        complex=True,
    ),
    _e(
        lin=["sed -i.bak -E 's/(db|redis)_host=.*/\\1_host=10.0.0.14/' config/*.yaml",
             "sed -n '/^server:/,/^$/p' nginx.conf"],
        ok=["updated config/prod.yaml", "updated config/staging.yaml", "server {",
            "    listen 80;", "    server_name app.internal;", "}"],
        bad=["sed: can't read config/*.yaml: No such file or directory"],
        deny=["sed: couldn't open temporary file config/.sedAb12: Permission denied"],
        complex=True,
    ),
    _e(
        lin=["rsync -avc --delete --exclude='*.tmp' /srv/app/ backup:/srv/app/",
             "rsync -av --dry-run --itemize-changes /srv/app/ backup:/srv/app/"],
        ok=["sending incremental file list", "src/main.py", "src/worker.py",
            "sent 184,221 bytes  received 12,904 bytes  131,516.67 bytes/sec",
            "total size is 48,221,904  speedup is 244.55"],
        bad=["rsync: [sender] change_dir \"/srv/app\" failed: No such file or directory (2)"],
        deny=["rsync: [receiver] mkdir \"/srv/app\" failed: Permission denied (13)"],
        complex=True,
    ),
    _e(
        lin=["tar -czf release.tgz build/ && sha256sum release.tgz",
             "tar -xzf release.tgz -C /opt/app && ls -lh /opt/app/build"],
        ok=["build/", "build/app", "build/libapp.so", "build/config.yaml",
            "b1946ac92492d2347c6235b4d2611184  release.tgz"],
        bad=["tar: build: Cannot stat: No such file or directory"],
        deny=["tar: release.tgz: Cannot open: Permission denied"],
        complex=True,
    ),
    _e(
        lin=["git fetch origin && git rebase origin/develop",
             "git log --oneline --decorate -n 5"],
        ok=["Successfully rebased and updated refs/heads/feature/auth.",
            "a1b2c3d (HEAD -> feature/auth) fix token refresh",
            "d4e5f6a (origin/develop) merge billing"],
        bad=["CONFLICT (content): Merge conflict in src/auth.py",
             "error: could not apply a1b2c3d... refresh token"],
        deny=["fatal: unable to access 'https://github.com/org/app.git/': 403"],
        complex=True,
    ),
    _e(
        lin=["sudo apt-get update && sudo apt-get full-upgrade -y && sudo dpkg --configure -a",
             "sudo apt-get autoremove --purge -y"],
        ok=["Reading package lists... Done", "Calculating upgrade... Done",
            "The following packages will be upgraded:", "  libc6 libssl3 openssl",
            "Setting up libssl3:amd64 ...", "Processing triggers for libc-bin ...",
            "0 upgraded, 0 newly installed, 0 to remove and 0 not upgraded."],
        bad=["E: Failed to fetch http://deb.debian.org/debian/pool/main/o/openssl 404 Not Found",
             "E: Unable to fetch some archives"],
        deny=["E: Could not open lock file /var/lib/dpkg/lock-frontend - open (13: Permission denied)"],
        complex=True,
    ),
    _e(
        lin=["python -m pip wheel . -w dist/ --no-deps && python -m pip install --force-reinstall dist/app-*.whl",
             "python -m pip check"],
        ok=["Processing ./", "Building wheel for app (pyproject.toml): finished with status 'done'",
            "Successfully built app-2.4.1-py3-none-any.whl",
            "Successfully installed app-2.4.1", "No broken requirements found."],
        bad=["ERROR: Failed building wheel for app", "error: subprocess-exited-with-error"],
        deny=["ERROR: Could not install packages due to an OSError: [Errno 13] Permission denied: '/usr/lib/python3.10/site-packages'"],
        complex=True,
    ),
    _e(
        lin=["conda env create -f environment.yml && conda run -n app python -c 'import app'",
             "conda list -n app | tail -n 5"],
        ok=["Collecting package metadata (repodata.json): done", "Solving environment: done",
            "Preparing transaction: done", "Executing transaction: done",
            "app 2.4.1 pypi_0 pypi", "numpy 2.2.2 py310h1f7b6c4_0 conda-forge"],
        bad=["LibMambaUnsatisfiableError: Encountered problems while solving",
             "package app requires python >=3.11, but none of the providers can be installed"],
        deny=["EnvironmentNotWritableError: The current user does not have write permissions"],
        complex=True,
    ),
    _e(
        lin=["cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build --output-on-failure",
             "cmake --install build --prefix /opt/app"],
        ok=["-- Configuring done", "-- Generating done", "[ 50%] Building CXX object CMakeFiles/app.dir/src/main.cpp.o",
            "[100%] Linking CXX executable app", "Test project build", "1/1 Test #1: smoke .... Passed",
            "100% tests passed, 0 tests failed out of 1"],
        bad=["CMake Error at CMakeLists.txt:12 (find_package): Could not find OpenCV",
             "make: *** [Makefile:24: all] Error 1"],
        deny=["CMake Error: could not create directory /opt/app: Permission denied"],
        complex=True,
    ),
    _e(
        lin=["cargo test --release --all-features", "cargo clippy --all-targets -- -D warnings"],
        ok=["   Compiling app v0.1.0 (/home/dev/proj)", "    Finished release [optimized] target(s)",
            "     Running unittests src/main.rs", "test result: ok. 42 passed; 0 failed; 0 ignored",
            "    Checking app v0.1.0", "    Finished dev [unoptimized] target(s)"],
        bad=["thread 'test_parse' panicked at src/parser.rs:42: assertion failed",
             "test result: FAILED. 41 passed; 1 failed"],
        deny=["error: failed to write to /usr/local/cargo/registry: Permission denied"],
        complex=True,
    ),
    _e(
        lin=["go test ./... -race -count=1", "go build -trimpath -o bin/app ./cmd/server"],
        ok=["ok   app/cmd/server 2.418s", "ok   app/internal/parser 0.812s",
            "ok   app/internal/store 1.204s", "built bin/app"],
        bad=["--- FAIL: TestParsePayload (0.01s)", "    parser_test.go:42: got error, want nil",
             "FAIL app/internal/parser 0.014s"],
        deny=["go: writing stat cache: open /home/dev/.cache/go-build: Permission denied"],
        complex=True,
    ),
    _e(
        lin=["python -m pytest --cov=app --cov-report=term-missing -q",
             "python -m pytest tests/integration -q --maxfail=1"],
        ok=["128 passed in 12.48s", "Name                 Stmts   Miss  Cover   Missing",
            "app/api.py             184     12    93%   88-92", "TOTAL                  964     71    93%",
            "12 passed in 4.02s"],
        bad=["FAILED tests/test_api.py::test_parse_response - AssertionError: assert 'error' == 'ok'",
             "1 failed, 127 passed in 12.48s"],
        deny=["ERROR: could not open coverage file .coverage: Permission denied"],
        complex=True,
    ),
    _e(
        lin=["docker compose -f compose.prod.yml build --pull && docker compose up -d",
             "docker compose ps --format json"],
        ok=["[+] Building 48.2s (12/12) FINISHED", " => [4/5] RUN pip install -r requirements.txt",
            " => => naming to docker.io/library/app:latest", "[+] Running 3/3",
            " ✔ Network prod_default  Created", " ✔ Container prod-app-1  Started"],
        bad=["ERROR: Service 'app' failed to build: process '/bin/sh -c pip install' returned non-zero",
             "Error response from daemon: no space left on device"],
        deny=["Error response from daemon: pull access denied for registry/app, may require 'docker login'"],
        complex=True,
    ),
    _e(
        lin=["kubectl -n prod rollout status deployment/app --timeout=120s && kubectl -n prod logs deploy/app --tail=20",
             "kubectl -n prod get pods -o wide"],
        ok=["deployment \"app\" successfully rolled out", "2026-11-06T10:14:22Z INFO starting worker",
            "2026-11-06T10:14:23Z INFO connected to database", "NAME                   READY   STATUS    RESTARTS",
            "app-7d9f8c6b5-2xk4p   1/1     Running   0"],
        bad=["error: deployment \"app\" exceeded its progress deadline",
             "Error from server (NotFound): deployments.apps \"app\" not found"],
        deny=["Error from server (Forbidden): deployments.apps \"app\" is forbidden"],
        complex=True,
    ),
    _e(
        lin=["terraform init -input=false && terraform plan -out=tfplan",
             "terraform show -no-color tfplan | head -n 30"],
        ok=["Initializing provider plugins...", "Terraform has been successfully initialized!",
            "Plan: 3 to add, 0 to change, 0 to destroy.",
            "  + aws_instance.web", "      + instance_type = \"t3.medium\""],
        bad=["Error: Error acquiring the state lock", "Lock Info: ID: 8842"],
        deny=["Error: Insufficient permissions to create resources"],
        complex=True,
    ),
    _e(
        lin=["psql \"$DATABASE_URL\" -v ON_ERROR_STOP=1 -f migrations/0042_add_index.sql",
             "psql \"$DATABASE_URL\" -c 'SELECT count(*) FROM orders WHERE created_at > now() - interval \\'1 day\\';"],
        ok=["CREATE INDEX", "ANALYZE", " count ", "-------", " 18422", "(1 row)"],
        bad=["ERROR:  relation \"orders\" does not exist", "LINE 1: SELECT count(*) FROM orders"],
        deny=["ERROR:  permission denied for table orders"],
        complex=True,
    ),
    _e(
        lin=["ffprobe -v error -show_entries stream=codec_name,width,height -of csv=p=0 input.mp4 && ffmpeg -i input.mp4 -vf scale=1280:-2 -c:v libx264 -crf 23 output.mp4",
             "ffmpeg -i output.mp4 -f null - 2>&1 | tail -n 3"],
        ok=["h264,1920,1080", "aac,0,0", "frame= 2504 fps=30 q=23.0 size=11264kB time=00:01:23.45",
            "video:10932kB audio:1284kB muxing overhead: 0.421%", "frame= 2504 fps=30 speed=1.00x"],
        bad=["input.mp4: Invalid data found when processing input", "Conversion failed!"],
        deny=["Could not open file : output.mp4", "Permission denied"],
        complex=True,
    ),
    _e(
        lin=["sudo systemctl restart app.service && journalctl -u app.service -n 20 --no-pager",
             "systemctl status app.service --no-pager"],
        ok=["● app.service - Application API", "     Active: active (running) since Fri 2026-11-06 10:14:22 UTC",
            "   Main PID: 4120 (python3)", "Started Application API."],
        bad=["Job for app.service failed because the control process exited with error code.",
             "See \"systemctl status app.service\" for details."],
        deny=["Failed to restart app.service: Access denied", "Interactive authentication required."],
        complex=True,
    ),
]
CATALOG.extend(_COMPLEX)

_WIN_CATALOG: List[Dict] = [
    _e(
        win=["ver", "systeminfo"],
        ok=["Microsoft Windows [Version 10.0.22631.4169]",
            "Host Name:                 WORKSTATION",
            "OS Name:                   Microsoft Windows 11 Pro",
            "OS Version:                10.0.22631 N/A Build 22631",
            "System Type:               x64-based PC",
            "Total Physical Memory:     32,614 MB",
            "Available Physical Memory: 18,204 MB"],
        bad=["'ver' is not recognized as an internal or external command."],
    ),
    _e(
        win=["dir C:\\proj", "cd /d C:\\proj && dir"],
        ok=[" Volume in drive C has no label.", " Volume Serial Number is 4A2B-9C1D",
            " Directory of C:\\proj", "",
            "11/06/2026  10:14 AM    <DIR>          .",
            "11/06/2026  10:14 AM    <DIR>          ..",
            "11/06/2026  10:14 AM             1,204 main.py",
            "               1 File(s)          1,204 bytes"],
        bad=["File Not Found"],
        deny=["Access is denied."],
    ),
    _e(
        win=["mkdir C:\\proj\\build", "copy main.py D:\\backup\\",
             "move old.log D:\\archive\\", "del /Q temp.txt"],
        ok=["1 file(s) copied.", "1 file(s) moved.", "deleted C:\\proj\\temp.txt"],
        bad=["The system cannot find the file specified."],
        deny=["Access is denied."],
    ),
    _e(
        win=["type C:\\proj\\main.py", "where python",
             "findstr /N \"error\" app.log"],
        ok=["import os", "C:\\Users\\dev\\AppData\\Local\\Programs\\Python\\Python310\\python.exe",
            "42:ERROR connection reset by peer"],
        bad=["The system cannot find the path specified."],
        deny=["Access is denied."],
    ),
    _e(
        win=["tasklist", "taskkill /PID 4120", "taskkill /F /IM python.exe"],
        ok=["Image Name                     PID Session Name        Session#    Mem Usage",
            "========================= ======== ================ =========== ============",
            "python.exe                    4120 Console                    1     82,144 K",
            "SUCCESS: Sent termination signal to the process with PID 4120."],
        bad=["ERROR: The process \"4120\" not found."],
        deny=["ERROR: Access is denied."],
    ),
    _e(
        win=["netstat -ano", "nslookup example.com", "tracert example.com"],
        ok=["Active Connections", "  Proto  Local Address          Foreign Address        State           PID",
            "  TCP    0.0.0.0:80             0.0.0.0:0              LISTENING       4120",
            "Server:  dns.corp.local", "Address:  10.0.0.1", "Name:    example.com",
            "Address:  203.0.113.7"],
        bad=["*** dns.corp.local can't find example.com: Non-existent domain"],
    ),
    _e(
        win=["icacls C:\\proj\\main.py", "takeown /F C:\\proj\\main.py"],
        ok=["C:\\proj\\main.py BUILTIN\\Users:(I)(RX)", "              NT AUTHORITY\\SYSTEM:(I)(F)",
            "Successfully processed 1 files; Failed processing 0 files"],
        bad=["The system cannot find the file specified."],
        deny=["Access is denied.", "ERROR: Access is denied."],
    ),
    _e(
        win=["whoami", "net user", "net localgroup administrators"],
        ok=["workstation\\dev", "User accounts for \\\\WORKSTATION", "--------------------------------",
            "Administrator            DefaultAccount           dev",
            "Guest                    WDAGUtilityAccount", "Alias name     administrators",
            "Members", "-------------------------------------------------------------------------------",
            "Administrator", "dev", "The command completed successfully."],
        bad=["The user name could not be found."],
        deny=["System error 5 has occurred.", "Access is denied."],
    ),
    _e(
        win=["sc query", "sc query {svc}"],
        ok=["SERVICE_NAME: {svc}", "        TYPE               : 10  WIN32_OWN_PROCESS",
            "        STATE              : 4  RUNNING",
            "        WIN32_EXIT_CODE    : 0  (0x0)"],
        bad=["[SC] EnumQueryServicesStatus:OpenService FAILED 1060:", "The specified service does not exist as an installed service."],
        deny=["[SC] OpenSCManager FAILED 5:", "Access is denied."],
    ),
    _e(
        win=["reg query HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion",
             "reg add HKCU\\Software\\App /v Mode /t REG_SZ /d production"],
        ok=["HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion",
            "    ProgramFilesDir    REG_SZ    C:\\Program Files",
            "The operation completed successfully."],
        bad=["ERROR: The system was unable to find the specified registry key or value."],
        deny=["ERROR: Access is denied."],
    ),
    _e(
        win=["wmic logicaldisk get caption,freespace,size",
             "wmic os get freephysicalmemory,totalvisiblememorysize"],
        ok=["Caption  FreeSpace     Size", "C:       58234134528   512110190592",
            "FreePhysicalMemory  TotalVisibleMemorySize", "18432000            33423360"],
        bad=["'wmic' is not recognized as an internal or external command."],
    ),
    _e(
        win=["powershell -c \"Get-Process python\"",
             "powershell -c \"Get-Service | Where-Object Status -eq 'Running'\"",
             "powershell -c \"Get-Volume\""],
        ok=["Handles  NPM(K)    PM(K)      WS(K)     CPU(s)     Id  SI ProcessName",
            "    412      28    82144      92160       3.12   4120   1 python",
            "Status   Name               DisplayName", "------   ----               -----------",
            "Running  Appinfo            Application Information",
            "DriveLetter FriendlyName FileSystemType HealthStatus SizeRemaining      Size",
            "C           Windows      NTFS           Healthy           55.2 GB 468.0 GB"],
        bad=["Get-Process : Cannot find a process with the name \"python\"."],
        deny=["Get-Service : Access is denied."],
    ),
    _e(
        win=["diskpart", "chkdsk C: /f", "format D: /FS:NTFS /Q"],
        ok=["Microsoft DiskPart version 10.0.22621.1",
            "DISKPART> list disk", "  Disk ###  Status         Size     Free     Dyn  Gpt",
            "  --------  -------------  -------  -------  ---  ---", "  Disk 0    Online          476 GB      0 B        *",
            "The type of the file system is NTFS.",
            "Stage 1: Examining basic file system structure ...",
            "Windows has scanned the file system and found no problems.",
            "Initializing the file allocation table (FAT)...", "Format completed successfully."],
        bad=["The system cannot find the drive specified."],
        deny=["Access is denied.", "You do not have sufficient permissions."],
    ),
]


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def _render(items, ctx, label="neutral") -> List[Line]:
    out = []
    for item in items:
        if isinstance(item, tuple):
            text, lab = item
        else:
            text, lab = item, label
        try:
            text = text.format(**ctx)
        except (KeyError, IndexError, ValueError):
            pass
        out.append((text, lab))
    return out


def gen_commands(rng: random.Random, platform: str, status: str, cfg):
    split = getattr(cfg, "_split", "train")
    base = _WIN_CATALOG if platform == "windows" else CATALOG
    complex_pool = [c for c in base if c.get("complex")]
    plain_pool = [c for c in base if not c.get("complex")]
    if split == "train":
        pool = plain_pool
    elif split == "val":
        pool = complex_pool if (complex_pool and rng.random() < 0.5) else plain_pool
    else:
        pool = complex_pool or plain_pool
    spec = _p(rng, pool)
    cmds = spec["win"] if platform == "windows" else spec["lin"]
    cmd = _p(rng, cmds)
    ctx = _ctx(rng, platform)
    try:
        cmd = cmd.format(**ctx)
    except (KeyError, IndexError, ValueError):
        pass
    head = [_prompt(rng, platform, str(ctx["dir"]) if platform == "windows" else "~", cmd)]

    ok_lines = _render(spec["ok"], ctx)
    if status == "success":
        if not ok_lines:
            ok_lines = [("command completed successfully", "success")]
        else:
            # last line of normal output is the status marker
            text, _ = ok_lines[-1]
            ok_lines[-1] = (text, "success")
        return _join(head, ok_lines, [_footer(rng, platform, 0)]), 0

    if status == "fail":
        bad = _render(spec["bad"], ctx, label="fail")
        if not bad:
            bad = [("command failed: No such file or directory", "fail")]
        return _join(head, bad, [_footer(rng, platform, 1)]), 1

    if status == "block":
        deny = _render(spec["deny"], ctx, label="block")
        if not deny:
            deny = [("Operation not permitted: access denied by policy", "block")]
        return _join(head, deny, [_footer(rng, platform, 126)]), 126

    # unknown: partial output, no terminal marker
    if ok_lines:
        cut = max(1, len(ok_lines) // 2)
        partial = [(text, "neutral") for text, _ in ok_lines[:cut]]
    else:
        partial = [("partial output", "neutral")]
    return _join(head, partial), None


COMMAND_GENERATORS = {"commands": gen_commands}
