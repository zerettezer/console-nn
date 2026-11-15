# Real Linux/Windows error sessions

Supplied concrete error families. No `[exit code]` lines.

| file | platform | status | command |
|---|---|---|---|
| [r01-make_openssl_missing](r01-make_openssl_missing.log) | linux | fail | `make` |
| [r02-nvidia_nvml_mismatch](r02-nvidia_nvml_mismatch.log) | linux | fail | `watch nvidia-smi` |
| [r03-apt_lock](r03-apt_lock.log) | linux | block | `apt install nginx` |
| [r04-apt_unable_locate](r04-apt_unable_locate.log) | linux | fail | `sudo apt install nginx` |
| [r05-docker_permission](r05-docker_permission.log) | linux | block | `docker run hello-world` |
| [r06-git_push_rejected](r06-git_push_rejected.log) | linux | fail | `git push` |
| [r07-npm_eresolve_react](r07-npm_eresolve_react.log) | linux | fail | `npm install` |
| [r08-python_flask_missing](r08-python_flask_missing.log) | linux | fail | `python3 app.py` |
| [r09-pytest_assertion](r09-pytest_assertion.log) | linux | fail | `pytest` |
| [r10-cargo_borrow_moved](r10-cargo_borrow_moved.log) | linux | fail | `cargo build` |
| [r11-ssh_host_key_changed](r11-ssh_host_key_changed.log) | linux | block | `ssh user@192.168.1.10` |
| [r12-systemctl_myapp_failed](r12-systemctl_myapp_failed.log) | linux | fail | `systemctl start myapp` |
| [r13-kubectl_validation](r13-kubectl_validation.log) | linux | fail | `kubectl apply -f deploy.yaml` |
| [r14-terraform_lock](r14-terraform_lock.log) | linux | fail | `terraform apply` |
| [r15-curl_ssl_cert](r15-curl_ssl_cert.log) | linux | fail | `curl https://example.com` |
| [r16-msbuild_cs0103](r16-msbuild_cs0103.log) | windows | fail | `msbuild App.sln` |
| [r17-powershell_execution_policy](r17-powershell_execution_policy.log) | windows | block | `.\script.ps1` |
| [r18-windows_pip_missing](r18-windows_pip_missing.log) | windows | fail | `pip install numpy` |
| [r19-npm_openssl_unsupported](r19-npm_openssl_unsupported.log) | windows | fail | `npm run build` |
| [r20-windows_access_denied_hosts](r20-windows_access_denied_hosts.log) | windows | block | `del C:\Windows\System32\drivers\etc\hosts` |
| [r21-windows_net_start_access](r21-windows_net_start_access.log) | windows | block | `net start MySQL` |
| [r22-windows_get_content_missing](r22-windows_get_content_missing.log) | windows | fail | `Get-Content C:\logs\app.log` |
| [r23-winget_msstore_missing](r23-winget_msstore_missing.log) | windows | fail | `winget install Git.Git` |
