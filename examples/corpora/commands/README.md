# Complex command examples

Held-out command families used only for validation/test generation.
Each file is one command session.

| # | status | command |
|---|---|---|
| [c01](c01.log) | success | `grep -RIn --include='*.py' 'TODO' src/ | xargs -r wc -l` |
| [c02](c02.log) | fail | `find /srv/app -type f -size +100M -printf '%s %p
' | sort -rn | head` |
| [c03](c03.log) | block | `awk '$9 ~ /^5[0-9][0-9]$/ {print $1, $7, $9}' access.log | sort | uniq -c` |
| [c04](c04.log) | success | `sed -i.bak -E 's/(db|redis)_host=.*/\1_host=10.0.0.14/' config/*.yaml` |
| [c05](c05.log) | fail | `rsync -av --dry-run --itemize-changes /srv/app/ backup:/srv/app/` |
| [c06](c06.log) | block | `tar -czf release.tgz build/ && sha256sum release.tgz` |
| [c07](c07.log) | success | `git fetch origin && git rebase origin/develop` |
| [c08](c08.log) | fail | `sudo apt-get autoremove --purge -y` |
| [c09](c09.log) | block | `python -m pip wheel . -w dist/ --no-deps && python -m pip install --force-reinstall dist/app-*.whl` |
| [c10](c10.log) | success | `conda list -n app | tail -n 5` |
| [c11](c11.log) | fail | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build --output-on-failure` |
| [c12](c12.log) | block | `cargo clippy --all-targets -- -D warnings` |
| [c13](c13.log) | success | `go test ./... -race -count=1` |
| [c14](c14.log) | fail | `python -m pytest tests/integration -q --maxfail=1` |
| [c15](c15.log) | block | `docker compose -f compose.prod.yml build --pull && docker compose up -d` |
| [c16](c16.log) | success | `kubectl -n prod rollout status deployment/app --timeout=120s && kubectl -n prod logs deploy/app --tail=20` |
| [c17](c17.log) | fail | `terraform init -input=false && terraform plan -out=tfplan` |
| [c18](c18.log) | block | `psql "$DATABASE_URL" -c 'SELECT count(*) FROM orders WHERE created_at > now() - interval \'1 day\';` |
| [c19](c19.log) | success | `ffmpeg -i output.mp4 -f null - 2>&1 | tail -n 3` |
| [c20](c20.log) | fail | `systemctl status app.service --no-pager` |
