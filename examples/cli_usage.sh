#!/usr/bin/env bash
set -euo pipefail

# Analyze the bundled examples
python -m console_nn predict -f examples/logs/02-pytest-failure.log --format brief
python -m console_nn predict -f examples/logs/02-pytest-failure.log --format png --out report.png

# Capture a live command
python -m console_nn predict --exec "python3 -c 'import sys; sys.exit(1)'" --format brief

# Validation and deployment checks
python -m console_nn validate --run artifacts/cpu --data data/mixed --split test --strict
python -m console_nn doctor --run artifacts/cpu --data data/real --strict
