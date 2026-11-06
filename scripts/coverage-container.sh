#!/usr/bin/env bash
# Measure test coverage in a Linux container, as CI does (Linux, Python 3.10).
#
# Coverage cannot be measured on macOS arm64 (torch + coverage.py collection
# failure, then a PyO3 double initialisation). This runs the required suite
# under coverage inside a disposable Linux container instead. It does not change
# what CI measures: CI remains the authoritative number.
#
# The working tree is streamed into the container rather than bind-mounted, so
# it works with any Docker runtime (Colima and Docker Desktop do not share every
# host path). The virtual environment and the uv cache live in the container and
# are discarded with it (--rm).
#
# Usage: scripts/coverage-container.sh [extra pytest arguments]
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="python:3.10-bookworm"
uv_version="0.10.9"

tar -C "$repo_root" \
  --exclude=./.git --exclude=./.venv --exclude=./wiki --exclude=./coverage \
  --exclude=./.pytest_cache --exclude='__pycache__' --exclude=./.coverage \
  -cf - . |
  docker run --rm -i --name "mistify-coverage-$$" \
    --env UV_PROJECT_ENVIRONMENT=/tmp/venv \
    --env UV_NO_PROGRESS=1 \
    --env LOAD_MODELS_ON_STARTUP=false \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env HF_HUB_DISABLE_TELEMETRY=1 \
    "$image" \
    sh -c 'mkdir /src && tar -x -C /src && cd /src \
      && pip install --quiet --disable-pip-version-check "uv==$0" \
      && uv sync --frozen --python 3.10 \
      && uv run --frozen --python 3.10 pytest -m "not slow" -q -p no:cacheprovider \
           --cov=src --cov-report=term-missing "$@"' "$uv_version" "$@"
