#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

uv pip install --python .venv/bin/python -r explorer/backend/requirements.txt
npm --prefix explorer/frontend install

echo "Explorer installed. Start it with: bash explorer/run.sh"
