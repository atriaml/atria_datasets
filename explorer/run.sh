#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ ! -x .venv/bin/uvicorn ]]; then
  echo "Backend dependencies are missing. Run: bash explorer/setup.sh" >&2
  exit 1
fi
if [[ ! -d explorer/frontend/node_modules ]]; then
  echo "Frontend dependencies are missing. Run: bash explorer/setup.sh" >&2
  exit 1
fi

.venv/bin/uvicorn explorer.backend.app:app \
  --host 127.0.0.1 \
  --port 8000 \
  --reload \
  --reload-dir explorer/backend &
backend_pid=$!
trap 'kill "$backend_pid" 2>/dev/null || true' EXIT INT TERM

cd explorer/frontend
npm run dev
