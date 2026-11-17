#!/usr/bin/env bash
# One-command local start (macOS / Linux): creates .venv on first run, then launches the app.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements-dev.txt
fi
exec .venv/bin/streamlit run app.py "$@"
