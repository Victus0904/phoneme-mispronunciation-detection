#!/usr/bin/env sh
# Pronunciation Coach -- start the local app (macOS/Linux).
cd "$(dirname "$0")"
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
exec "$PY" server.py "$@"
