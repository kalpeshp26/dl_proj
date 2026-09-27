#!/bin/sh
# Cloud entrypoint — initialises DB then launches FastAPI
set -e

echo "=== DeepRetail Cloud Startup ==="

# Seed the database from inventory.csv if DB doesn't already exist
if [ ! -f "deepretail.db" ]; then
  echo "Initialising database from inventory.csv..."
  python - <<'PY'
import sys
sys.path.insert(0, ".")
from src.backend.db import init_db
init_db()
print("Database ready.")
PY
fi

echo "Starting FastAPI backend on port ${PORT:-8000}..."
exec uvicorn src.backend.main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --workers 1 \
  --log-level info
