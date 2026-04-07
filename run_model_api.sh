#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

VENV_PY="$ROOT_DIR/.venv/bin/python"

# Runtime defaults (override by exporting env vars before running)
export RUNTIME_DATA_DIR="${RUNTIME_DATA_DIR:-required_data}"
export ONLINE_RETRAIN_THRESHOLD="${ONLINE_RETRAIN_THRESHOLD:-3}"
export PORT="${PORT:-8001}"
export HOST="${HOST:-0.0.0.0}"

DEFAULT_MODEL_PATH="output/models_temp/hardcoded_shortlist_job_notebook/model.pt"
DEFAULT_GRAPH_PATH="output/graph_store/hardcoded_shortlist_job_notebook/graph.pt"
DEFAULT_MAPPINGS_PATH="output/graph_store/hardcoded_shortlist_job_notebook/mappings.pt"
if [[ -z "${RUNTIME_MODEL_PATH:-}" ]]; then
  export RUNTIME_MODEL_PATH="$DEFAULT_MODEL_PATH"
fi

echo "[PathFinder] Runtime data dir: $RUNTIME_DATA_DIR"
echo "[PathFinder] Runtime model path: $RUNTIME_MODEL_PATH"
echo "[PathFinder] Retrain threshold: $ONLINE_RETRAIN_THRESHOLD"
echo "[PathFinder] Serving on: http://$HOST:$PORT"

if [[ ! -d "$RUNTIME_DATA_DIR" ]]; then
  if [[ -f "$DEFAULT_GRAPH_PATH" && -f "$DEFAULT_MAPPINGS_PATH" ]]; then
    echo ""
    echo "WARNING: Data directory '$RUNTIME_DATA_DIR' not found."
    echo "Using existing graph snapshot artifacts instead:"
    echo "  - $DEFAULT_GRAPH_PATH"
    echo "  - $DEFAULT_MAPPINGS_PATH"
  else
    echo ""
    echo "ERROR: Data directory '$RUNTIME_DATA_DIR' not found."
    echo "Fix option 1: extract required_data.rar so the '$RUNTIME_DATA_DIR' folder exists."
    echo "Fix option 2: provide graph snapshot artifacts at:"
    echo "  - $DEFAULT_GRAPH_PATH"
    echo "  - $DEFAULT_MAPPINGS_PATH"
    exit 1
  fi
fi

if [[ ! -f "$RUNTIME_MODEL_PATH" ]]; then
  echo ""
  echo "WARNING: model file not found at '$RUNTIME_MODEL_PATH'."
  echo "Server will still start, but recommendation endpoints will return model-unavailable until weights are provided."
fi

if [[ -x "$VENV_PY" ]]; then
  exec "$VENV_PY" -m uvicorn api_server:app --host "$HOST" --port "$PORT" --reload
fi

if command -v uvicorn >/dev/null 2>&1; then
  exec uvicorn api_server:app --host "$HOST" --port "$PORT" --reload
fi

if command -v python >/dev/null 2>&1; then
  exec python -m uvicorn api_server:app --host "$HOST" --port "$PORT" --reload
fi

echo ""
echo "ERROR: uvicorn is not available in the current environment."
echo "Fix: source .venv/bin/activate && pip install -r requirements.txt"
exit 1
