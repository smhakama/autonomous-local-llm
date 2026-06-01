#!/usr/bin/env bash
# Phase 3.8a-Mini: nightly experiment matrix entrypoint.
#
# Default matrix: experiments/matrices/phase_38a_num_thread_sweep.yaml.
# Override with MATRIX_YAML env var when invoking from systemd or crontab.
#
# Logs to $LLM_NIGHTLY_LOG (default ~/cron-llm-nightly.log). The runner writes
# its own per-stage logs under experiments/runs/<matrix_id>/<timestamp>/.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="${LLM_NIGHTLY_LOG:-$HOME/cron-llm-nightly.log}"
MATRIX_YAML="${MATRIX_YAML:-experiments/matrices/phase_38a_num_thread_sweep.yaml}"
PYTHON_BIN="${LLM_RUNNER_PYTHON:-$HOME/ai_agents_env/bin/python}"

cd "$REPO_DIR"

{
  echo
  echo "=== llm_nightly_experiment.sh started at $(date -Iseconds) ==="
  echo "matrix=$MATRIX_YAML python=$PYTHON_BIN repo=$REPO_DIR"
} >> "$LOG_FILE"

# Best-effort ollama precheck; runner.py also re-checks.
if command -v systemctl >/dev/null 2>&1; then
  if ! systemctl is-active --quiet ollama; then
    echo "ollama.service not active, skipping nightly matrix" >> "$LOG_FILE"
    exit 0
  fi
fi

exec "$PYTHON_BIN" experiments/runner.py --matrix "$MATRIX_YAML" >> "$LOG_FILE" 2>&1
