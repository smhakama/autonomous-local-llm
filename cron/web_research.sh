#!/usr/bin/env bash
# Phase 2.1b daily web research batch
# Invoked by crontab; see ./README.md for setup.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="${WEB_RESEARCH_LOG:-$HOME/cron-web_research.log}"

cd "$REPO_DIR"

# Activate venv (adjust if your venv lives elsewhere)
# shellcheck disable=SC1091
source "$HOME/ai_agents_env/bin/activate"

exec python web_research.py \
    --themes-file themes.txt \
    --max-pages 4 \
    --lock-file /tmp/web_research.lock \
    >> "$LOG_FILE" 2>&1
