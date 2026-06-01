"""Phase 3.8b: router_runs.jsonl writer (schema v1).

Records are written to a *separate* file from ``distill_runs.jsonl`` to
keep schemas decoupled — distill records describe a single LLM call's
shape, router records describe a multi-model orchestration's shape. The
two are joined later (offline) on ``(theme, started_at)`` if needed.
"""

from __future__ import annotations

import json
import socket
import subprocess
from pathlib import Path
from typing import Any

from .strategies import RouterResult


ROUTER_SCHEMA_VERSION = 1


def _safe_git_rev(repo_dir: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir) if repo_dir else None,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _safe_hostname() -> str | None:
    try:
        return socket.gethostname()
    except OSError:
        return None


def build_router_record(
    *,
    result: RouterResult,
    theme: str,
    options: dict[str, Any] | None,
    repo_dir: Path | None = None,
) -> dict[str, Any]:
    """Build the dict that will become one line in router_runs.jsonl.

    ``options`` is the merged runtime options (eg ``{"num_thread": 6}``)
    snapshotted at call time so the record is reproducible.
    """
    proposer = result.proposer_output
    critic = result.critic_output
    return {
        "schema_version": ROUTER_SCHEMA_VERSION,
        "strategy_name": result.strategy_name,
        "theme": theme,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "parallel_wall_sec": round(result.parallel_wall_sec, 3),
        "outputs": [
            {
                "role": proposer.role,
                "model_id": proposer.model_id,
                "prompt_eval_count": proposer.prompt_eval_count,
                "eval_count": proposer.eval_count,
                "eval_duration_ns": proposer.eval_duration_ns,
                "total_duration_ns": proposer.total_duration_ns,
                "text_len_chars": len(proposer.text),
            },
            {
                "role": critic.role,
                "model_id": critic.model_id,
                "prompt_eval_count": critic.prompt_eval_count,
                "eval_count": critic.eval_count,
                "eval_duration_ns": critic.eval_duration_ns,
                "total_duration_ns": critic.total_duration_ns,
                "text_len_chars": len(critic.text),
            },
        ],
        "critic_findings_count": len(result.critic_findings),
        "critic_findings": list(result.critic_findings),
        "options": dict(options or {}),
        "meta": {
            "git_commit": _safe_git_rev(repo_dir),
            "host": _safe_hostname(),
        },
    }


def append_router_record(path: Path, record: dict[str, Any]) -> None:
    """Append one record as a JSON line. Creates parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
