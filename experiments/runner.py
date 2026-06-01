#!/usr/bin/env python3
"""Phase 3.8a-Mini: yaml-driven nightly experiment runner.

Reads a matrix yaml (e.g. experiments/matrices/phase_38a_num_thread_sweep.yaml),
runs each sweep entry as a subprocess against the configured bench script,
and writes per-stage logs + a summary.jsonl under
experiments/runs/<matrix_id>/<timestamp>/.

Safety:
- Requires ollama.service to be active before launching anything.
- Continues to the next sweep entry when one fails (does not abort the matrix).
- Never touches the bench's metrics file; the bench appends to it as usual.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "experiments" / "runs"
SCHEMA = 1


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _local_stamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit(f"PyYAML missing: {e}. Install with `pip install pyyaml`.")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _validate_matrix(matrix: dict[str, Any], matrix_path: Path) -> None:
    required_top = {"schema", "matrix_id", "bench", "sweep"}
    missing = required_top - matrix.keys()
    if missing:
        raise SystemExit(f"{matrix_path}: missing top-level keys: {missing}")
    if matrix["schema"] != SCHEMA:
        raise SystemExit(
            f"{matrix_path}: schema {matrix['schema']} != supported {SCHEMA}"
        )
    if not isinstance(matrix["sweep"], list) or not matrix["sweep"]:
        raise SystemExit(f"{matrix_path}: sweep must be a non-empty list")
    seen: set[str] = set()
    for i, entry in enumerate(matrix["sweep"]):
        if "config_id" not in entry:
            raise SystemExit(f"{matrix_path}: sweep[{i}] missing config_id")
        cid = entry["config_id"]
        if cid in seen:
            raise SystemExit(f"{matrix_path}: duplicate config_id={cid!r}")
        seen.add(cid)
        if "extra_args" in entry and not isinstance(entry["extra_args"], list):
            raise SystemExit(
                f"{matrix_path}: sweep[{i}].extra_args must be a list"
            )


def _check_ollama_active() -> bool:
    sysctl = shutil.which("systemctl")
    if sysctl is None:
        return True  # not a systemd host; trust the caller
    rc = subprocess.run(
        [sysctl, "is-active", "ollama"],
        capture_output=True,
        text=True,
    ).returncode
    return rc == 0


def _python_for_bench() -> str:
    venv_py = os.environ.get("LLM_RUNNER_PYTHON")
    if venv_py:
        return venv_py
    candidates = [
        Path.home() / "ai_agents_env" / "bin" / "python",
        Path(sys.executable),
    ]
    for p in candidates:
        if Path(p).exists():
            return str(p)
    return sys.executable


def _run_one(
    *,
    matrix: dict[str, Any],
    entry: dict[str, Any],
    out_dir: Path,
    python_bin: str,
) -> dict[str, Any]:
    config_id = entry["config_id"]
    bench_path = REPO_ROOT / matrix["bench"]
    if not bench_path.exists():
        return {
            "config_id": config_id,
            "rc": -1,
            "error": f"bench script not found: {bench_path}",
            "started_at": _utc_now_iso(),
            "finished_at": _utc_now_iso(),
            "duration_sec": 0,
        }
    cmd: list[str] = [python_bin, str(bench_path)]
    cmd.extend(str(a) for a in matrix.get("common_args", []))
    cmd.extend(str(a) for a in entry.get("extra_args", []))
    if "--config-id" not in cmd:
        cmd.extend(["--config-id", config_id])
    log_path = out_dir / f"{config_id}.log"
    started = time.monotonic()
    started_iso = _utc_now_iso()
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"# cmd: {shlex.join(cmd)}\n")
        logf.write(f"# started_at: {started_iso}\n")
        logf.flush()
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = round(time.monotonic() - started, 1)
    return {
        "config_id": config_id,
        "rc": proc.returncode,
        "cmd": cmd,
        "log_path": str(log_path.relative_to(REPO_ROOT)),
        "started_at": started_iso,
        "finished_at": _utc_now_iso(),
        "duration_sec": elapsed,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 3.8a-Mini matrix runner")
    ap.add_argument(
        "--matrix",
        required=True,
        type=Path,
        help="Path to matrix yaml (e.g. experiments/matrices/phase_38a_num_thread_sweep.yaml)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run, no execution",
    )
    ap.add_argument(
        "--skip-ollama-check",
        action="store_true",
        help="Skip `systemctl is-active ollama` precheck (for offline test)",
    )
    args = ap.parse_args(argv)

    matrix_path = args.matrix.resolve()
    if not matrix_path.exists():
        print(f"matrix yaml not found: {matrix_path}", file=sys.stderr)
        return 2
    matrix = _load_yaml(matrix_path)
    _validate_matrix(matrix, matrix_path)

    if not args.skip_ollama_check and not args.dry_run:
        if not _check_ollama_active():
            print(
                "ollama.service is not active; aborting matrix run.",
                file=sys.stderr,
            )
            return 3

    python_bin = _python_for_bench()
    matrix_id = matrix["matrix_id"]

    print(f"matrix_id={matrix_id} entries={len(matrix['sweep'])}")
    print(f"python_bin={python_bin}")
    if args.dry_run:
        for entry in matrix["sweep"]:
            cmd = [python_bin, str(REPO_ROOT / matrix["bench"])]
            cmd.extend(str(a) for a in matrix.get("common_args", []))
            cmd.extend(str(a) for a in entry.get("extra_args", []))
            if "--config-id" not in cmd:
                cmd.extend(["--config-id", entry["config_id"]])
            print(f"[dry-run] {entry['config_id']}: {shlex.join(cmd)}")
        return 0

    out_dir = RUNS_DIR / matrix_id / _local_stamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.jsonl"
    print(f"out_dir={out_dir}")

    fail_count = 0
    with summary_path.open("w", encoding="utf-8") as sf:
        for entry in matrix["sweep"]:
            print(f"[run] {entry['config_id']} ...", flush=True)
            result = _run_one(
                matrix=matrix,
                entry=entry,
                out_dir=out_dir,
                python_bin=python_bin,
            )
            if result["rc"] != 0:
                fail_count += 1
            sf.write(json.dumps(result, ensure_ascii=False) + "\n")
            sf.flush()
            print(
                f"[done] {entry['config_id']} rc={result['rc']} "
                f"dur={result['duration_sec']}s log={result.get('log_path')}",
                flush=True,
            )

    print(f"matrix done: summary={summary_path} fails={fail_count}")
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
