#!/usr/bin/env python3
"""hotfix_loop.py — Phase A→C オーケストレーション ミニ実装

仕様書 §4 Autonomous Local LLM SDLC の最小実装:
  Phase A : codebase → bge-m3 → Qdrant (web research は今回 skip)
  Phase B : deepseek-r1:14b で plan → Aider (qwen2.5-coder:7b) で edit
  Phase C : pytest 実行 → 失敗時 14b に出力食わせて re-plan → green で git commit

実行:
  source ~/ai_agents_env/bin/activate
  python ~/hotfix_loop.py ~/aider_smoke \\
    "Tests in tests/test_hello.py are failing. Make all tests pass."
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

OLLAMA_URL = "http://127.0.0.1:11434"
QDRANT_HOST, QDRANT_PORT = "127.0.0.1", 6333
EMBED_MODEL = "bge-m3"
PLAN_MODEL = "deepseek-r1:14b"
EDIT_MODEL = "ollama_chat/qwen2.5-coder:7b"
AIDER_BIN = str(Path(sys.executable).parent / "aider")
PYTEST_BIN = str(Path(sys.executable).parent / "pytest")
VEC_DIM = 1024
CHUNK_LINES = 100
CHUNK_STRIDE = 75
KEEP_ALIVE = "1h"
MAX_ITER = 3
RETRIEVE_K = 5
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def log(phase: str, msg: str) -> None:
    print(f"[{phase}] {msg}", flush=True)


def embed_text(text: str) -> list[float]:
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text, "keep_alive": KEEP_ALIVE},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def iter_chunks(repo: Path):
    skip_parts = {".git", "__pycache__", ".pytest_cache", ".aider.tags.cache.v4"}
    for py in sorted(repo.rglob("*.py")):
        if any(p in skip_parts for p in py.parts):
            continue
        try:
            text = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        if not lines:
            continue
        rel = py.relative_to(repo)
        start = 0
        while start < len(lines):
            end = min(start + CHUNK_LINES, len(lines))
            chunk = "\n".join(lines[start:end]).strip()
            if chunk:
                yield {
                    "file": str(rel),
                    "start": start + 1,
                    "end": end,
                    "code": chunk,
                }
            if end >= len(lines):
                break
            start += CHUNK_STRIDE


def index_codebase(repo: Path, collection: str) -> int:
    qd = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    if qd.collection_exists(collection):
        qd.delete_collection(collection)
    qd.create_collection(
        collection,
        vectors_config=VectorParams(size=VEC_DIM, distance=Distance.COSINE),
    )
    points = []
    for c in iter_chunks(repo):
        v = embed_text(c["code"])
        points.append(PointStruct(id=str(uuid.uuid4()), vector=v, payload=c))
    if points:
        qd.upsert(collection, points=points)
    return len(points)


def retrieve(query: str, collection: str, k: int = RETRIEVE_K) -> list[dict]:
    qd = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    qv = embed_text(query)
    pts = qd.query_points(
        collection_name=collection, query=qv, limit=k
    ).points
    return [p.payload or {} for p in pts]


def plan_with_14b(task: str, context: list[dict], prior_error: str = "") -> str:
    ctx = "\n\n".join(
        f"--- {c['file']}:{c['start']}-{c['end']} ---\n{c['code']}" for c in context
    )
    prior = ""
    if prior_error:
        prior = (
            "\n## Previous attempt failed with this output:\n"
            f"```\n{prior_error}\n```\n"
            "Account for this in your new plan.\n"
        )
    prompt = f"""You are a software engineer planning a minimal code fix.

## Task
{task}

## Relevant code context (top retrieved chunks)
{ctx}
{prior}
## Instructions
Write a CONCRETE, MINIMAL plan: which file to edit and what specific change to make.
Be precise about the exact code change (e.g., 'change `return "Hello"` to `return f"Hello, {{name}}!"`').
Keep the plan to 1-3 sentences. Output the plan only, no preamble."""

    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": PLAN_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "keep_alive": KEEP_ALIVE,
            "options": {"num_predict": 2048, "temperature": 0.2},
        },
        timeout=900,
    )
    r.raise_for_status()
    content = r.json()["message"]["content"]
    content = THINK_RE.sub("", content).strip()
    return content


def implement_with_aider(plan: str, target_files: list[Path], repo: Path) -> int:
    env = os.environ.copy()
    env["OLLAMA_API_BASE"] = OLLAMA_URL
    env["OLLAMA_KEEP_ALIVE"] = KEEP_ALIVE
    cmd = [
        AIDER_BIN,
        "--model", EDIT_MODEL,
        "--message", plan,
        "--yes-always",
        "--no-stream",
        "--no-auto-commits",
        "--no-show-model-warnings",
        "--no-check-update",
        *[str(f.relative_to(repo)) for f in target_files],
    ]
    p = subprocess.run(cmd, env=env, cwd=str(repo), timeout=600)
    return p.returncode


def run_tests(repo: Path) -> tuple[bool, str]:
    p = subprocess.run(
        [PYTEST_BIN, "tests/", "-v", "--tb=short"],
        cwd=str(repo), capture_output=True, text=True, timeout=120,
    )
    output = (p.stdout or "") + (p.stderr or "")
    return p.returncode == 0, output


def git_commit(repo: Path, message: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=str(repo), check=True)


def main() -> int:
    if len(sys.argv) < 3:
        sys.exit(f"usage: {sys.argv[0]} <repo_path> <task_description>")
    repo = Path(sys.argv[1]).expanduser().resolve()
    task = sys.argv[2]
    if not (repo / ".git").exists():
        sys.exit(f"not a git repo: {repo}")

    collection = f"hotfix_{repo.name}_{int(time.time())}"
    log("Phase A", f"codebase 索引 → collection={collection}")
    t0 = time.time()
    n = index_codebase(repo, collection)
    log("Phase A", f"投入 {n} chunks ({time.time()-t0:.1f}s)")

    skip_parts = {".git", "__pycache__", ".pytest_cache", "tests"}
    target_files = sorted(
        f for f in repo.rglob("*.py")
        if not any(p in skip_parts for p in f.parts)
    )
    log("Phase A", f"edit 対象: {[str(f.relative_to(repo)) for f in target_files]}")

    prior_error = ""
    for it in range(1, MAX_ITER + 1):
        log("Loop", f"=== iteration {it}/{MAX_ITER} ===")

        log("Phase B", "14b で plan 作成中…")
        t0 = time.time()
        context = retrieve(task, collection, k=RETRIEVE_K)
        plan = plan_with_14b(task, context, prior_error=prior_error)
        log("Phase B", f"plan 作成完了 ({time.time()-t0:.1f}s):")
        for line in plan.splitlines():
            log("Phase B", f"  > {line}")

        log("Phase B", "Aider (7b) で edit 実行中…")
        t0 = time.time()
        rc = implement_with_aider(plan, target_files, repo)
        log("Phase B", f"edit 完了 ({time.time()-t0:.1f}s) rc={rc}")

        log("Phase C", "pytest 実行中…")
        t0 = time.time()
        passed, output = run_tests(repo)
        log("Phase C", f"pytest 完了 ({time.time()-t0:.1f}s) — {'PASS' if passed else 'FAIL'}")

        if passed:
            commit_msg = f"hotfix: {task[:60]}\n\nPlan (iter {it}):\n{plan}"
            log("Phase C", "全 test PASS — git commit")
            git_commit(repo, commit_msg)
            log("Done", f"成功 (iteration {it})")
            return 0

        log("Phase C", "失敗内容 (head):")
        for line in output.splitlines()[:30]:
            log("Phase C", f"  | {line}")
        prior_error = output[-2000:]

    log("Done", f"{MAX_ITER} 回 iteration で test green にできず断念")
    return 1


if __name__ == "__main__":
    sys.exit(main())
