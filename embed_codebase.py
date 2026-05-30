#!/usr/bin/env python3
"""embed_codebase.py — bge-m3 + Qdrant の embedding PoC

~/.hermes/hermes-agent/agent/ 配下の *.py を chunk して bge-m3 で embed、
Qdrant collection に投入。完了後、サンプルクエリを実行して top-3 を表示。

実行: source ~/ai_agents_env/bin/activate && python embed_codebase.py
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path
from typing import Iterator

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

CODE_ROOT = Path.home() / ".hermes" / "hermes-agent" / "agent"
OLLAMA_URL = "http://127.0.0.1:11434"
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
COLLECTION = "hermes_agent_code"
EMBED_MODEL = "bge-m3"
VEC_DIM = 1024
CHUNK_LINES = 100
CHUNK_STRIDE = 75  # 25 行の overlap

SAMPLE_QUERIES = [
    "tool calling and function dispatch logic",
    "MCP server registration and lifecycle",
    "model loading and Ollama provider config",
    "session save and resume from disk",
    "rate limit and retry handling",
    "ACP protocol stdio JSON-RPC handler",
]


def iter_chunks(root: Path) -> Iterator[dict]:
    for py in sorted(root.rglob("*.py")):
        try:
            text = py.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"  skip {py}: {e}", file=sys.stderr)
            continue
        lines = text.splitlines()
        if not lines:
            continue
        rel = py.relative_to(root)
        start = 0
        while start < len(lines):
            end = min(start + CHUNK_LINES, len(lines))
            chunk_text = "\n".join(lines[start:end])
            if chunk_text.strip():
                yield {
                    "file": str(rel),
                    "start_line": start + 1,
                    "end_line": end,
                    "code": chunk_text,
                }
            if end >= len(lines):
                break
            start += CHUNK_STRIDE


def embed(text: str) -> list[float]:
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def main() -> None:
    print(f"=== codebase: {CODE_ROOT} ===")
    if not CODE_ROOT.exists():
        sys.exit(f"missing: {CODE_ROOT}")

    qd = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if qd.collection_exists(COLLECTION):
        print(f"既存 collection {COLLECTION!r} を削除して再作成")
        qd.delete_collection(COLLECTION)
    qd.create_collection(
        COLLECTION,
        vectors_config=VectorParams(size=VEC_DIM, distance=Distance.COSINE),
    )
    print(f"collection 作成: {COLLECTION} (dim={VEC_DIM}, distance=cosine)")

    chunks = list(iter_chunks(CODE_ROOT))
    print(f"chunks 数: {len(chunks)} (chunk={CHUNK_LINES} 行 / stride={CHUNK_STRIDE} 行)")

    t0 = time.time()
    points: list[PointStruct] = []
    BATCH = 64
    for i, c in enumerate(chunks):
        vec = embed(c["code"])
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vec, payload=c))

        if (i + 1) % 50 == 0 or i == len(chunks) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(chunks) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:>4}/{len(chunks)}] {rate:.2f} chunks/s  ETA {eta:.0f}s")

        if len(points) >= BATCH:
            qd.upsert(COLLECTION, points=points)
            points = []

    if points:
        qd.upsert(COLLECTION, points=points)

    total = time.time() - t0
    info = qd.get_collection(COLLECTION)
    print()
    print("=== 投入完了 ===")
    print(f"  total chunks: {len(chunks)}")
    print(f"  elapsed: {total:.1f}s ({len(chunks)/total:.2f} chunks/s avg)")
    print(f"  qdrant points: {info.points_count}")

    print()
    print("=== サンプル search (top-3, cosine score) ===")
    for q in SAMPLE_QUERIES:
        print()
        print(f">>> query: {q!r}")
        qv = embed(q)
        results = qd.query_points(
            collection_name=COLLECTION, query=qv, limit=3
        ).points
        for j, r in enumerate(results, 1):
            p = r.payload or {}
            print(
                f"  [{j}] score={r.score:.4f}  "
                f"{p.get('file')}:{p.get('start_line')}-{p.get('end_line')}"
            )
            head_lines = (p.get("code") or "").splitlines()[:3]
            for h in head_lines:
                print(f"      | {h[:100]}")


if __name__ == "__main__":
    main()
