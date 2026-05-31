#!/usr/bin/env python3
"""cleanse_chunk.py — Phase 2.5a PoC: deepseek-r1:14b で Qdrant chunks を cleanse.

Karpathy Layer 2 Wiki Pages 化の技術検証スクリプト。1 chunk = ~3000 chars を
14B に投入し、ナビゲーション / footer / 広告 / 関連記事 / 重複セクションを
除去した cleansed markdown を取得 → 元/cleansed を並列表示 + cosine 比較。

PoC スコープ:
- Qdrant `web_brain` から N chunks 取得 (default 3)
- ollama REST /api/generate 経由で deepseek-r1:14b に投入
- <think>...</think> ブロックを post-process で除去 (DeepSeek-R1 reasoning モデル)
- bge-m3 で 元/cleansed を embed → cosine 計算
- Qdrant 投入なし (純粋な目視・品質評価のため)

含まない (Phase 2.5b 以降):
- web_research.py 統合 (--cleanse オプション)
- Modelfile.deepseek-r1-14b-32k (page 単位 cleanse 時に必要)
- 新 Qdrant collection web_brain_clean
- 全 chunks の cleanse バッチ
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import time

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

OLLAMA_URL = "http://127.0.0.1:11434"
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
COLLECTION = "web_brain"
CLEANSE_MODEL = "deepseek-r1:14b"
EMBED_MODEL = "bge-m3"
DEFAULT_LIMIT = 3
GENERATE_TIMEOUT_SEC = 300  # 14B cold load 85s + inference 余裕
NUM_PREDICT = 2048

CLEANSE_PROMPT_TEMPLATE = """You are a knowledge-base cleaner. Below is a Markdown chunk scraped from a webpage. Clean it for storage in a semantic search index:

REMOVE:
- Navigation menus, breadcrumbs, sidebars
- Footer text (copyright, cookie notices, subscribe forms)
- Advertisements and promotional links
- "Related articles" / "You may also like" sections
- Duplicate consecutive headings or text

KEEP:
- Body text (paragraphs, explanations)
- Code blocks (```...```) verbatim
- Quotations and lists
- Headings (# / ## / ###) that introduce real content

Output ONLY the cleaned Markdown, no commentary, no explanation.

--- BEGIN CHUNK ---
{text}
--- END CHUNK ---"""


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks (DeepSeek-R1 reasoning output)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def embed(text: str) -> list[float]:
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def cleanse_chunk(text: str, *, timeout: int = GENERATE_TIMEOUT_SEC) -> str:
    """Call deepseek-r1:14b via Ollama /api/generate, return cleansed text."""
    prompt = CLEANSE_PROMPT_TEMPLATE.format(text=text)
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": CLEANSE_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": NUM_PREDICT},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    raw = r.json().get("response", "")
    return strip_think_blocks(raw)


def fetch_chunks(qd: QdrantClient, limit: int, theme: str | None) -> list[dict]:
    flt = None
    if theme:
        flt = Filter(
            must=[FieldCondition(key="theme", match=MatchValue(value=theme))]
        )
    results = qd.scroll(
        collection_name=COLLECTION,
        limit=limit,
        with_payload=True,
        scroll_filter=flt,
    )[0]
    return [p.payload for p in results]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 2.5a PoC: cleanse chunks with deepseek-r1:14b"
    )
    ap.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"chunks to cleanse (default {DEFAULT_LIMIT})",
    )
    ap.add_argument(
        "--theme", default=None, help="filter by payload.theme (exact match)",
    )
    ap.add_argument(
        "--preview-chars", type=int, default=1500,
        help="chars to print for original/cleansed preview (default 1500)",
    )
    args = ap.parse_args()

    qd = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    chunks = fetch_chunks(qd, args.limit, args.theme)
    if not chunks:
        print(
            f"[ERROR] no chunks in {COLLECTION} (theme filter: {args.theme!r})",
            file=sys.stderr,
        )
        return 1

    print(
        f"=== Phase 2.5a PoC: cleanse {len(chunks)} chunks via {CLEANSE_MODEL} ===\n"
    )

    summary: list[tuple[int, int, float, float]] = []

    for i, payload in enumerate(chunks, 1):
        text = payload.get("text", "")
        heading = payload.get("heading_path", "(no heading)")
        url = payload.get("url", "")
        theme = payload.get("theme", "")

        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(chunks)}] heading: {heading}")
        print(f"  theme: {theme!r}")
        print(f"  url: {url}")
        print(f"  original: {len(text)} chars")

        t0 = time.time()
        try:
            cleansed = cleanse_chunk(text)
        except requests.exceptions.RequestException as e:
            print(f"  [ERROR] cleanse failed: {e}", file=sys.stderr)
            continue
        elapsed = time.time() - t0
        delta_pct = (
            (len(text) - len(cleansed)) / max(len(text), 1) * 100
        )
        print(
            f"  cleansed: {len(cleansed)} chars "
            f"({elapsed:.1f}s, {delta_pct:+.1f}% reduction)"
        )

        cos = 0.0
        if cleansed.strip():
            try:
                v_orig = embed(text)
                v_clean = embed(cleansed)
                cos = cosine_similarity(v_orig, v_clean)
                print(f"  cosine(orig, cleansed) = {cos:.4f}")
            except requests.exceptions.RequestException as e:
                print(f"  [WARN] embed failed: {e}", file=sys.stderr)
        else:
            print("  [WARN] cleansed empty, skip cosine")

        summary.append((len(text), len(cleansed), elapsed, cos))

        n = args.preview_chars
        print(f"\n--- ORIGINAL (first {n} chars) ---\n{text[:n]}"
              f"{'...' if len(text) > n else ''}")
        print(f"\n--- CLEANSED (first {n} chars) ---\n{cleansed[:n]}"
              f"{'...' if len(cleansed) > n else ''}")

    print(f"\n\n{'=' * 70}")
    print(f"=== SUMMARY (n={len(summary)}) ===")
    if summary:
        total_orig = sum(s[0] for s in summary)
        total_clean = sum(s[1] for s in summary)
        total_elapsed = sum(s[2] for s in summary)
        avg_cos = sum(s[3] for s in summary) / len(summary)
        print(f"  total chars: {total_orig} → {total_clean} "
              f"({(total_orig - total_clean) / max(total_orig, 1) * 100:+.1f}%)")
        print(f"  total elapsed: {total_elapsed:.1f}s "
              f"(avg {total_elapsed / len(summary):.1f}s/chunk)")
        print(f"  avg cosine: {avg_cos:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
