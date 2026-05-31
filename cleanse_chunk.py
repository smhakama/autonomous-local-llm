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
import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

OLLAMA_URL = "http://127.0.0.1:11434"
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
COLLECTION = "web_brain"
CLEANSE_MODEL = "deepseek-r1:14b"
EMBED_MODEL = "bge-m3"
DEFAULT_LIMIT = 20  # Qdrant scroll 上限 (filter 前の input pool)
GENERATE_TIMEOUT_SEC = 300  # 14B cold load 85s + inference 余裕
NUM_PREDICT = 2048

# --- Phase 2.5b1 セーフティネット定数 ---
DEFAULT_MAX_CHUNKS = 5  # 1 batch の最大 cleanse 件数 (暴走防止)
DEFAULT_PER_CHUNK_TIMEOUT = 180  # 1 chunk あたり (秒)
DEFAULT_FILTER_MAX_CHARS = 1500  # 短い chunk のみ cleanse 対象
DEFAULT_FILTER_HEADING_KEYWORDS = (
    "nav", "footer", "sidebar", "related", "menu", "breadcrumb",
)
DEFAULT_SIDECAR_DIR = "cleanse_output"
DEFAULT_OUTPUT_MODE = "sidecar"  # 最安全 (Qdrant 触らず)
OUTPUT_MODES = ("sidecar", "new-collection", "update-payload", "preview")


@dataclass(frozen=True)
class CleanseResult:
    """1 chunk の cleanse 結果 (immutable)。"""

    chunk_id: str
    heading_path: str
    url: str
    theme: str
    original_text: str
    cleansed_text: str
    original_chars: int
    cleansed_chars: int
    elapsed_sec: float
    cosine: float
    skipped: bool
    skip_reason: str | None
    batch_id: str

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


_PROMPT_MARKER_PATTERN = re.compile(
    r"^\s*---\s*(BEGIN|END)\s+CHUNK\s*---\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def strip_prompt_markers(text: str) -> str:
    """Remove '--- BEGIN/END CHUNK ---' markers if model echoes prompt."""
    return _PROMPT_MARKER_PATTERN.sub("", text).strip()


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
    return strip_prompt_markers(strip_think_blocks(raw))


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


def passes_filter(
    payload: dict,
    filter_max_chars: int,
    filter_heading_keywords: tuple[str, ...],
) -> tuple[bool, str | None]:
    """セーフティ filter: AND ロジックで「短い + nav-like heading」のみ採択。

    filter_max_chars <= 0 で文字数チェック無効。
    filter_heading_keywords が空 tuple で keyword チェック無効。
    """
    text = payload.get("text", "")
    heading = (payload.get("heading_path") or "").lower()

    if filter_max_chars > 0 and len(text) > filter_max_chars:
        return False, f"chars {len(text)} > max {filter_max_chars}"
    if filter_heading_keywords:
        if not any(kw.lower() in heading for kw in filter_heading_keywords):
            preview = heading[:40] + ("…" if len(heading) > 40 else "")
            return False, f"heading {preview!r} has no nav-like keyword"
    return True, None


def _make_chunk_id(payload: dict) -> str:
    return f"{payload.get('url', '')}#{payload.get('chunk_idx', 0)}"


def _write_sidecar(results: list[CleanseResult], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            row = {
                "batch_id": r.batch_id,
                "chunk_id": r.chunk_id,
                "heading_path": r.heading_path,
                "url": r.url,
                "theme": r.theme,
                "original_chars": r.original_chars,
                "cleansed_chars": r.cleansed_chars,
                "elapsed_sec": round(r.elapsed_sec, 2),
                "cosine": round(r.cosine, 4),
                "skipped": r.skipped,
                "skip_reason": r.skip_reason,
                "cleansed_text": r.cleansed_text,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


DEFAULT_CLEAN_COLLECTION = "web_brain_clean"
CLEAN_VEC_DIM = 1024  # bge-m3 dim


def _upsert_to_clean_collection(
    qd: QdrantClient,
    results: list[CleanseResult],
    *,
    collection: str = DEFAULT_CLEAN_COLLECTION,
    vec_dim: int = CLEAN_VEC_DIM,
) -> int:
    """cleanse 結果を別 collection に upsert (元 collection は無変更)。

    point ID = sha1("clean::" + chunk_id + "::" + batch_id) → UUID で決定論的。
    payload に source_chunk_id / cosine_with_original / 元/cleansed chars を保存
    することで追跡可能性 + 品質メトリクスを残す。
    """
    if not qd.collection_exists(collection):
        qd.create_collection(
            collection,
            vectors_config=VectorParams(size=vec_dim, distance=Distance.COSINE),
        )
        print(f"created clean collection: {collection} (dim={vec_dim})")

    points: list[PointStruct] = []
    for r in results:
        if r.skipped or not r.cleansed_text.strip():
            continue
        try:
            vec = embed(r.cleansed_text)
        except requests.exceptions.RequestException as e:
            print(f"  [WARN] embed for upsert failed: {e}", file=sys.stderr)
            continue
        h = hashlib.sha1(
            f"clean::{r.chunk_id}::{r.batch_id}".encode("utf-8")
        ).hexdigest()
        pid = str(uuid.UUID(hex=h[:32]))
        points.append(
            PointStruct(
                id=pid,
                vector=vec,
                payload={
                    "batch_id": r.batch_id,
                    "source_chunk_id": r.chunk_id,
                    "theme": r.theme,
                    "url": r.url,
                    "heading_path": r.heading_path,
                    "original_chars": r.original_chars,
                    "cleansed_chars": r.cleansed_chars,
                    "cosine_with_original": r.cosine,
                    "text": r.cleansed_text,
                },
            )
        )
    if points:
        qd.upsert(collection, points=points)
    return len(points)


def cleanse_batch(
    chunks: list[dict],
    *,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    per_chunk_timeout: int = DEFAULT_PER_CHUNK_TIMEOUT,
    dry_run: bool = False,
    output_mode: str = DEFAULT_OUTPUT_MODE,
    sidecar_path: str | None = None,
    batch_id: str | None = None,
    filter_max_chars: int = DEFAULT_FILTER_MAX_CHARS,
    filter_heading_keywords: tuple[str, ...] = DEFAULT_FILTER_HEADING_KEYWORDS,
    clean_collection: str = DEFAULT_CLEAN_COLLECTION,
    qd_client: QdrantClient | None = None,
) -> list[CleanseResult]:
    """セーフティネット込みのバッチ cleanse。

    順序: filter → max_chunks で truncate → dry-run なら一覧のみ
          → 各 chunk を timeout 保護で cleanse → output_mode で保存。

    Returns: CleanseResult のリスト (dry-run 時は空 list)。
    """
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f"invalid output_mode {output_mode!r}, expect one of {OUTPUT_MODES}")
    if batch_id is None:
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. filter
    eligible: list[dict] = []
    skipped: list[tuple[dict, str]] = []
    for p in chunks:
        ok, reason = passes_filter(p, filter_max_chars, filter_heading_keywords)
        if ok:
            eligible.append(p)
        else:
            skipped.append((p, reason or "filtered"))

    # 2. max_chunks で truncate
    selected = eligible[:max_chunks]
    truncated = len(eligible) - len(selected)

    print(f"=== batch_id={batch_id} mode={output_mode} dry_run={dry_run} ===")
    print(
        f"input={len(chunks)} → filtered_out={len(skipped)} → "
        f"eligible={len(eligible)} → selected={len(selected)} "
        f"(max_chunks={max_chunks}, truncated={truncated})"
    )
    for p, reason in skipped[:5]:
        head = (p.get("heading_path") or "")[:50]
        print(f"  - SKIP heading={head!r}: {reason}")
    if len(skipped) > 5:
        print(f"  ... and {len(skipped) - 5} more skipped")

    # 3. dry-run → 推定時間と一覧のみ
    if dry_run:
        avg_per_chunk = 60  # filter 後の短い chunk 想定 (PoC は 110s/chunk)
        est = len(selected) * avg_per_chunk
        print(
            f"DRY-RUN: estimated ~{est}s "
            f"({est // 60}m {est % 60}s) at ~{avg_per_chunk}s/chunk"
        )
        for i, p in enumerate(selected, 1):
            head = (p.get("heading_path") or "")[:60]
            print(f"  [{i}] heading={head!r} ({len(p.get('text', ''))} chars)")
        return []

    # 4. 実行 (per-chunk timeout で保護)
    results: list[CleanseResult] = []
    for i, p in enumerate(selected, 1):
        text = p.get("text", "")
        heading = p.get("heading_path") or ""
        chunk_id = _make_chunk_id(p)
        print(f"\n[{i}/{len(selected)}] cleanse: heading={heading[:50]!r} ({len(text)} chars)")

        t0 = time.time()
        try:
            cleansed = cleanse_chunk(text, timeout=per_chunk_timeout)
            elapsed = time.time() - t0
            cos = 0.0
            if cleansed.strip():
                try:
                    v_orig = embed(text)
                    v_clean = embed(cleansed)
                    cos = cosine_similarity(v_orig, v_clean)
                except requests.exceptions.RequestException as e:
                    print(f"  [WARN] embed failed: {e}", file=sys.stderr)
            print(
                f"  → {len(cleansed)} chars ({elapsed:.1f}s, cosine={cos:.4f})"
            )
            results.append(
                CleanseResult(
                    chunk_id=chunk_id, heading_path=heading,
                    url=p.get("url", ""), theme=p.get("theme", ""),
                    original_text=text, cleansed_text=cleansed,
                    original_chars=len(text), cleansed_chars=len(cleansed),
                    elapsed_sec=elapsed, cosine=cos,
                    skipped=False, skip_reason=None, batch_id=batch_id,
                )
            )
        except requests.exceptions.Timeout:
            elapsed = time.time() - t0
            print(f"  [TIMEOUT] {per_chunk_timeout}s exceeded, skip", file=sys.stderr)
            results.append(
                CleanseResult(
                    chunk_id=chunk_id, heading_path=heading,
                    url=p.get("url", ""), theme=p.get("theme", ""),
                    original_text=text, cleansed_text="",
                    original_chars=len(text), cleansed_chars=0,
                    elapsed_sec=elapsed, cosine=0.0,
                    skipped=True, skip_reason=f"timeout {per_chunk_timeout}s",
                    batch_id=batch_id,
                )
            )
        except requests.exceptions.RequestException as e:
            elapsed = time.time() - t0
            print(f"  [ERROR] cleanse failed: {e}", file=sys.stderr)
            results.append(
                CleanseResult(
                    chunk_id=chunk_id, heading_path=heading,
                    url=p.get("url", ""), theme=p.get("theme", ""),
                    original_text=text, cleansed_text="",
                    original_chars=len(text), cleansed_chars=0,
                    elapsed_sec=elapsed, cosine=0.0,
                    skipped=True, skip_reason=f"error: {type(e).__name__}",
                    batch_id=batch_id,
                )
            )

    # 5. output_mode 別保存
    if output_mode == "sidecar":
        path = sidecar_path or os.path.join(DEFAULT_SIDECAR_DIR, f"{batch_id}.jsonl")
        _write_sidecar(results, path)
        print(f"\nsidecar written: {path} ({len(results)} lines)")
    elif output_mode == "new-collection":
        qd = qd_client or QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        n = _upsert_to_clean_collection(
            qd, results, collection=clean_collection,
        )
        print(
            f"\nupserted to clean collection '{clean_collection}': "
            f"{n} points ({len(results) - n} skipped/empty)"
        )
    elif output_mode == "update-payload":
        print(
            "[WARN] output-mode 'update-payload' is not yet implemented "
            "(future phase)",
            file=sys.stderr,
        )
    # preview mode は保存なし (caller 側で目視表示)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Phase 2.5b1: cleanse chunks with deepseek-r1:14b "
            "(セーフティネット付き batch + library)"
        )
    )
    ap.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"Qdrant scroll 取得上限 (filter 前 input pool、default {DEFAULT_LIMIT})",
    )
    ap.add_argument(
        "--theme", default=None,
        help="filter by payload.theme (exact match)",
    )
    ap.add_argument(
        "--max-chunks", type=int, default=DEFAULT_MAX_CHUNKS,
        help=f"max chunks to cleanse per batch (default {DEFAULT_MAX_CHUNKS}、暴走防止)",
    )
    ap.add_argument(
        "--per-chunk-timeout", type=int, default=DEFAULT_PER_CHUNK_TIMEOUT,
        help=f"per-chunk timeout sec (default {DEFAULT_PER_CHUNK_TIMEOUT})",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="list target chunks + 推定時間表示のみ、Ollama 呼出なし",
    )
    ap.add_argument(
        "--output-mode", choices=OUTPUT_MODES, default=DEFAULT_OUTPUT_MODE,
        help=f"output mode (default {DEFAULT_OUTPUT_MODE})",
    )
    ap.add_argument(
        "--sidecar-path", default=None,
        help=f"sidecar JSONL path (default {DEFAULT_SIDECAR_DIR}/<batch_id>.jsonl)",
    )
    ap.add_argument(
        "--batch-id", default=None,
        help="custom batch id (default timestamp YYYYMMDD_HHMMSS)",
    )
    ap.add_argument(
        "--filter-max-chars", type=int, default=DEFAULT_FILTER_MAX_CHARS,
        help=(
            f"filter: 短い chunk のみ採択 (default {DEFAULT_FILTER_MAX_CHARS}、"
            f"0 で無効)"
        ),
    )
    ap.add_argument(
        "--filter-heading-keywords",
        default=",".join(DEFAULT_FILTER_HEADING_KEYWORDS),
        help=(
            "filter: heading に含むべき comma-separated keywords "
            f"(default '{','.join(DEFAULT_FILTER_HEADING_KEYWORDS)}'、"
            "空文字で無効)"
        ),
    )
    ap.add_argument(
        "--preview-chars", type=int, default=1500,
        help="preview mode 時の表示文字数 (default 1500)",
    )
    ap.add_argument(
        "--clean-collection", default=DEFAULT_CLEAN_COLLECTION,
        help=(
            f"output-mode=new-collection 時の投入先 collection "
            f"(default {DEFAULT_CLEAN_COLLECTION})"
        ),
    )
    args = ap.parse_args()

    raw_kw = (args.filter_heading_keywords or "").strip()
    if raw_kw:
        keywords: tuple[str, ...] = tuple(
            k.strip() for k in raw_kw.split(",") if k.strip()
        )
    else:
        keywords = ()

    qd = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    chunks = fetch_chunks(qd, args.limit, args.theme)
    if not chunks:
        print(
            f"[ERROR] no chunks in {COLLECTION} "
            f"(theme filter: {args.theme!r}, limit: {args.limit})",
            file=sys.stderr,
        )
        return 1

    results = cleanse_batch(
        chunks,
        max_chunks=args.max_chunks,
        per_chunk_timeout=args.per_chunk_timeout,
        dry_run=args.dry_run,
        output_mode=args.output_mode,
        sidecar_path=args.sidecar_path,
        batch_id=args.batch_id,
        filter_max_chars=args.filter_max_chars,
        filter_heading_keywords=keywords,
        clean_collection=args.clean_collection,
        qd_client=qd,
    )

    # preview mode: 元/cleansed の並列表示 (PoC 互換)
    if args.output_mode == "preview" and results:
        n = args.preview_chars
        for i, r in enumerate(results, 1):
            print(f"\n{'=' * 70}")
            print(f"[{i}/{len(results)}] heading: {r.heading_path}")
            print(
                f"  cosine={r.cosine:.4f}, "
                f"{r.original_chars}→{r.cleansed_chars} chars "
                f"({r.elapsed_sec:.1f}s)"
            )
            if r.skipped:
                print(f"  SKIPPED: {r.skip_reason}")
                continue
            print(
                f"\n--- ORIGINAL (first {n} chars) ---\n{r.original_text[:n]}"
                f"{'...' if r.original_chars > n else ''}"
            )
            print(
                f"\n--- CLEANSED (first {n} chars) ---\n{r.cleansed_text[:n]}"
                f"{'...' if r.cleansed_chars > n else ''}"
            )

    if results:
        total_orig = sum(r.original_chars for r in results)
        total_clean = sum(r.cleansed_chars for r in results)
        total_elapsed = sum(r.elapsed_sec for r in results)
        avg_cos = sum(r.cosine for r in results) / len(results)
        n_skip = sum(1 for r in results if r.skipped)
        print(f"\n{'=' * 70}")
        print(f"=== SUMMARY (n={len(results)}, skipped={n_skip}) ===")
        print(
            f"  total chars: {total_orig} → {total_clean} "
            f"({(total_orig - total_clean) / max(total_orig, 1) * 100:+.1f}%)"
        )
        print(
            f"  total elapsed: {total_elapsed:.1f}s "
            f"(avg {total_elapsed / len(results):.1f}s/chunk)"
        )
        print(f"  avg cosine: {avg_cos:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
