#!/usr/bin/env python3
"""web_research.py — テーマ駆動 web research → bge-m3 → Qdrant.

Phase 2 v2: Discovery は検索エンジン HTML を直接スクレイプ (LLM 不使用)。
v1 (Agent ベース) は qwen2.5:7b-32k が Tool call の URL フィールドに思考を
混入する破綻が出たため、Discovery を決定論的処理に置き換えた。

    [SearxNG JSON API] → URL 候補抽出
        ↓
    [Domain Allowlist/Denylist] → 優先度スコアで採択
        ↓
    [Playwright direct fetch] → 各 URL の HTML 取得
        ↓
    [html2text] → ノイズ除去 + Markdown 化
        ↓
    [heading-based chunking] → # / ## 境界、1000-3000 文字目安
        ↓
    [bge-m3 embed] → Qdrant upsert
        (payload = theme/url/title/fetched_at/heading_path)
        ↓
    [類似検索デモ]

CLI:
    python web_research.py "<theme>"
    python web_research.py "<theme>" --max-pages 5
    python web_research.py --search "<query>" --top 5
    python web_research.py --test-markdown <URL>

Phase 2 で意図的に未実装 (Phase 2.1+ で追加予定):
- query expansion (1-shot LLM call で検索キーワード拡張) — Phase 2.1c
- deepseek-r1:14b による 2nd-stage cleanse — Phase 2.5

Phase 2.1b で追加:
- --themes-file <path>: 1 行 1 テーマのバッチ実行
- --lock-file: fcntl.flock による多重起動防止
- --strict: 1 件失敗で全体停止 (default は継続)

Discovery backend (2026-05-31):
- DuckDuckGo HTML (html.duckduckgo.com/html/) は WSL 帯から
  status=202 + 空結果を返すようになり scrape 不能。
- Brave HTML scrape も試したが rate limit が厳しく (~1 req で 429、
  cooldown 30-60 秒)、--themes-file × --expand-query 運用に不適。
- SearxNG (self-host, http://127.0.0.1:8888) に置換。
  内部で brave/mojeek/qwant 等を並列叩き、JSON で集約結果を返す。
  rate limit は self-host なので無し。container 設定は ~/searxng/。
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator
from urllib.parse import urlparse

import html2text
import requests
from playwright.async_api import (
    Error as PlaywrightError,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from cleanse_chunk import cleanse_batch
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

OLLAMA_URL = "http://127.0.0.1:11434"
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
COLLECTION = "web_brain"
EMBED_MODEL = "bge-m3"
VEC_DIM = 1024
MAX_PAGES_DEFAULT = 4
EMBED_TIMEOUT_SEC = 120
FETCH_TIMEOUT_MS = 30000
FETCH_MAX_ATTEMPTS = 3
FETCH_BACKOFF_MIN_SEC = 1.0
FETCH_BACKOFF_MAX_SEC = 8.0
HEADING_CHUNK_MAX_CHARS = 3000
HEADING_CHUNK_MIN_CHARS = 200

SEARXNG_URL = "http://127.0.0.1:8888/search"
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
SEARCH_FETCH_TIMEOUT_SEC = 30
SEARCH_CANDIDATES_PER_QUERY = 30  # スコアリング前の候補上限

DOMAIN_PRIORITY: dict[str, int] = {
    "fastapi.tiangolo.com": 5,
    "docs.python.org": 5,
    "developer.mozilla.org": 5,
    "kubernetes.io": 5,
    "github.com": 4,
    "zenn.dev": 3,
    "qiita.com": 3,
    "dev.to": 3,
    "ja.wikipedia.org": 3,
    "en.wikipedia.org": 3,
    "medium.com": 2,
    "note.com": 1,
}

DOMAIN_DENYLIST: set[str] = {
    "pinterest.com", "pinterest.jp",
    "facebook.com", "twitter.com", "x.com",
    "instagram.com", "tiktok.com",
    "youtube.com", "m.youtube.com",
}


@dataclass(frozen=True)
class PageDoc:
    url: str
    title: str
    html: str
    fetched_at: str


@dataclass(frozen=True)
class Chunk:
    heading_path: str
    text: str


def _normalize_host(netloc: str) -> str:
    host = netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _is_article_url(url: str) -> bool:
    try:
        p = urlparse(url)
    except ValueError:
        return False
    if p.scheme not in ("http", "https"):
        return False
    if not p.netloc:
        return False
    host = _normalize_host(p.netloc)
    if host in DOMAIN_DENYLIST:
        return False
    return True


def _domain_score(url: str) -> int:
    try:
        host = _normalize_host(urlparse(url).netloc)
    except ValueError:
        return 0
    return DOMAIN_PRIORITY.get(host, 0)


def discover_urls_via_searxng(theme: str, max_pages: int) -> list[str]:
    """SearxNG (self-host) JSON API → URL リストを優先度順に返す。"""
    print(f"=== SearxNG search: {theme!r}")
    try:
        r = requests.get(
            SEARXNG_URL,
            params={"q": theme, "format": "json", "language": "all"},
            headers={"User-Agent": BROWSER_USER_AGENT},
            timeout=SEARCH_FETCH_TIMEOUT_SEC,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[ERROR] SearxNG fetch failed: {e}", file=sys.stderr)
        return []

    raw_results = data.get("results", [])
    unresponsive = data.get("unresponsive_engines", [])
    if unresponsive:
        print(f"  (unresponsive engines: {unresponsive})")

    raw_candidates: list[str] = []
    for item in raw_results:
        url = item.get("url", "")
        if url:
            raw_candidates.append(url)
        if len(raw_candidates) >= SEARCH_CANDIDATES_PER_QUERY:
            break

    print(f"SearxNG raw candidates: {len(raw_candidates)} 件")
    seen: set[str] = set()
    filtered: list[str] = []
    for url in raw_candidates:
        if url in seen:
            continue
        if not _is_article_url(url):
            continue
        seen.add(url)
        filtered.append(url)
    print(f"after filter (article-like + denylist): {len(filtered)} 件")

    ranked = sorted(filtered, key=lambda u: (-_domain_score(u), filtered.index(u)))
    picked = ranked[:max_pages]
    print(f"採用 URL (top {len(picked)}):")
    for u in picked:
        print(f"  ✓ [score={_domain_score(u)}] {u}")
    return picked


_RETRIABLE_NET_ERRORS: tuple[str, ...] = (
    "ERR_NETWORK_CHANGED",
    "ERR_CONNECTION_RESET",
    "ERR_CONNECTION_REFUSED",
    "ERR_NAME_NOT_RESOLVED",
    "ERR_INTERNET_DISCONNECTED",
    "ERR_TIMED_OUT",
    "net::ERR_",
)


def _is_retriable_fetch_error(exc: BaseException) -> bool:
    if isinstance(exc, PlaywrightTimeoutError):
        return True
    if isinstance(exc, PlaywrightError):
        msg = str(exc)
        return any(tag in msg for tag in _RETRIABLE_NET_ERRORS)
    return False


async def fetch_page(url: str) -> PageDoc | None:
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                ctx = await browser.new_context(user_agent=BROWSER_USER_AGENT)
                page = await ctx.new_page()

                @retry(
                    retry=retry_if_exception(_is_retriable_fetch_error),
                    wait=(
                        wait_exponential(
                            multiplier=1,
                            min=FETCH_BACKOFF_MIN_SEC,
                            max=FETCH_BACKOFF_MAX_SEC,
                        )
                        + wait_random(0, 1)
                    ),
                    stop=stop_after_attempt(FETCH_MAX_ATTEMPTS),
                    reraise=True,
                )
                async def _fetch_with_retry() -> tuple[str, str]:
                    try:
                        await page.goto(
                            url,
                            timeout=FETCH_TIMEOUT_MS,
                            wait_until="domcontentloaded",
                        )
                    except Exception as e:
                        print(
                            f"[RETRY] goto failed for {url}: "
                            f"{type(e).__name__}: {e}",
                            file=sys.stderr,
                        )
                        raise
                    title = (await page.title()) or url
                    html = await page.content()
                    return title, html

                title, html = await _fetch_with_retry()
            finally:
                await browser.close()
    except Exception as e:
        print(
            f"[WARN] fetch failed for {url} after "
            f"{FETCH_MAX_ATTEMPTS} attempts: {e}",
            file=sys.stderr,
        )
        return None
    return PageDoc(
        url=url,
        title=title,
        html=html,
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def htmlize_to_markdown(html: str) -> str:
    converter = html2text.HTML2Text()
    converter.body_width = 0
    converter.ignore_images = True
    converter.ignore_emphasis = False
    converter.ignore_links = False
    converter.skip_internal_links = True
    converter.single_line_break = True
    md = converter.handle(html)
    cleaned: list[str] = []
    for line in md.splitlines():
        s = line.rstrip()
        if not s:
            cleaned.append("")
            continue
        if s.lstrip().startswith(("![", "data:")):
            continue
        cleaned.append(s)
    return "\n".join(cleaned).strip()


def _heading_level(line: str) -> int:
    # Must start at column 0 (avoid matching indented code like `# -*- coding -*-`)
    if not line.startswith("#"):
        return 0
    hashes = 0
    for c in line:
        if c == "#":
            hashes += 1
        else:
            break
    if hashes > 6:
        return 0
    if len(line) > hashes and line[hashes] != " ":
        return 0
    return hashes


def _heading_text(line: str) -> str:
    return line.lstrip("#").strip()


def chunk_by_heading(md: str) -> list[Chunk]:
    out: list[Chunk] = []
    if not md.strip():
        return out
    heading_stack: list[str] = []
    buf: list[str] = []
    in_fence = False

    def flush(path: str) -> None:
        text = "\n".join(buf).strip()
        if not text:
            return
        out.append(Chunk(heading_path=path, text=text))

    def current_path() -> str:
        return " / ".join(heading_stack) if heading_stack else "(no heading)"

    for line in md.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            buf.append(line)
            continue
        if in_fence:
            buf.append(line)
            continue
        level = _heading_level(line)
        if level > 0:
            flush(current_path())
            buf = []
            while len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(_heading_text(line))
            continue
        buf.append(line)
        if sum(len(x) for x in buf) >= HEADING_CHUNK_MAX_CHARS:
            flush(current_path())
            buf = []

    flush(current_path())
    return [c for c in out if len(c.text) >= HEADING_CHUNK_MIN_CHARS]


def embed(text: str) -> list[float]:
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=EMBED_TIMEOUT_SEC,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def ensure_collection(qd: QdrantClient) -> None:
    if not qd.collection_exists(COLLECTION):
        qd.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=VEC_DIM, distance=Distance.COSINE),
        )
        print(f"collection 作成: {COLLECTION} (dim={VEC_DIM}, distance=cosine)")
    else:
        print(f"collection 既存: {COLLECTION}")


def stable_id(theme: str, url: str, chunk_idx: int) -> str:
    key = f"{theme}::{url}::{chunk_idx}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return str(uuid.UUID(hex=h[:32]))


def upsert_page(qd: QdrantClient, theme: str, page: PageDoc, chunks: list[Chunk]) -> int:
    if not chunks:
        return 0
    points: list[PointStruct] = []
    for i, c in enumerate(chunks):
        vec = embed(c.text)
        points.append(
            PointStruct(
                id=stable_id(theme, page.url, i),
                vector=vec,
                payload={
                    "theme": theme,
                    "url": page.url,
                    "title": page.title,
                    "fetched_at": page.fetched_at,
                    "heading_path": c.heading_path,
                    "chunk_idx": i,
                    "text": c.text,
                },
            )
        )
    qd.upsert(COLLECTION, points=points)
    return len(points)


def demo_search(qd: QdrantClient, query: str, top: int = 5) -> None:
    qv = embed(query)
    results = qd.query_points(
        collection_name=COLLECTION, query=qv, limit=top
    ).points
    print()
    print(f"=== similarity search for {query!r} (top-{top}) ===")
    if not results:
        print("  (no results)")
        return
    for j, r in enumerate(results, 1):
        p = r.payload or {}
        head = (p.get("text") or "")[:140].replace("\n", " ")
        print(
            f"  [{j}] score={r.score:.4f}  "
            f"theme={p.get('theme')!r}  url={p.get('url')!r}"
        )
        print(f"      heading={p.get('heading_path')!r}")
        print(f"      | {head}")


ASK_GEMINI_PROMPT_TEMPLATE = (
    "Generate {n} alternative search queries for the topic below. "
    "Output ONLY the queries, one per line, no numbering, no quotes, "
    "no commentary, no markdown.\nTopic: {theme}"
)


def expand_query_via_gemini(
    theme: str,
    bin_path: str = "ask_gemini",
    limit: int = 5,
    timeout_sec: int = 30,
) -> list[str]:
    """ask_gemini で検索ワード拡張。返値は [theme, expanded_1, ...] (重複除去済)。

    失敗時 (timeout / 不在 / 非 0 終了) は [theme] にフォールバック (継続)。
    """
    prompt = ASK_GEMINI_PROMPT_TEMPLATE.format(n=limit, theme=theme)
    try:
        r = subprocess.run(
            [bin_path, prompt],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        print(
            f"[WARN] ask_gemini timeout ({timeout_sec}s) for {theme!r}, "
            f"fallback to single query",
            file=sys.stderr,
        )
        return [theme]
    except FileNotFoundError:
        print(
            f"[WARN] ask_gemini not found ({bin_path!r}), "
            f"fallback to single query",
            file=sys.stderr,
        )
        return [theme]

    if r.returncode != 0:
        msg = (r.stderr or "").strip()[:200]
        print(
            f"[WARN] ask_gemini rc={r.returncode} for {theme!r}: {msg}",
            file=sys.stderr,
        )
        return [theme]

    seen: set[str] = {theme.lower()}
    queries: list[str] = [theme]
    for line in r.stdout.splitlines():
        q = line.strip().strip('"').strip("'")
        if not q or q.lower() in seen:
            continue
        seen.add(q.lower())
        queries.append(q)
        if len(queries) >= limit + 1:  # +1 for original theme
            break

    print(f"[expand-query] {len(queries)} queries ({len(queries) - 1} expanded)")
    for i, q in enumerate(queries):
        print(f"  [{i}] {q}")
    return queries


def parse_themes_file(path: str) -> list[str]:
    """themes.txt をパースして正規化済 theme リストを返す。

    フォーマット:
    - 1 行 1 テーマ
    - 空行スキップ
    - 行頭 # コメント行スキップ
    - UTF-8 BOM 除去 (utf-8-sig)
    - 行末空白 trim
    - 大文字小文字無視で重複検出 → 警告 + 1 回だけ採用
    """
    with open(path, encoding="utf-8-sig") as f:
        raw_lines = f.readlines()
    seen_lower: set[str] = set()
    themes: list[str] = []
    for lineno, line in enumerate(raw_lines, 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        key = s.lower()
        if key in seen_lower:
            print(
                f"[WARN] themes-file line {lineno}: duplicate skipped: {s!r}",
                file=sys.stderr,
            )
            continue
        seen_lower.add(key)
        themes.append(s)
    return themes


def _parse_cleanse_keywords(raw: str) -> tuple[str, ...]:
    """comma-separated string → tuple of stripped keywords (空文字で空 tuple)。"""
    raw = (raw or "").strip()
    if not raw:
        return ()
    return tuple(k.strip() for k in raw.split(",") if k.strip())


async def run_research(
    theme: str,
    max_pages: int,
    *,
    expand_query: bool = False,
    expand_limit: int = 5,
    ask_gemini_bin: str = "ask_gemini",
    ask_gemini_timeout: int = 30,
    cleanse: bool = False,
    cleanse_max_chunks_per_page: int = 5,
    cleanse_per_chunk_timeout: int = 180,
    cleanse_filter_max_chars: int = 1500,
    cleanse_filter_heading_keywords: tuple[str, ...] = (
        "nav", "footer", "sidebar", "related", "menu", "breadcrumb",
    ),
    cleanse_collection: str = "web_brain_clean",
    cleanse_filter_mode: str = "nav-only",
    cleanse_body_min_chars: int = 1000,
    cleanse_body_max_chars: int = 2000,
    cleanse_per_url_top: int = 3,
    cleanse_cosine_gate: float = 0.9,
) -> int:
    print(f"=== theme: {theme!r} ===")
    print(f"=== embed model: {EMBED_MODEL} ===")
    qd = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    ensure_collection(qd)

    queries = (
        expand_query_via_gemini(
            theme,
            bin_path=ask_gemini_bin,
            limit=expand_limit,
            timeout_sec=ask_gemini_timeout,
        )
        if expand_query
        else [theme]
    )

    t0 = time.time()
    seen_urls: set[str] = set()
    all_urls: list[str] = []
    for q in queries:
        for u in discover_urls_via_searxng(q, max_pages=max_pages):
            if u not in seen_urls:
                seen_urls.add(u)
                all_urls.append(u)
    # 全クエリで集約後、ドメイン優先度で再ソートし top-N 採用
    ranked = sorted(all_urls, key=lambda u: (-_domain_score(u), all_urls.index(u)))
    urls = ranked[:max_pages]
    discover_secs = time.time() - t0
    print(
        f"discover: {discover_secs:.1f}s, "
        f"{len(queries)} queries → {len(all_urls)} unique URLs → {len(urls)} 採用"
    )

    if not urls:
        print("[ERROR] no article URLs discovered", file=sys.stderr)
        return 1

    total_chunks = 0
    total_cleansed = 0
    for url in urls:
        print(f"--- fetching: {url}")
        page = await fetch_page(url)
        if page is None:
            continue
        md = htmlize_to_markdown(page.html)
        chunks = chunk_by_heading(md)
        n = upsert_page(qd, theme, page, chunks)
        total_chunks += n
        print(
            f"    title={page.title[:60]!r}  md={len(md)}文字  "
            f"chunks={len(chunks)} (upserted={n})"
        )

        if cleanse and chunks:
            chunk_payloads = [
                {
                    "text": c.text,
                    "heading_path": c.heading_path,
                    "url": page.url,
                    "theme": theme,
                    "chunk_idx": i,
                }
                for i, c in enumerate(chunks)
            ]
            cleanse_results = cleanse_batch(
                chunk_payloads,
                max_chunks=cleanse_max_chunks_per_page,
                per_chunk_timeout=cleanse_per_chunk_timeout,
                output_mode="new-collection",
                filter_mode=cleanse_filter_mode,
                filter_max_chars=cleanse_filter_max_chars,
                filter_heading_keywords=cleanse_filter_heading_keywords,
                body_min_chars=cleanse_body_min_chars,
                body_max_chars=cleanse_body_max_chars,
                per_url_top=cleanse_per_url_top,
                cosine_gate=cleanse_cosine_gate,
                clean_collection=cleanse_collection,
                qd_client=qd,
            )
            n_clean = sum(
                1 for r in cleanse_results
                if not r.skipped and r.cleansed_text.strip()
            )
            total_cleansed += n_clean
            print(f"    cleansed → {cleanse_collection}: {n_clean} chunks")

    info = qd.get_collection(COLLECTION)
    print(
        f"upserted total {total_chunks} chunks. "
        f"collection {COLLECTION} total points: {info.points_count}"
    )
    if cleanse:
        try:
            info_clean = qd.get_collection(cleanse_collection)
            print(
                f"cleansed total {total_cleansed} chunks. "
                f"collection {cleanse_collection} total points: "
                f"{info_clean.points_count}"
            )
        except Exception as e:
            print(f"[WARN] cannot stat {cleanse_collection}: {e}", file=sys.stderr)
    demo_search(qd, theme)
    return 0


async def amain(args: argparse.Namespace) -> int:
    if args.test_markdown:
        page = await fetch_page(args.test_markdown)
        if page is None:
            return 1
        md = htmlize_to_markdown(page.html)
        chunks = chunk_by_heading(md)
        print(f"=== {page.url} ===")
        print(f"title: {page.title}")
        print(f"markdown chars: {len(md)}")
        print(f"chunks: {len(chunks)}")
        for i, c in enumerate(chunks, 1):
            head = c.text[:120].replace("\n", " ")
            print(
                f"  [{i}] {c.heading_path!r}  ({len(c.text)}字)  | {head}"
            )
        return 0

    if args.search:
        qd = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        if not qd.collection_exists(COLLECTION):
            print(f"[ERROR] collection {COLLECTION} does not exist", file=sys.stderr)
            return 1
        demo_search(qd, args.search, top=args.top)
        return 0

    if args.themes_file:
        try:
            themes = parse_themes_file(args.themes_file)
        except FileNotFoundError:
            print(f"[ERROR] themes file not found: {args.themes_file}", file=sys.stderr)
            return 1
        if not themes:
            print(f"[ERROR] no themes in {args.themes_file}", file=sys.stderr)
            return 1

        try:
            lock_fp = open(args.lock_file, "w")
        except OSError as e:
            print(f"[ERROR] cannot open lock file {args.lock_file}: {e}", file=sys.stderr)
            return 1
        try:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                f"[ERROR] another instance is running (lock: {args.lock_file})",
                file=sys.stderr,
            )
            lock_fp.close()
            return 1

        failed: list[str] = []
        try:
            for i, theme in enumerate(themes, 1):
                print(f"\n=== [{i}/{len(themes)}] theme: {theme!r} ===")
                try:
                    rc = await run_research(
                        theme,
                        args.max_pages,
                        expand_query=args.expand_query,
                        expand_limit=args.expand_limit,
                        ask_gemini_bin=args.ask_gemini_bin,
                        ask_gemini_timeout=args.ask_gemini_timeout,
                        cleanse=args.cleanse,
                        cleanse_max_chunks_per_page=args.cleanse_max_chunks_per_page,
                        cleanse_per_chunk_timeout=args.cleanse_per_chunk_timeout,
                        cleanse_filter_max_chars=args.cleanse_filter_max_chars,
                        cleanse_filter_heading_keywords=_parse_cleanse_keywords(
                            args.cleanse_filter_heading_keywords
                        ),
                        cleanse_collection=args.cleanse_collection,
                        cleanse_filter_mode=args.cleanse_filter_mode,
                        cleanse_body_min_chars=args.cleanse_body_min_chars,
                        cleanse_body_max_chars=args.cleanse_body_max_chars,
                        cleanse_per_url_top=args.cleanse_per_url_top,
                        cleanse_cosine_gate=args.cleanse_cosine_gate,
                    )
                    if rc != 0:
                        failed.append(theme)
                        if args.strict:
                            print(
                                f"[ERROR] --strict: stopping on {theme!r}",
                                file=sys.stderr,
                            )
                            return rc
                except Exception as e:
                    failed.append(theme)
                    print(
                        f"[ERROR] theme {theme!r} raised "
                        f"{type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
                    if args.strict:
                        return 1
        finally:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
            lock_fp.close()

        if failed:
            print(
                f"\n[SUMMARY] {len(failed)}/{len(themes)} themes failed: "
                f"{failed}",
                file=sys.stderr,
            )
            return 1 if args.strict else 0
        print(f"\n[SUMMARY] all {len(themes)} themes succeeded")
        return 0

    if not args.theme:
        print(
            "[ERROR] theme is required "
            "(or use --themes-file / --test-markdown / --search)",
            file=sys.stderr,
        )
        return 2

    return await run_research(
        args.theme,
        args.max_pages,
        expand_query=args.expand_query,
        expand_limit=args.expand_limit,
        ask_gemini_bin=args.ask_gemini_bin,
        ask_gemini_timeout=args.ask_gemini_timeout,
        cleanse=args.cleanse,
        cleanse_max_chunks_per_page=args.cleanse_max_chunks_per_page,
        cleanse_per_chunk_timeout=args.cleanse_per_chunk_timeout,
        cleanse_filter_max_chars=args.cleanse_filter_max_chars,
        cleanse_filter_heading_keywords=_parse_cleanse_keywords(
            args.cleanse_filter_heading_keywords
        ),
        cleanse_collection=args.cleanse_collection,
        cleanse_filter_mode=args.cleanse_filter_mode,
        cleanse_body_min_chars=args.cleanse_body_min_chars,
        cleanse_body_max_chars=args.cleanse_body_max_chars,
        cleanse_per_url_top=args.cleanse_per_url_top,
        cleanse_cosine_gate=args.cleanse_cosine_gate,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Theme-driven web research → bge-m3 → Qdrant (Phase 2 v2)"
    )
    ap.add_argument("theme", nargs="?", default=None, help="research theme")
    ap.add_argument(
        "--max-pages", type=int, default=MAX_PAGES_DEFAULT,
        help="number of article pages to crawl",
    )
    ap.add_argument(
        "--search", default=None,
        help="run only similarity search against existing collection",
    )
    ap.add_argument("--top", type=int, default=5, help="top-k for --search")
    ap.add_argument(
        "--test-markdown", default=None,
        help="fetch one URL and dump markdown / chunks (no embed, no upsert)",
    )
    ap.add_argument(
        "--themes-file", default=None,
        help="path to themes file (1 theme per line, # for comments)",
    )
    ap.add_argument(
        "--lock-file", default="/tmp/web_research.lock",
        help="lock file path for multi-instance prevention (themes-file mode)",
    )
    ap.add_argument(
        "--strict", action="store_true",
        help="themes-file mode: stop on first failure (default: continue)",
    )
    ap.add_argument(
        "--expand-query", action="store_true",
        help="ask_gemini で検索ワード拡張 (default OFF, Phase 2.1c)",
    )
    ap.add_argument(
        "--expand-limit", type=int, default=5,
        help="--expand-query 時の最大拡張クエリ数 (default 5)",
    )
    ap.add_argument(
        "--ask-gemini-bin", default="ask_gemini",
        help="ask_gemini wrapper の path (default: PATH 検索)",
    )
    ap.add_argument(
        "--ask-gemini-timeout", type=int, default=30,
        help="ask_gemini 呼出 timeout 秒 (default 30)",
    )
    ap.add_argument(
        "--cleanse", action="store_true",
        help=(
            "page 取得後に cleanse_batch → web_brain_clean 投入 "
            "(Phase 2.5b2、default OFF)"
        ),
    )
    ap.add_argument(
        "--cleanse-max-chunks-per-page", type=int, default=5,
        help="--cleanse 時、1 page あたりの cleanse 最大 chunk 数 (default 5)",
    )
    ap.add_argument(
        "--cleanse-per-chunk-timeout", type=int, default=180,
        help="--cleanse の per-chunk timeout 秒 (default 180)",
    )
    ap.add_argument(
        "--cleanse-filter-max-chars", type=int, default=1500,
        help="--cleanse の filter: 短い chunk のみ採択 (default 1500、0 で無効)",
    )
    ap.add_argument(
        "--cleanse-filter-heading-keywords",
        default="nav,footer,sidebar,related,menu,breadcrumb",
        help="--cleanse の filter: comma-separated heading keywords",
    )
    ap.add_argument(
        "--cleanse-collection", default="web_brain_clean",
        help="cleanse 結果を投入する Qdrant collection (default web_brain_clean)",
    )
    # --- Phase 2.5b3 body-sample filter mode 引数 (cleanse_chunk.py へ pass-through) ---
    ap.add_argument(
        "--cleanse-filter-mode",
        choices=("nav-only", "body-sample", "all"), default="nav-only",
        help=(
            "filter mode (default nav-only): nav-only=既存 / "
            "body-sample=本文 chunk URL 別 top-N / all=フィルタ無効"
        ),
    )
    ap.add_argument(
        "--cleanse-body-min-chars", type=int, default=1000,
        help="body-sample mode: chars 下限 (default 1000)",
    )
    ap.add_argument(
        "--cleanse-body-max-chars", type=int, default=2000,
        help="body-sample mode: chars 上限 (default 2000、num_predict 1200 と整合)",
    )
    ap.add_argument(
        "--cleanse-per-url-top", type=int, default=3,
        help="body-sample mode: URL あたり代表 chunk 数 (default 3、0 で無効)",
    )
    ap.add_argument(
        "--cleanse-cosine-gate", type=float, default=0.9,
        help="upsert 時 cosine 閾値 (default 0.9、0.0 で gate 無効)",
    )
    args = ap.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
