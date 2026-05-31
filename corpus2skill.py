#!/usr/bin/env python3
"""corpus2skill.py — Phase 3 PoC: distill chunks → reusable Python skill.

Karpathy Layer 4 (Skills) の自動生成パイプライン。web_brain_clean (Layer 2,
Wiki Pages) からテーマ別 chunks を取り出し、deepseek-r1:14b で共通パターンを
抽出 → 単一 Python モジュール (skills/<theme>.py) を生成する。

Pipeline:
    1. fetch_chunks_by_theme  - Qdrant scroll, payload.theme フィルタ
    2. build_distillation_prompt
    3. call_14b (Ollama /api/generate, num_predict=NUM_PREDICT)
    4. strip_think → extract_code_block (```python ... ``` regex)
    5. validate_python_syntax (ast.parse)
    6. write_skill (skills/<slug>.py、由来 docstring header 付与)
    7. verify_import (subprocess で隔離 import + callable 列挙)

Retry: syntax error または code 抽出失敗時に最大 MAX_RETRIES 回まで、prompt に
       前回の失敗理由を付加して再試行。
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

OLLAMA_URL = "http://127.0.0.1:11434"
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
DEFAULT_COLLECTION = "web_brain_clean"
DEFAULT_MODEL = "deepseek-r1:14b"
DEFAULT_TIMEOUT = 600  # 14B コード生成は 5-10 min 想定 (input + think + output)
NUM_PREDICT = 2500  # 蒸留出力 ~1000 tokens + think 余地
SKILLS_DIR = Path(__file__).parent / "skills"
MAX_RETRIES = 3

DISTILL_PROMPT_TEMPLATE = """You are a code-skill distiller. Below are {n_chunks} Markdown chunks of cleaned documentation about "{theme}". Your task: extract the common, reusable patterns and output a single Python module with helper functions that future agents (Aider, browser-use) can import directly.

CRITICAL RULES:
- Do NOT output <think> blocks. Skip all chain-of-thought reasoning. Output Python code directly.
- Output ONLY a single ```python ... ``` code block. No commentary, explanation, or preamble outside it.
- The module MUST be importable: when Python imports this file, no error should occur.
- Output ONLY def/class definitions, type aliases, and imports at the top level. Do NOT include module-level executable statements that reference undefined names (no "X = SomeClass(undef_func)" example usage at the top level).
- Every name referenced in the code MUST be either imported at the top of the module, or defined as a def/class within the module. Do NOT use placeholder names like AnonymousUser / common_parameters unless you define them too.
- Include 1-5 reusable functions, each with a clear docstring (Args / Returns).
- Add a module-level docstring that names the theme and what the helpers do.
- Prefer pure Python stdlib + the framework being documented. Do not invent libraries.
- If you need to show example usage, put it inside `if __name__ == "__main__":` block at the very bottom; otherwise omit examples entirely.

--- BEGIN CHUNKS ---
{joined_chunks}
--- END CHUNKS ---

Output the Python module now (ONLY the ```python``` code block):"""


@dataclass(frozen=True)
class DistillResult:
    """Immutable distillation outcome."""

    theme: str
    n_chunks: int
    source_urls: tuple[str, ...]
    raw_response: str
    extracted_code: str
    valid: bool
    error: str | None
    skill_path: Path | None
    elapsed_sec: float
    attempts: int


def fetch_chunks_by_theme(
    qd: QdrantClient, collection: str, theme: str, limit: int = 50
) -> list[dict]:
    """Qdrant scroll で payload.theme が一致する全 chunks を返す。"""
    flt = Filter(
        must=[FieldCondition(key="theme", match=MatchValue(value=theme))]
    )
    results = qd.scroll(
        collection_name=collection,
        scroll_filter=flt,
        limit=limit,
        with_payload=True,
    )[0]
    return [p.payload for p in results]


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks (DeepSeek-R1 reasoning output)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


_CODE_BLOCK_PATTERN = re.compile(r"```(?:python)?\n?(.*?)```", re.DOTALL)


def extract_code_block(text: str) -> str | None:
    """Return the first ```python ... ``` block; fallback to raw text if it
    looks like Python (starts with import/def/class/docstring/comment)."""
    m = _CODE_BLOCK_PATTERN.search(text)
    if m:
        return m.group(1).strip()
    stripped = text.strip()
    fallback_prefixes = ('import ', 'from ', 'def ', 'class ', '"""', "'''", '#')
    if any(stripped.startswith(p) for p in fallback_prefixes):
        return stripped
    return None


def validate_python_syntax(code: str) -> tuple[bool, str | None]:
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def build_distillation_prompt(theme: str, chunks: list[dict]) -> str:
    joined = "\n\n--- CHUNK SEPARATOR ---\n\n".join(
        f"### Source: {c.get('url', 'unknown')}\n"
        f"### Heading: {c.get('heading_path', '')}\n\n"
        f"{c.get('text', '')}"
        for c in chunks
    )
    return DISTILL_PROMPT_TEMPLATE.format(
        theme=theme,
        n_chunks=len(chunks),
        joined_chunks=joined,
    )


def call_14b(prompt: str, *, model: str = DEFAULT_MODEL, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Ollama /api/generate に投げ、response 文字列を返す。"""
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": NUM_PREDICT},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json().get("response", "")


def theme_to_slug(theme: str) -> str:
    """'FastAPI dependency injection patterns' → 'fastapi_dependency_injection_patterns'."""
    s = re.sub(r"[^\w\s-]", "", theme.lower()).strip()
    return re.sub(r"\s+", "_", s)


def write_skill(
    theme: str,
    code: str,
    source_urls: list[str],
    chunks_count: int,
    model: str,
) -> Path:
    """skills/<slug>.py に header docstring + code を書き出す。"""
    SKILLS_DIR.mkdir(exist_ok=True)
    init_path = SKILLS_DIR / "__init__.py"
    if not init_path.exists():
        init_path.write_text(
            '"""Auto-distilled skills (Karpathy Layer 4).\n\n'
            'Generated by corpus2skill.py from web_brain_clean (Layer 2)."""\n',
            encoding="utf-8",
        )
    slug = theme_to_slug(theme)
    path = SKILLS_DIR / f"{slug}.py"
    urls_block = "\n".join(f"- {u}" for u in source_urls)
    header = (
        f'"""Auto-generated by corpus2skill.py (Phase 3).\n\n'
        f'Theme: {theme}\n'
        f'Source chunks: {chunks_count}\n'
        f'Source URLs:\n'
        f'{urls_block}\n'
        f'Generated: {datetime.now().isoformat(timespec="seconds")}\n'
        f'Model: {model}\n'
        f'\nDo not edit by hand; rerun corpus2skill.py to regenerate.\n"""\n\n'
    )
    path.write_text(header + code, encoding="utf-8")
    return path


def verify_import(skill_path: Path) -> tuple[bool, str | None, list[str]]:
    """subprocess で隔離 import + top-level callables を列挙。"""
    module_name = f"skills.{skill_path.stem}"
    code = (
        f"import {module_name} as m; "
        "print('CALLABLES:' + ','.join("
        "n for n in dir(m) "
        "if callable(getattr(m, n)) and not n.startswith('_')))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(__file__).parent),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        return False, (proc.stderr or "").strip()[:600], []
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if line.startswith("CALLABLES:"):
        items = [c for c in line[len("CALLABLES:"):].split(",") if c]
        return True, None, items
    return False, f"unexpected stdout: {line[:200]!r}", []


def distill(
    theme: str,
    *,
    collection: str = DEFAULT_COLLECTION,
    model: str = DEFAULT_MODEL,
    max_retries: int = MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    verify: bool = True,
) -> DistillResult:
    """蒸留 + L1 (syntax) + L2 (import) + L3 (callables ≥1) を retry loop に組み込む。

    verify=False のとき L1 のみ検査して即返却 (legacy --no-verify 用)。
    """
    qd = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    chunks = fetch_chunks_by_theme(qd, collection, theme)
    if not chunks:
        return DistillResult(
            theme=theme, n_chunks=0, source_urls=(), raw_response="",
            extracted_code="", valid=False,
            error=f"no chunks for theme {theme!r} in {collection}",
            skill_path=None, elapsed_sec=0.0, attempts=0,
        )

    source_urls = tuple(sorted({c.get("url", "") for c in chunks if c.get("url")}))
    print(f"[1/3] fetched {len(chunks)} chunks from {collection} (theme={theme!r})")
    print(f"      source URLs ({len(source_urls)}):")
    for u in source_urls:
        print(f"        - {u}")

    prompt = build_distillation_prompt(theme, chunks)
    print(f"[2/3] prompt size: {len(prompt)} chars")

    last_err: str | None = None
    raw = ""
    code = ""
    attempts = 0
    t0 = time.time()
    for attempt in range(1, max_retries + 1):
        attempts = attempt
        print(
            f"[3/3] attempt {attempt}/{max_retries}: "
            f"distilling via {model} (timeout={timeout}s)..."
        )
        try:
            raw = call_14b(prompt, model=model, timeout=timeout)
        except requests.exceptions.RequestException as e:
            last_err = f"14B request failed: {type(e).__name__}: {e}"
            print(f"      [ERROR] {last_err}", file=sys.stderr)
            continue
        cleaned = strip_think(raw)
        candidate = extract_code_block(cleaned)
        if not candidate:
            last_err = "no Python code block found in response"
            print(f"      [WARN] {last_err}, retrying with stricter prompt", file=sys.stderr)
            prompt = prompt + (
                "\n\nREMINDER: Output ONLY ```python ... ``` block, nothing else."
            )
            continue
        # L1: syntax
        syntax_ok, syntax_err = validate_python_syntax(candidate)
        if not syntax_ok:
            last_err = syntax_err
            code = candidate
            print(f"      [WARN] L1 syntax error: {syntax_err}, retrying", file=sys.stderr)
            prompt = prompt + (
                f"\n\nPREVIOUS ATTEMPT FAILED L1 (syntax): {syntax_err}. "
                f"Fix the syntax error and try again."
            )
            continue
        code = candidate
        print(f"      [OK] L1 syntax valid ({len(code)} chars)")

        if not verify:
            elapsed = time.time() - t0
            path = write_skill(theme, code, list(source_urls), len(chunks), model)
            print(f"      [OK] written (verify=False): {path}, {elapsed:.1f}s")
            return DistillResult(
                theme=theme, n_chunks=len(chunks), source_urls=source_urls,
                raw_response=raw, extracted_code=code, valid=True, error=None,
                skill_path=path, elapsed_sec=elapsed, attempts=attempt,
            )

        # L2: import 検証 (ファイルを実 path に書いて subprocess import)
        path = write_skill(theme, code, list(source_urls), len(chunks), model)
        imp_ok, imp_err, callables = verify_import(path)
        if not imp_ok:
            last_err = f"L2 import: {imp_err}"
            print(f"      [WARN] L2 import failed: {imp_err}", file=sys.stderr)
            prompt = prompt + (
                f"\n\nPREVIOUS ATTEMPT FAILED L2 (import): {imp_err}\n"
                f"Names referenced must all be either imported at top, or defined "
                f"as def/class in this module. Do NOT use undefined placeholder "
                f"names. Do NOT execute SomeClass(undef_name) at module level."
            )
            continue
        # L3: callables >= 1
        if not callables:
            last_err = "L3 no top-level callables"
            print(f"      [WARN] L3 no callables, retrying", file=sys.stderr)
            prompt = prompt + (
                "\n\nPREVIOUS ATTEMPT FAILED L3: the module imported but exposed "
                "no top-level def/class. You MUST define at least one reusable "
                "function or class."
            )
            continue
        # 全段 PASS
        elapsed = time.time() - t0
        print(f"      [OK] L2+L3 verified ({len(callables)} callable(s), {elapsed:.1f}s)")
        return DistillResult(
            theme=theme, n_chunks=len(chunks), source_urls=source_urls,
            raw_response=raw, extracted_code=code, valid=True, error=None,
            skill_path=path, elapsed_sec=elapsed, attempts=attempt,
        )

    elapsed = time.time() - t0
    return DistillResult(
        theme=theme, n_chunks=len(chunks), source_urls=source_urls,
        raw_response=raw, extracted_code=code, valid=False, error=last_err,
        skill_path=None, elapsed_sec=elapsed, attempts=attempts,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Phase 3 Corpus2Skill PoC: web_brain_clean からテーマ別 chunks を "
            "蒸留 → skills/<slug>.py 自動生成"
        )
    )
    ap.add_argument(
        "--theme", required=True,
        help="payload.theme と一致する theme 文字列 (exact match)",
    )
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    ap.add_argument(
        "--no-verify", action="store_true",
        help="生成後の subprocess import 検証を skip",
    )
    args = ap.parse_args()

    result = distill(
        args.theme,
        collection=args.collection,
        model=args.model,
        max_retries=args.max_retries,
        timeout=args.timeout,
        verify=not args.no_verify,
    )
    if not result.valid:
        print(
            f"\n[FAIL] distillation failed after {result.attempts} attempts: "
            f"{result.error}",
            file=sys.stderr,
        )
        return 1

    # callable 数を最終 SUMMARY 用に再列挙 (verify=True なら distill 内で確認済)
    callables: list[str] = []
    if not args.no_verify and result.skill_path is not None:
        _ok, _err, callables = verify_import(result.skill_path)

    print("\n=== SUMMARY ===")
    print(f"theme:    {result.theme}")
    print(f"chunks:   {result.n_chunks} from {len(result.source_urls)} URLs")
    print(f"skill:    {result.skill_path}")
    print(
        f"code:     {len(result.extracted_code)} chars, "
        f"{len(callables)} callable(s)"
    )
    print(f"callables: {callables}")
    print(f"attempts: {result.attempts}/{args.max_retries}")
    print(f"elapsed:  {result.elapsed_sec:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
