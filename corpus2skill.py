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
import dataclasses
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

# Phase 3.8b: 多モデルルーター (proposer + critic 並列)。default "none" 経路では
# import するだけで未使用なので副作用ゼロ、後方互換。
from router import (
    AsymmetricDebateStrategy,
    OllamaRunner,
    RouterResult,
    append_router_record,
    build_critic_prompt,
    build_router_record,
    format_critic_hint,
)

OLLAMA_URL = "http://127.0.0.1:11434"
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
DEFAULT_COLLECTION = "web_brain_clean"
DEFAULT_MODEL = "deepseek-r1:14b"
DEFAULT_TIMEOUT = 600  # 14B コード生成は 5-10 min 想定 (input + think + output)
NUM_PREDICT = 2500  # 蒸留出力 ~1000 tokens + think 余地
SKILLS_DIR = Path(__file__).parent / "skills"
MAX_RETRIES = 3

# --- Phase 3.2 quality-loop 定数 ---
DEFAULT_QUALITY_MAX_RETRIES = 2  # 品質ループの retry 回数 (L1-L3 retry とは別カウント)
DEFAULT_MYPY_BIN = "mypy"
DEFAULT_ASK_GEMINI_BIN = "ask_gemini"
DEFAULT_GEMINI_REVIEW_TIMEOUT = 90  # full skill review は時間がかかる

# --- Phase 3.3 RAG-augmented distillation 定数 ---
DEFAULT_RAG_COLLECTION = "web_brain"  # Layer 1 raw chunks (125 pts)
DEFAULT_RAG_TOP_K = 2  # 注入する追加 chunks 数 (prompt 増大による 14B 混乱回避)
DEFAULT_RAG_MAX_CHARS = 1000  # 各 RAG chunk の最大文字数 (truncate 後 ...)
RAG_EMBED_MODEL = "bge-m3"

# --- Phase 3.4 adaptive RAG 定数 ---
# schedule[q_attempt] = その q_attempt で注入する RAG top_k 数。0 = RAG OFF。
# default = "0→2→3" (初回 OFF → quality FAIL なら top_k=2 → さらに FAIL なら top_k=3)
DEFAULT_RAG_ADAPTIVE_SCHEDULE: tuple[int, ...] = (0, 2, 3)

# --- Phase 3.5 quality-retry inner retry 定数 ---
# quality retry 内で L2 (import) / L3 (callables) 失敗時、同じ q_attempt のまま
# corrective hint を prompt に追加して再生成する最大回数。0 = 従来動作 (一発外したら break)。
DEFAULT_QUALITY_INNER_RETRIES = 2

# --- Phase 3.6 model registry + multi-model fallback 定数 ---
# 将来の即乗せ換え用 model registry (cf. [[feedback_local_llm_research_orientation]])。
# 新モデルを試すときは: 1) Ollama pull <name>、2) ここに 1 entry 追加、
# 3) CLI で --primary-model <name> または --fallback-model <name> 指定。
# YAML/JSON 外部化は Phase 3.7+ で metrics 基盤と一緒に検討予定。
MODEL_REGISTRY: dict[str, dict[str, str | int]] = {
    "deepseek-r1:14b": {
        "role": "planner",
        "size_gb": 9,
        "notes": "reasoning 強、コード生成は弱点 (asyncio 系 import 幻覚多発、Phase 3.5 で観測)",
    },
    "qwen2.5-coder:7b": {
        "role": "coder",
        "size_gb": 5,
        "notes": "コード忠実性高、Aider で実証済、Phase 3.6 default fallback",
    },
    "qwen2.5:7b-instruct": {
        "role": "general",
        "size_gb": 5,
        "notes": "汎用、未評価",
    },
    "llama3.1:8b": {
        "role": "general",
        "size_gb": 5,
        "notes": "未評価",
    },
    "gemma2:9b-instruct-q4_K_M": {
        "role": "critic",
        "size_gb": 6,
        "notes": (
            "CPU only 推奨 (num_gpu=0)、Phase 3.8a で 8GB VRAM 環境の並列パートナーとして検証済 "
            "(NT6 sum_conc 9.39 tps)、日本語 9B 分水嶺、Phase 3.8b critic default"
        ),
    },
}
# 旧 DEFAULT_MODEL の意味的 alias (将来差替え時の単一変更点)
PRIMARY_MODEL = DEFAULT_MODEL

# 7B fallback (Phase 3.6 A 案: inner retry 最終段で 14B → fallback model に切替)。
# default OFF で完全後方互換、`--enable-7b-fallback` で opt-in。
DEFAULT_FALLBACK_MODEL = "qwen2.5-coder:7b"
DEFAULT_ENABLE_FALLBACK = False

# --- Phase 3.8b router/critic 定数 ---
# router_strategy="none" で従来挙動 (single primary model 呼出) を完全保持。
# "asymmetric_debate" で初回 attempt のみ proposer (primary) + critic を並列実行し、
# critic findings は router_runs.jsonl に記録、code 採用は proposer 出力 (PoC)。
DEFAULT_ROUTER_STRATEGY = "none"
DEFAULT_CRITIC_MODEL = "gemma2:9b-instruct-q4_K_M"
DEFAULT_ROUTER_METRICS_FILE = "metrics/router_runs.jsonl"
# Phase 3.8a NT6 verdict 固定値 — 半年先の再評価で別 sweet spot が出るまで固定。
ROUTER_NUM_THREAD = 6

# --- Phase 3.8c router-feedback (critic findings → proposer retry) ---
# router_feedback='none' で Phase 3.8b 完全互換 (router は attempt 1 のみ、inject なし)。
# 'on-retry'    : attempt 1 router の findings を memoize、attempts 2+ で 1 度だけ inject
#                  し以降 prompt 持ち回し (router は 1 回のみ稼働、コスト = 3.8b 同等)。
# 'every-attempt': 全 attempt で router 稼働、各 attempt の critic 出力を NEXT attempt の
#                  prompt に inject (毎回最新の findings に replace、累積はしない)。
#                  Gemini second-opinion 推し、Phase 3.8c の研究 default。
DEFAULT_ROUTER_FEEDBACK = "every-attempt"
ROUTER_FEEDBACK_CHOICES = ("none", "on-retry", "every-attempt")

QUALITY_REVIEW_PROMPT_TEMPLATE = """Review each public callable (top-level def/class, no leading underscore) in the following Python module.

Output EXACTLY one line per callable, format:
<name>: <USEFUL|OK_WITH_FIX|WRONG>: <one-line reason>

If WRONG or OK_WITH_FIX, briefly note the fix in the reason.
Do NOT include preamble, headers, or trailing summary. Only the verdict lines.

```python
{code}
```"""

# Gemini レスポンス行を parse する regex (numbered / bulleted / bold バリエーション許容)
_QUALITY_LINE_PATTERN = re.compile(
    r"""
    ^\s*
    (?:[-*]\s+|\d+\.\s*)?      # optional list marker
    (?:\*\*)?                    # optional bold open
    (?P<name>[\w]+)              # function name
    (?:\*\*)?                    # optional bold close
    \s*[:\-]\s*                  # separator
    (?:\*\*)?                    # optional bold open on verdict
    (?P<verdict>USEFUL|OK_WITH_FIX|WRONG)
    (?:\*\*)?                    # optional bold close
    \s*[:\-]?\s*                 # optional separator before reason
    (?P<reason>.*?)\s*$
    """,
    re.VERBOSE | re.IGNORECASE | re.MULTILINE,
)

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
{rag_section}
Output the Python module now (ONLY the ```python``` code block):"""

RAG_CONTEXT_SECTION_TEMPLATE = """
--- ADDITIONAL CONTEXT (raw doc chunks, use for API/syntax accuracy, do NOT copy verbatim) ---
{rag_joined}
--- END ADDITIONAL CONTEXT ---
"""


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
    rag_top_k_history: tuple[int, ...] = ()
    # Phase 3.5: 各 quality retry (q_attempt) 内で実行された inner retry 数の履歴。
    # 例 (0, 2) = q_attempt 0 で inner 不要、q_attempt 1 で 2 回 inner retry を消費。
    inner_retry_history: tuple[int, ...] = ()
    # Phase 3.6: inner retry 最終段で 7B fallback が発火したときの model 名 (例
    # 'qwen2.5-coder:7b')。未発火または --enable-7b-fallback OFF は None。
    fallback_model_used: str | None = None
    # Phase 3.8b: router strategy が初回 attempt で稼働したときの metadata。
    # router_strategy=None かつ wall_sec/findings_count=None で従来挙動 (single
    # model 呼出) を区別。
    router_strategy: str | None = None
    router_wall_sec: float | None = None
    router_critic_findings_count: int | None = None
    # Phase 3.8c: critic findings を proposer retry に feed する merge loop の
    # 実行 metadata。feedback mode == 'none' または router 未稼働なら None。
    # router_findings_injected_count = critic hint が proposer prompt に注入された
    # attempts 数 (on-retry なら最大 max_retries-1、every-attempt なら同上 + α)。
    router_feedback_mode: str | None = None
    router_findings_injected_count: int | None = None


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


# --- Phase 3.5 corrective hint extraction ---
# 14B が幻覚しやすい代表症状 4 種に対し、prompt に注入する 1 行 hint を生成する。
# Phase 3.3 v3 asyncio domain で観測された `module 'asyncio' has no attribute 'Coroutine'`
# のような symbol 幻覚を、retry 時に明示的に矯正する目的。

_TYPING_LIKE_MODULES = frozenset({"typing", "collections.abc", "typing_extensions"})

_IMPORT_ERROR_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # AttributeError: module 'X' has no attribute 'Y'[. Did you mean: 'Z'?]
    (
        re.compile(
            r"AttributeError: module '(?P<mod>[\w\.]+)' has no attribute "
            r"'(?P<attr>\w+)'(?:\. Did you mean: '(?P<suggest>\w+)'\?)?"
        ),
        "attr",
    ),
    # NameError: name 'X' is not defined
    (
        re.compile(r"NameError: name '(?P<name>\w+)' is not defined"),
        "name",
    ),
    # ModuleNotFoundError: No module named 'X'
    (
        re.compile(r"ModuleNotFoundError: No module named '(?P<mod>[\w\.]+)'"),
        "module",
    ),
    # ImportError: cannot import name 'X' from 'Y'
    (
        re.compile(
            r"ImportError: cannot import name '(?P<name>\w+)' from '(?P<mod>[\w\.]+)'"
        ),
        "import",
    ),
)


def extract_import_error_hint(err: str | None) -> str | None:
    """L2 import 失敗時のエラー文字列から、14B 向けの矯正 hint を 1 行生成する。

    既知 4 種 (AttributeError / NameError / ModuleNotFoundError / ImportError) を
    複数行 traceback の末尾優先でマッチ。マッチなしは None (上位で feedback のみ流す)。
    """
    if not err:
        return None
    for line in reversed(err.splitlines()):
        line = line.strip()
        for pat, kind in _IMPORT_ERROR_PATTERNS:
            m = pat.search(line)
            if not m:
                continue
            if kind == "attr":
                mod = m["mod"]
                attr = m["attr"]
                suggest = m.group("suggest")
                base = f"'{mod}.{attr}' does not exist."
                if suggest and suggest != attr:
                    base += f" Python suggested '{mod}.{suggest}' as the nearest name."
                if mod in _TYPING_LIKE_MODULES:
                    base += (
                        f" Verify the actual public API of '{mod}' "
                        "(e.g. use 'collections.abc' for runtime generic types)."
                    )
                else:
                    base += (
                        f" Verify the actual public API of '{mod}' — "
                        "for generic types use 'typing' or 'collections.abc' instead "
                        "(e.g. 'from typing import Coroutine')."
                    )
                return base
            if kind == "name":
                return (
                    f"Name '{m['name']}' is referenced but never imported or defined. "
                    "Either import it at the top of the module, define it as a def/class, "
                    "or remove the reference."
                )
            if kind == "module":
                return (
                    f"Module '{m['mod']}' does not exist or is not installed. "
                    "Use only stdlib or the framework being documented; "
                    "do not invent package names."
                )
            if kind == "import":
                return (
                    f"'{m['name']}' is not exported by '{m['mod']}'. "
                    f"Verify the actual public API of '{m['mod']}' before importing."
                )
    return None


def build_distillation_prompt(
    theme: str,
    chunks: list[dict],
    *,
    rag_chunks: list[dict] | None = None,
) -> str:
    joined = "\n\n--- CHUNK SEPARATOR ---\n\n".join(
        f"### Source: {c.get('url', 'unknown')}\n"
        f"### Heading: {c.get('heading_path', '')}\n\n"
        f"{c.get('text', '')}"
        for c in chunks
    )
    rag_section = ""
    if rag_chunks:
        rag_joined = "\n\n--- CHUNK SEPARATOR ---\n\n".join(
            f"### Source: {c.get('url', 'unknown')}\n"
            f"### Heading: {c.get('heading_path', '')}\n\n"
            f"{c.get('text', '')}"
            for c in rag_chunks
        )
        rag_section = RAG_CONTEXT_SECTION_TEMPLATE.format(rag_joined=rag_joined)
    return DISTILL_PROMPT_TEMPLATE.format(
        theme=theme,
        n_chunks=len(chunks),
        joined_chunks=joined,
        rag_section=rag_section,
    )


def embed_query(text: str, *, model: str = RAG_EMBED_MODEL) -> list[float]:
    """bge-m3 (default) で 1 query を embed。corpus2skill 専用の薄い wrapper。"""
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def retrieve_rag_chunks(
    qd: QdrantClient,
    query: str,
    *,
    collection: str = DEFAULT_RAG_COLLECTION,
    top_k: int = DEFAULT_RAG_TOP_K,
    max_chars: int = DEFAULT_RAG_MAX_CHARS,
) -> list[dict]:
    """bge-m3 で query を embed → collection に対し vector search top-K の
    payload を返す。max_chars > 0 なら各 payload['text'] を truncate。
    collection 不在/embed 失敗時は [] (上位で graceful skip)。"""
    try:
        vec = embed_query(query)
    except requests.exceptions.RequestException as e:
        print(f"[RAG] embed failed: {e}", file=sys.stderr)
        return []
    try:
        resp = qd.query_points(
            collection_name=collection,
            query=vec,
            limit=top_k,
            with_payload=True,
        )
    except Exception as e:
        print(f"[RAG] search failed: {type(e).__name__}: {e}", file=sys.stderr)
        return []
    payloads = [dict(p.payload) for p in resp.points]
    if max_chars > 0:
        for p in payloads:
            text = p.get("text", "")
            if len(text) > max_chars:
                p["text"] = text[:max_chars] + " ..."
    return payloads


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


def static_analyze(
    skill_path: Path,
    *,
    mypy_bin: str = DEFAULT_MYPY_BIN,
    timeout: int = 60,
) -> tuple[bool, list[str]]:
    """mypy を skill_path に走らせ、issue 一覧を返す。

    Returns: (ran_ok, issues)
        ran_ok=True かつ issues==[]: mypy clean
        ran_ok=True かつ issues!=[]: mypy detected problems
        ran_ok=False: mypy 不在 or 実行失敗 (graceful skip 用)
    """
    import shutil
    bin_path = shutil.which(mypy_bin)
    if not bin_path:
        return False, [f"mypy bin {mypy_bin!r} not found in PATH"]
    try:
        proc = subprocess.run(
            [
                bin_path, "--no-incremental", "--no-error-summary",
                "--show-column-numbers", "--no-color-output",
                str(skill_path),
            ],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, [f"mypy timeout after {timeout}s"]
    # mypy returncode: 0=clean, 1=errors, 2=other
    if proc.returncode not in (0, 1):
        return False, [
            f"mypy rc={proc.returncode}: "
            f"{(proc.stderr or proc.stdout or '').strip()[:300]}"
        ]
    issues = [
        line.strip()
        for line in (proc.stdout or "").splitlines()
        if line.strip()
        and any(tag in line for tag in ("error:", "warning:", "note:"))
    ]
    return True, issues


def parse_gemini_review(raw: str) -> list[tuple[str, str, str]]:
    """Gemini verdict response を (name, verdict, reason) リストに変換。
    USEFUL は除外 (problems のみ拾う)、OK_WITH_FIX と WRONG だけ返す。"""
    results: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for m in _QUALITY_LINE_PATTERN.finditer(raw):
        name = m.group("name")
        if name in seen:
            continue
        seen.add(name)
        verdict_raw = m.group("verdict").upper().replace(" ", "_")
        # 正規化: USEFUL / OK_WITH_FIX / WRONG のいずれか
        if verdict_raw not in ("USEFUL", "OK_WITH_FIX", "WRONG"):
            continue
        if verdict_raw == "USEFUL":
            continue
        results.append((name, verdict_raw, (m.group("reason") or "").strip()))
    return results


def gemini_review(
    skill_path: Path,
    *,
    ask_gemini_bin: str = DEFAULT_ASK_GEMINI_BIN,
    timeout: int = DEFAULT_GEMINI_REVIEW_TIMEOUT,
) -> tuple[bool, list[tuple[str, str, str]]]:
    """ask_gemini wrapper 経由で skill の品質レビュー取得。
    Returns: (ran_ok, issues_excluding_useful)"""
    if not skill_path.exists():
        return False, []
    code = skill_path.read_text(encoding="utf-8")
    prompt = QUALITY_REVIEW_PROMPT_TEMPLATE.format(code=code)
    try:
        proc = subprocess.run(
            [ask_gemini_bin, prompt],
            capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"      [WARN] gemini_review skipped: {type(e).__name__}", file=sys.stderr)
        return False, []
    if proc.returncode != 0:
        print(
            f"      [WARN] gemini_review rc={proc.returncode}: "
            f"{(proc.stderr or '').strip()[:200]}",
            file=sys.stderr,
        )
        return False, []
    return True, parse_gemini_review(proc.stdout)


# --- Phase 3.7 metrics JSONL writer ---
# 各 distill() 実行の結果と config snapshot を 1 行 JSONL で追記し、run-to-run
# 変動や model × theme × schedule の比較を後段で集計可能にする。
# raw_response / extracted_code は冗長 (skill_path から物理ファイル復元可) のため
# JSONL では除外し、ファイル肥大化を抑制する。schema 進化は schema_version で管理。

# Phase 3.8b: schema v3 = v2 + DistillResult.router_strategy/wall_sec/critic_findings_count
# (default None で v2 parser 互換)、config snapshot に router_strategy/critic_model 追加。
# Phase 3.8c: schema v4 = v3 + DistillResult.router_feedback_mode/findings_injected_count
# (default None で v3 parser 互換、additive のみ)、config snapshot に router_feedback 追加。
METRICS_SCHEMA_VERSION = 4


def _safe_int_subproc(cmd: list[str], timeout: int = 3) -> int | None:
    """subprocess で先頭行の int を取得。失敗は全て None で吸収 (記録の degrade を許容)。"""
    try:
        out = subprocess.check_output(
            cmd, text=True, stderr=subprocess.DEVNULL, timeout=timeout
        )
        return int(out.strip().splitlines()[0])
    except (
        subprocess.SubprocessError,
        FileNotFoundError,
        OSError,
        ValueError,
        IndexError,
    ):
        return None


def _safe_ram_used_mb() -> int | None:
    """/proc/meminfo から (MemTotal - MemAvailable) MB を取得。"""
    try:
        info: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            parts = line.split(":")
            if len(parts) == 2:
                info[parts[0].strip()] = int(parts[1].split()[0])  # kB
        return (info["MemTotal"] - info["MemAvailable"]) // 1024
    except (OSError, ValueError, KeyError):
        return None


def _safe_loadavg() -> list[float] | None:
    try:
        return list(os.getloadavg())
    except OSError:
        return None


def _safe_concurrent_models(
    base_url: str = "http://127.0.0.1:11434", timeout: int = 3
) -> list[str] | None:
    """Ollama /api/ps で同時 load 中の model name list を取得 (stdlib のみ)。
    Ollama 未起動・network 失敗は None で degrade。"""
    try:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            f"{base_url}/api/ps", headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        return [
            m["name"] for m in data.get("models", []) if isinstance(m.get("name"), str)
        ]
    except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError):
        return None


def _collect_system_snapshot() -> dict:
    """nvidia-smi / meminfo / loadavg / Ollama loaded models の現在値 snapshot。
    Phase 3.7d で schema_version=2 から追加。各項目は失敗時 None で degrade。"""
    vram = _safe_int_subproc(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    )
    return {
        "vram_used_mb": vram,
        "nvidia_smi_available": vram is not None,
        "ram_used_mb": _safe_ram_used_mb(),
        "loadavg": _safe_loadavg(),
        "concurrent_models": _safe_concurrent_models(),
    }


def _safe_git_rev() -> str | None:
    """git rev-parse HEAD を安全に取得。git 未インストール / 非 git dir / timeout
    すべて None で吸収。"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def _safe_hostname() -> str | None:
    try:
        import socket
        return socket.gethostname()
    except OSError:
        return None


def _build_metrics_record(
    result: "DistillResult",
    args: argparse.Namespace,
    started_at: str,
    system_baseline: dict | None = None,
) -> dict:
    """DistillResult + CLI config snapshot + 環境メタ + system snapshot を 1 dict にまとめる。
    Path は str 化、tuple は list 化 (json.dumps 互換)。
    Phase 3.7d (schema_version=2): system_baseline / system_end snapshot を追加。"""
    d = dataclasses.asdict(result)
    # 重複かつ大きいフィールドを drop
    d.pop("raw_response", None)
    d.pop("extracted_code", None)
    # Path → str
    if d.get("skill_path") is not None:
        d["skill_path"] = str(d["skill_path"])
    # CLI config snapshot — 再現性に効く knob のみ抜粋
    d["config"] = {
        "primary_model": args.model,  # dest=model (--primary-model alias)
        "fallback_model": args.fallback_model,
        "enable_fallback": args.enable_7b_fallback,
        "quality_loop": args.quality_loop,
        "quality_max_retries": args.quality_max_retries,
        "quality_inner_retries": args.quality_inner_retries,
        "rag_augmented": args.rag_augmented,
        "rag_adaptive": args.rag_adaptive,
        "rag_top_k": args.rag_top_k,
        "rag_max_chars": args.rag_max_chars,
        "rag_adaptive_schedule": args.rag_adaptive_schedule,
        "collection": args.collection,
        "max_retries": args.max_retries,
        "timeout": args.timeout,
        # Phase 3.8b: router 設定 snapshot
        "router_strategy": args.router_strategy,
        "critic_model": args.critic_model,
        "router_metrics_file": args.router_metrics_file,
        # Phase 3.8c: critic→proposer feedback mode
        "router_feedback": args.router_feedback,
    }
    # 環境メタ
    d["meta"] = {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _safe_git_rev(),
        "host": _safe_hostname(),
        "schema_version": METRICS_SCHEMA_VERSION,
    }
    # Phase 3.7d: system snapshot (baseline は distill 開始前に取得、end は本記録時)
    d["system"] = {
        "baseline": system_baseline if system_baseline is not None else {},
        "end": _collect_system_snapshot(),
    }
    return d


def _append_metrics_jsonl(path: Path, record: dict) -> bool:
    """1 行 JSONL を append。親 directory が無ければ作成。
    エラーは warn のみで main 処理を阻害しない。成功時 True を返す。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except OSError as e:
        print(
            f"[WARN] metrics 書出し失敗 ({path}): {e}、distill 結果は正常",
            file=sys.stderr,
        )
        return False


def _join_chunks_for_critic(chunks: list[dict]) -> str:
    """build_distillation_prompt と同じ join 形式で critic 用 chunks を直列化。"""
    return "\n\n--- CHUNK SEPARATOR ---\n\n".join(
        f"### Source: {c.get('url', 'unknown')}\n"
        f"### Heading: {c.get('heading_path', '')}\n\n"
        f"{c.get('text', '')}"
        for c in chunks
    )


def _run_asymmetric_debate(
    *,
    theme: str,
    chunks: list[dict],
    proposer_prompt: str,
    proposer_model: str,
    critic_model: str,
    timeout: int,
    router_metrics_file: Path | None,
) -> RouterResult:
    """proposer + critic を並列実行し、router_runs.jsonl に 1 record 追記する。

    Phase 3.8b PoC #1: critic は chunks のみを独立評価し、proposer の出力は
    見ない (完全並列、wall ≈ max(proposer, critic))。chosen_text は proposer
    出力を採用、critic findings は metrics に記録のみ (next phase で feedback)。
    """
    critic_prompt = build_critic_prompt(
        theme=theme,
        n_chunks=len(chunks),
        joined_chunks=_join_chunks_for_critic(chunks),
    )
    # NT6 verdict (Phase 3.8a): num_thread=6 が並列 sum_conc 最大スイートスポット。
    # critic は gemma2:9b で CPU only (num_gpu=0)、proposer は default GPU。
    proposer_runner = OllamaRunner(
        role="proposer",
        model_id=proposer_model,
        timeout_sec=float(timeout),
        default_options={"num_thread": ROUTER_NUM_THREAD, "num_predict": NUM_PREDICT},
    )
    critic_runner = OllamaRunner(
        role="critic",
        model_id=critic_model,
        timeout_sec=float(timeout),
        default_options={
            "num_thread": ROUTER_NUM_THREAD,
            "num_gpu": 0,
            "num_predict": NUM_PREDICT,
            "temperature": 0.1,
        },
    )
    print(
        f"      [router] asymmetric_debate: proposer={proposer_model!r} (GPU) | "
        f"critic={critic_model!r} (CPU only, num_gpu=0)、num_thread={ROUTER_NUM_THREAD}"
    )
    result = AsymmetricDebateStrategy().route(
        proposer_prompt=proposer_prompt,
        critic_prompt=critic_prompt,
        proposer=proposer_runner,
        critic=critic_runner,
    )
    print(
        f"      [router] wall={result.parallel_wall_sec:.1f}s "
        f"proposer_eval={result.proposer_output.eval_count} "
        f"critic_eval={result.critic_output.eval_count} "
        f"findings={len(result.critic_findings)}"
    )
    if router_metrics_file is not None:
        rec = build_router_record(
            result=result,
            theme=theme,
            options={
                "num_thread": ROUTER_NUM_THREAD,
                "num_predict": NUM_PREDICT,
                "critic_num_gpu": 0,
                "proposer_model": proposer_model,
                "critic_model": critic_model,
            },
            repo_dir=Path(__file__).resolve().parent,
        )
        try:
            append_router_record(router_metrics_file, rec)
            print(
                f"      [router] appended to {router_metrics_file}",
                file=sys.stderr,
            )
        except OSError as e:
            print(
                f"      [WARN] router_runs.jsonl 書出し失敗 ({router_metrics_file}): {e}、"
                "distill は継続",
                file=sys.stderr,
            )
    return result


def quality_check(
    skill_path: Path,
    *,
    mypy_bin: str = DEFAULT_MYPY_BIN,
    mypy_timeout: int = 60,
    ask_gemini_bin: str = DEFAULT_ASK_GEMINI_BIN,
    gemini_timeout: int = DEFAULT_GEMINI_REVIEW_TIMEOUT,
) -> tuple[bool, str]:
    """L4 (mypy) + L5 (Gemini review) を統合。

    Returns: (passes, feedback)
        passes=True: mypy clean かつ Gemini issue 無し (skip 含む)
        passes=False: いずれかで issue 検出、feedback は 14B retry 用の追記文字列
    """
    feedback_parts: list[str] = []

    print("      [L4] running mypy static analysis...")
    mypy_ok, mypy_issues = static_analyze(
        skill_path, mypy_bin=mypy_bin, timeout=mypy_timeout,
    )
    if mypy_ok and mypy_issues:
        feedback_parts.append(
            "STATIC ANALYSIS (mypy) issues:\n"
            + "\n".join(f"  - {iss}" for iss in mypy_issues[:10])
        )
        print(f"           {len(mypy_issues)} mypy issue(s)")
    elif mypy_ok:
        print("           mypy clean")
    else:
        msg = mypy_issues[0] if mypy_issues else "unknown"
        print(f"           mypy skipped: {msg}")

    print("      [L5] requesting Gemini quality review...")
    gem_ok, gem_issues = gemini_review(
        skill_path, ask_gemini_bin=ask_gemini_bin, timeout=gemini_timeout,
    )
    if gem_ok and gem_issues:
        feedback_parts.append(
            "GEMINI REVIEW issues (preserve USEFUL functions intact):\n"
            + "\n".join(
                f"  - {name}: {verdict}: {reason}"
                for name, verdict, reason in gem_issues
            )
        )
        print(f"           {len(gem_issues)} Gemini issue(s):")
        for name, verdict, reason in gem_issues:
            print(f"             {name}: {verdict}: {reason[:80]}")
    elif gem_ok:
        print("           Gemini all USEFUL")
    else:
        print("           Gemini skipped/unavailable")

    if not feedback_parts:
        return True, ""
    return False, "\n\n".join(feedback_parts)


def distill(
    theme: str,
    *,
    collection: str = DEFAULT_COLLECTION,
    model: str = DEFAULT_MODEL,
    max_retries: int = MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    verify: bool = True,
    quality_loop: bool = False,
    quality_max_retries: int = DEFAULT_QUALITY_MAX_RETRIES,
    mypy_bin: str = DEFAULT_MYPY_BIN,
    ask_gemini_bin: str = DEFAULT_ASK_GEMINI_BIN,
    gemini_timeout: int = DEFAULT_GEMINI_REVIEW_TIMEOUT,
    rag_augmented: bool = False,
    rag_top_k: int = DEFAULT_RAG_TOP_K,
    rag_collection: str = DEFAULT_RAG_COLLECTION,
    rag_max_chars: int = DEFAULT_RAG_MAX_CHARS,
    rag_adaptive: bool = False,
    rag_schedule: tuple[int, ...] = DEFAULT_RAG_ADAPTIVE_SCHEDULE,
    quality_inner_retries: int = DEFAULT_QUALITY_INNER_RETRIES,
    enable_fallback: bool = DEFAULT_ENABLE_FALLBACK,
    fallback_model: str = DEFAULT_FALLBACK_MODEL,
    router_strategy: str = DEFAULT_ROUTER_STRATEGY,
    critic_model: str = DEFAULT_CRITIC_MODEL,
    router_metrics_file: Path | None = None,
    router_feedback: str = DEFAULT_ROUTER_FEEDBACK,
) -> DistillResult:
    """蒸留 + L1 (syntax) + L2 (import) + L3 (callables ≥1) を retry loop に組み込む。

    verify=False のとき L1 のみ検査して即返却 (legacy --no-verify 用)。

    rag_adaptive=True のとき (Phase 3.4):
        rag_schedule[q_attempt] が各 quality retry で注入する RAG top_k を決める。
        schedule[0]=0 で初回 RAG OFF → quality_check FAIL なら schedule[1] 個注入 → さらに
        FAIL なら schedule[2] 個注入...という adaptive 戦略。Kubernetes 型ケース
        (RAG 自体が逆効果) を初回 PASS で救済する目的。rag_augmented とは排他。
    """
    if rag_augmented and rag_adaptive:
        raise ValueError(
            "rag_augmented と rag_adaptive は排他。どちらか片方のみ指定可。"
        )
    if rag_adaptive and not rag_schedule:
        raise ValueError("rag_adaptive=True のとき rag_schedule は非空必須")
    if router_feedback not in ROUTER_FEEDBACK_CHOICES:
        raise ValueError(
            f"router_feedback must be one of {ROUTER_FEEDBACK_CHOICES}, "
            f"got {router_feedback!r}"
        )
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

    rag_chunks: list[dict] = []
    rag_top_k_history: list[int] = []
    inner_retry_history: list[int] = []  # Phase 3.5: 各 q_attempt の inner retry 数
    fallback_model_used: str | None = None  # Phase 3.6: 7B fallback 発火時の model 名
    if rag_augmented:
        rag_chunks = retrieve_rag_chunks(
            qd, theme,
            collection=rag_collection, top_k=rag_top_k, max_chars=rag_max_chars,
        )
        rag_urls = sorted({c.get("url", "") for c in rag_chunks if c.get("url")})
        print(
            f"      [RAG] retrieved {len(rag_chunks)} additional chunks "
            f"from {rag_collection} ({len(rag_urls)} unique URLs)"
        )
        for u in rag_urls:
            print(f"        + {u}")
        rag_top_k_history.append(len(rag_chunks))
    elif rag_adaptive:
        initial_top_k = rag_schedule[0]
        if initial_top_k > 0:
            rag_chunks = retrieve_rag_chunks(
                qd, theme,
                collection=rag_collection, top_k=initial_top_k,
                max_chars=rag_max_chars,
            )
            rag_urls = sorted({c.get("url", "") for c in rag_chunks if c.get("url")})
            print(
                f"      [RAG-adaptive] initial top_k={initial_top_k}: "
                f"retrieved {len(rag_chunks)} chunks from {rag_collection} "
                f"({len(rag_urls)} unique URLs)"
            )
            for u in rag_urls:
                print(f"        + {u}")
        else:
            print(
                f"      [RAG-adaptive] schedule={list(rag_schedule)}, "
                f"initial top_k=0 (RAG OFF for first attempt)"
            )
        rag_top_k_history.append(initial_top_k)

    prompt = build_distillation_prompt(theme, chunks, rag_chunks=rag_chunks)
    print(f"[2/3] prompt size: {len(prompt)} chars")

    last_err: str | None = None
    raw = ""
    code = ""
    attempts = 0
    # Phase 3.8b: router state、attempt==1 で router_strategy != "none" のとき
    # のみ書き込まれる。"none" 経路では全て None のまま DistillResult に流れる。
    router_strategy_used: str | None = None
    router_wall_sec_used: float | None = None
    router_findings_count_used: int | None = None
    # Phase 3.8c: critic findings → proposer retry merge loop state。
    # last_critic_findings = 直前 attempt の critic 出力 (memoize)、空 tuple は
    # まだ critic 未実行か feedback=='none' (注入なし)。
    last_critic_findings: tuple[str, ...] = ()
    router_feedback_mode_used: str | None = None
    router_findings_injected_count_used: int = 0
    t0 = time.time()
    for attempt in range(1, max_retries + 1):
        attempts = attempt
        print(
            f"[3/3] attempt {attempt}/{max_retries}: "
            f"distilling via {model} (timeout={timeout}s)..."
        )
        # Phase 3.8c: critic hint overlay。prompt 自体は L1/L2/L3 feedback で
        # cumulative 変異させ、critic hint は per-attempt overlay で末尾追加
        # (毎 attempt 最新 findings に replace、累積はしない)。
        if router_feedback != "none" and last_critic_findings:
            critic_hint = format_critic_hint(last_critic_findings)
            effective_prompt = prompt + critic_hint
            if router_feedback_mode_used is None:
                router_feedback_mode_used = router_feedback
            router_findings_injected_count_used += 1
            print(
                f"      [router-feedback] {router_feedback} mode: injected "
                f"{len(last_critic_findings)} critic findings into proposer prompt"
            )
        else:
            effective_prompt = prompt
        # Phase 3.8b/3.8c: router strategy を稼働する attempt を decide。
        # Phase 3.8b 互換: feedback='none' のとき attempt==1 のみ router (legacy)。
        # Phase 3.8c on-retry: attempt==1 のみ router、findings は次以降の inject へ。
        # Phase 3.8c every-attempt: 全 attempt で router 稼働、findings 毎回 refresh。
        router_runs_this_attempt = (
            router_strategy == "asymmetric_debate"
            and (attempt == 1 or router_feedback == "every-attempt")
        )
        if router_runs_this_attempt:
            try:
                router_result = _run_asymmetric_debate(
                    theme=theme,
                    chunks=chunks,
                    proposer_prompt=effective_prompt,
                    proposer_model=model,
                    critic_model=critic_model,
                    timeout=timeout,
                    router_metrics_file=router_metrics_file,
                )
            except Exception as e:  # noqa: BLE001 — graceful degrade to single-model retry
                last_err = f"router failed: {type(e).__name__}: {e}"
                print(
                    f"      [WARN] router failed, falling back to single-model: {last_err}",
                    file=sys.stderr,
                )
                continue
            raw = router_result.chosen_text
            if attempt == 1:
                # Phase 3.8b legacy fields は attempt 1 router の値を保持 (後方互換)。
                router_strategy_used = router_result.strategy_name
                router_wall_sec_used = router_result.parallel_wall_sec
                router_findings_count_used = len(router_result.critic_findings)
            # Phase 3.8c: NEXT attempt の proposer prompt に inject するため memo。
            # feedback='none' でも router が走ったら memo するが、overlay 適用は
            # router_feedback != 'none' のときのみ (条件は loop 先頭で再評価)。
            last_critic_findings = router_result.critic_findings
        else:
            try:
                raw = call_14b(effective_prompt, model=model, timeout=timeout)
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
                rag_top_k_history=tuple(rag_top_k_history),
                inner_retry_history=tuple(inner_retry_history),
                fallback_model_used=fallback_model_used,
                router_strategy=router_strategy_used,
                router_wall_sec=router_wall_sec_used,
                router_critic_findings_count=router_findings_count_used,
                router_feedback_mode=router_feedback_mode_used,
                router_findings_injected_count=router_findings_injected_count_used,
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
        # L1+L2+L3 全段 PASS
        print(f"      [OK] L2+L3 verified ({len(callables)} callable(s))")

        if not quality_loop:
            elapsed = time.time() - t0
            return DistillResult(
                theme=theme, n_chunks=len(chunks), source_urls=source_urls,
                raw_response=raw, extracted_code=code, valid=True, error=None,
                skill_path=path, elapsed_sec=elapsed, attempts=attempt,
                rag_top_k_history=tuple(rag_top_k_history),
                inner_retry_history=tuple(inner_retry_history),
                fallback_model_used=fallback_model_used,
                router_strategy=router_strategy_used,
                router_wall_sec=router_wall_sec_used,
                router_critic_findings_count=router_findings_count_used,
                router_feedback_mode=router_feedback_mode_used,
                router_findings_injected_count=router_findings_injected_count_used,
            )

        # Phase 3.2 quality loop: L4 (mypy) + L5 (Gemini) を最大 quality_max_retries 回
        print(
            f"      [L4+L5] starting quality loop "
            f"(max {quality_max_retries} retries)"
        )
        # 各 retry の skill ファイルが import fail で path を破壊しないよう、
        # 直前の "L1+L2+L3 PASS skill" のテキストをバックアップとして保持。
        backup_text: str = path.read_text(encoding="utf-8")
        final_feedback = ""
        q_attempt = 0
        while True:
            passes, feedback = quality_check(
                path,
                mypy_bin=mypy_bin, mypy_timeout=60,
                ask_gemini_bin=ask_gemini_bin, gemini_timeout=gemini_timeout,
            )
            if passes:
                elapsed = time.time() - t0
                print(
                    f"      [OK] quality passed at q_attempt {q_attempt} "
                    f"(elapsed {elapsed:.1f}s)"
                )
                return DistillResult(
                    theme=theme, n_chunks=len(chunks), source_urls=source_urls,
                    raw_response=raw, extracted_code=code, valid=True, error=None,
                    skill_path=path, elapsed_sec=elapsed, attempts=attempt,
                    rag_top_k_history=tuple(rag_top_k_history),
                    inner_retry_history=tuple(inner_retry_history),
                    fallback_model_used=fallback_model_used,
                    router_strategy=router_strategy_used,
                    router_wall_sec=router_wall_sec_used,
                    router_critic_findings_count=router_findings_count_used,
                    router_feedback_mode=router_feedback_mode_used,
                    router_findings_injected_count=router_findings_injected_count_used,
                )
            final_feedback = feedback
            if q_attempt >= quality_max_retries:
                break
            q_attempt += 1
            print(
                f"      [L4+L5] quality retry {q_attempt}/{quality_max_retries}: "
                f"re-distilling with feedback"
            )
            # Phase 3.4: adaptive モード時、schedule[q_attempt] に従い RAG を再注入
            if rag_adaptive and q_attempt < len(rag_schedule):
                new_top_k = rag_schedule[q_attempt]
                if new_top_k > 0:
                    adapt_rag_chunks = retrieve_rag_chunks(
                        qd, theme,
                        collection=rag_collection,
                        top_k=new_top_k, max_chars=rag_max_chars,
                    )
                    adapt_urls = sorted(
                        {c.get("url", "") for c in adapt_rag_chunks if c.get("url")}
                    )
                    print(
                        f"      [RAG-adaptive] retry {q_attempt}: top_k={new_top_k} "
                        f"→ {len(adapt_rag_chunks)} chunks "
                        f"({len(adapt_urls)} unique URLs)"
                    )
                    for u in adapt_urls:
                        print(f"        + {u}")
                else:
                    adapt_rag_chunks = []
                    print(
                        f"      [RAG-adaptive] retry {q_attempt}: "
                        f"top_k=0 (RAG OFF)"
                    )
                rag_top_k_history.append(new_top_k)
                # base prompt を schedule の RAG 強度で再構築
                prompt = build_distillation_prompt(
                    theme, chunks, rag_chunks=adapt_rag_chunks,
                )
                print(f"      [RAG-adaptive] prompt size: {len(prompt)} chars")
            q_prompt_base = prompt + (
                f"\n\nQUALITY ISSUES FROM PREVIOUS GENERATION:\n{feedback}\n\n"
                f"Regenerate the entire module fixing these specific issues. "
                f"Keep correct functions intact. Output ONLY ```python``` block."
            )
            # --- Phase 3.5 inner retry: L2/L3 fail 時に corrective hint で
            # 同じ q_attempt のまま最大 quality_inner_retries 回再生成。
            # 全 inner 使い切ったら従来通り backup 復元 + outer break (fail-safe 維持)。
            inner_succeeded = False
            inner_abort = False           # 14B request 失敗等の不可逆系
            inner_used = 0                # 実際に消費した inner retry 回数
            last_inner_err: str = ""
            inner_prompt = q_prompt_base
            for inner_attempt in range(quality_inner_retries + 1):
                # Phase 3.6: 最終 inner retry (inner_attempt == quality_inner_retries)
                # かつ inner_attempt > 0 (= retry 1 回以上消費後の最終段) のとき、
                # --enable-7b-fallback で fallback_model に切替。inner_attempt == 0 を
                # 除外することで quality_inner_retries=0 の縮退ケースが 7B 即発火に
                # ならないことを保護 (backward compat)。
                active_model = model
                is_final_inner = (
                    inner_attempt > 0
                    and inner_attempt == quality_inner_retries
                )
                if enable_fallback and is_final_inner:
                    active_model = fallback_model
                    fallback_model_used = active_model
                if inner_attempt > 0:
                    inner_used = inner_attempt
                    fallback_note = (
                        f" [FALLBACK {model} → {active_model}]"
                        if active_model != model else ""
                    )
                    print(
                        f"      [L4+L5 inner] retry "
                        f"{inner_attempt}/{quality_inner_retries}: "
                        f"re-generating with corrective hint{fallback_note}"
                    )
                try:
                    raw = call_14b(inner_prompt, model=active_model, timeout=timeout)
                except requests.exceptions.RequestException as e:
                    print(
                        f"      [WARN] quality retry: 14B request failed: {e}",
                        file=sys.stderr,
                    )
                    inner_abort = True
                    break
                cleaned = strip_think(raw)
                candidate = extract_code_block(cleaned)
                if not candidate:
                    last_inner_err = "no Python code block in 14B response"
                    print(
                        f"      [WARN] quality inner: {last_inner_err}",
                        file=sys.stderr,
                    )
                    inner_prompt = q_prompt_base + (
                        "\n\nPREVIOUS INNER ATTEMPT FAILED: produced no ```python``` "
                        "code block. Output ONLY a single ```python``` block, nothing else."
                    )
                    continue
                s_ok, s_err = validate_python_syntax(candidate)
                if not s_ok:
                    last_inner_err = f"L1 syntax: {s_err}"
                    print(
                        f"      [WARN] quality inner: {last_inner_err}",
                        file=sys.stderr,
                    )
                    inner_prompt = q_prompt_base + (
                        f"\n\nPREVIOUS INNER ATTEMPT FAILED L1 (syntax): {s_err}. "
                        "Fix the syntax error and try again."
                    )
                    continue
                new_path = write_skill(
                    theme, candidate, list(source_urls), len(chunks), model,
                )
                imp_ok, imp_err, new_callables = verify_import(new_path)
                if not imp_ok:
                    last_inner_err = f"L2 import: {imp_err}"
                    hint = extract_import_error_hint(imp_err)
                    print(
                        f"      [WARN] quality inner: import fail "
                        f"{(imp_err or '')[:160]}",
                        file=sys.stderr,
                    )
                    if hint:
                        print(f"      [hint] {hint}", file=sys.stderr)
                    # corrective hint があれば prompt 先頭に明示挿入、なければ raw error のみ
                    hint_block = (
                        f"\n\nPREVIOUS INNER ATTEMPT FAILED L2 (import):\n"
                        f"{(imp_err or '').strip()}\n"
                    )
                    if hint:
                        hint_block += f"\nCORRECTIVE GUIDANCE: {hint}\n"
                    hint_block += (
                        "Names referenced must all be imported at the top or "
                        "defined as def/class in this module. Do NOT invent "
                        "symbols that do not exist in the named module."
                    )
                    inner_prompt = q_prompt_base + hint_block
                    # 退避: 失敗 skill は backup で上書き (path 自体は次 inner で再生成)
                    new_path.write_text(backup_text, encoding="utf-8")
                    continue
                if not new_callables:
                    last_inner_err = "L3 no callables"
                    print(
                        f"      [WARN] quality inner: {last_inner_err}",
                        file=sys.stderr,
                    )
                    inner_prompt = q_prompt_base + (
                        "\n\nPREVIOUS INNER ATTEMPT FAILED L3: imported but exposed "
                        "no top-level def/class. You MUST define at least one "
                        "reusable function or class at module top level."
                    )
                    new_path.write_text(backup_text, encoding="utf-8")
                    continue
                # L1+L2+L3 全 PASS、次 outer q_attempt のバックアップに昇格
                backup_text = new_path.read_text(encoding="utf-8")
                path = new_path
                code = candidate
                callables = new_callables
                inner_succeeded = True
                break
            inner_retry_history.append(inner_used)
            if inner_abort:
                # 14B 通信障害等は inner で復活不能、quality loop 終了
                break
            if not inner_succeeded:
                # inner 全消費しても回復せず → 従来動作: backup 復元 + outer break
                print(
                    f"      [WARN] quality inner: exhausted "
                    f"{quality_inner_retries} retries (last err: "
                    f"{last_inner_err[:120]}), restoring previous skill and "
                    f"aborting quality loop",
                    file=sys.stderr,
                )
                path.write_text(backup_text, encoding="utf-8")
                break
            # inner_succeeded=True 時の path/code/callables 更新は inner loop 内で完了済

        # 全 quality retry 後も issue 残る、最後の skill 保持で valid=True (warning)
        elapsed = time.time() - t0
        print(
            f"      [WARN] quality loop ended with issues after "
            f"{q_attempt} retries"
        )
        return DistillResult(
            theme=theme, n_chunks=len(chunks), source_urls=source_urls,
            raw_response=raw, extracted_code=code, valid=True,
            error=f"quality issues remained: {final_feedback[:300]}",
            skill_path=path, elapsed_sec=elapsed, attempts=attempt,
            rag_top_k_history=tuple(rag_top_k_history),
            inner_retry_history=tuple(inner_retry_history),
            fallback_model_used=fallback_model_used,
            router_strategy=router_strategy_used,
            router_wall_sec=router_wall_sec_used,
            router_critic_findings_count=router_findings_count_used,
            router_feedback_mode=router_feedback_mode_used,
            router_findings_injected_count=router_findings_injected_count_used,
        )

    elapsed = time.time() - t0
    return DistillResult(
        theme=theme, n_chunks=len(chunks), source_urls=source_urls,
        raw_response=raw, extracted_code=code, valid=False, error=last_err,
        skill_path=None, elapsed_sec=elapsed, attempts=attempts,
        rag_top_k_history=tuple(rag_top_k_history),
        inner_retry_history=tuple(inner_retry_history),
        fallback_model_used=fallback_model_used,
        router_strategy=router_strategy_used,
        router_wall_sec=router_wall_sec_used,
        router_critic_findings_count=router_findings_count_used,
        router_feedback_mode=router_feedback_mode_used,
        router_findings_injected_count=router_findings_injected_count_used,
    )


def main() -> int:
    # Phase 3.7: 実行開始時刻 (UTC ISO 8601)、metrics JSONL の started_at field 用
    started_at = datetime.now(timezone.utc).isoformat()
    # Phase 3.7d: distill 開始前の system snapshot (baseline)、record build 時に
    # end snapshot と対比する。collect 失敗は None で degrade、record には空 dict で残す。
    system_baseline = _collect_system_snapshot()
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
    # Phase 3.6: --primary-model が canonical、--model は後方互換 alias (dest=model 維持)
    ap.add_argument(
        "--primary-model", "--model", dest="model", default=PRIMARY_MODEL,
        help=(
            f"primary distillation model (default {PRIMARY_MODEL!r}、MODEL_REGISTRY 参照)。"
            "新モデル即乗せ換えのため --primary-model を canonical 名として推奨、"
            "--model は backward compatible alias。"
        ),
    )
    ap.add_argument(
        "--list-models", action="store_true",
        help="MODEL_REGISTRY を表示して exit (model 選定の参考用、Phase 3.6)",
    )
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    ap.add_argument(
        "--no-verify", action="store_true",
        help="生成後の subprocess import 検証を skip",
    )
    # --- Phase 3.2 quality-loop 引数 ---
    ap.add_argument(
        "--quality-loop", action="store_true",
        help=(
            "L1-L3 通過後に L4 (mypy) + L5 (Gemini review) で追加品質ガード、"
            "問題検出時は 14B に feedback して再蒸留 (default OFF、Phase 3.2)"
        ),
    )
    ap.add_argument(
        "--quality-max-retries", type=int, default=DEFAULT_QUALITY_MAX_RETRIES,
        help=f"品質ループの retry 回数 (default {DEFAULT_QUALITY_MAX_RETRIES})",
    )
    ap.add_argument(
        "--mypy-bin", default=DEFAULT_MYPY_BIN,
        help=f"mypy バイナリ名 (default {DEFAULT_MYPY_BIN!r}、不在時 graceful skip)",
    )
    ap.add_argument(
        "--ask-gemini-bin", default=DEFAULT_ASK_GEMINI_BIN,
        help=f"ask_gemini wrapper の path (default {DEFAULT_ASK_GEMINI_BIN!r})",
    )
    ap.add_argument(
        "--gemini-timeout", type=int, default=DEFAULT_GEMINI_REVIEW_TIMEOUT,
        help=f"Gemini review timeout 秒 (default {DEFAULT_GEMINI_REVIEW_TIMEOUT})",
    )
    # --- Phase 3.3 RAG-augmented distillation 引数 ---
    ap.add_argument(
        "--rag-augmented", action="store_true",
        help=(
            "bge-m3 + Qdrant vector search で関連 chunks を取得し、"
            "14B prompt に ADDITIONAL CONTEXT として注入 (default OFF、Phase 3.3)"
        ),
    )
    ap.add_argument(
        "--rag-top-k", type=int, default=DEFAULT_RAG_TOP_K,
        help=f"RAG retrieval top-K (default {DEFAULT_RAG_TOP_K})",
    )
    ap.add_argument(
        "--rag-collection", default=DEFAULT_RAG_COLLECTION,
        help=f"RAG retrieval source collection (default {DEFAULT_RAG_COLLECTION!r})",
    )
    ap.add_argument(
        "--rag-max-chars", type=int, default=DEFAULT_RAG_MAX_CHARS,
        help=(
            f"各 RAG chunk の最大文字数 (default {DEFAULT_RAG_MAX_CHARS}、"
            "prompt 肥大化抑制、0 で無効)"
        ),
    )
    # --- Phase 3.4 adaptive RAG 引数 ---
    ap.add_argument(
        "--rag-adaptive", action="store_true",
        help=(
            "適応的 RAG: 初回 RAG OFF → quality_check FAIL のとき "
            "schedule に従って RAG 強度を段階的に上げる (Phase 3.4)。"
            "--rag-augmented と排他。--quality-loop 併用前提。"
        ),
    )
    ap.add_argument(
        "--rag-adaptive-schedule",
        default=",".join(str(n) for n in DEFAULT_RAG_ADAPTIVE_SCHEDULE),
        help=(
            f"adaptive top_k schedule カンマ区切り "
            f"(default {','.join(str(n) for n in DEFAULT_RAG_ADAPTIVE_SCHEDULE)!r})。"
            "schedule[q_attempt] = その retry での top_k、0 = RAG OFF。"
        ),
    )
    # --- Phase 3.5 inner retry 引数 ---
    ap.add_argument(
        "--quality-inner-retries", type=int,
        default=DEFAULT_QUALITY_INNER_RETRIES,
        help=(
            f"quality retry 内で L2 (import) / L3 (callables) fail 時、"
            f"同じ q_attempt のまま corrective hint で再生成する最大回数 "
            f"(default {DEFAULT_QUALITY_INNER_RETRIES}、0=従来動作、Phase 3.5)"
        ),
    )
    # --- Phase 3.6 multi-model fallback 引数 ---
    ap.add_argument(
        "--enable-7b-fallback", action="store_true",
        help=(
            "quality inner retry 最終段 (inner_attempt == quality_inner_retries かつ "
            ">0) で primary model → fallback model に切替 (Phase 3.6、default OFF)"
        ),
    )
    ap.add_argument(
        "--fallback-model", default=DEFAULT_FALLBACK_MODEL,
        help=(
            f"fallback 時に使う Ollama model "
            f"(default {DEFAULT_FALLBACK_MODEL!r}、MODEL_REGISTRY 参照)"
        ),
    )
    # --- Phase 3.7 metrics JSONL 引数 ---
    ap.add_argument(
        "--metrics-file", default="metrics/distill_runs.jsonl",
        help=(
            "distill 結果を 1 行 JSONL で append するパス "
            "(default 'metrics/distill_runs.jsonl'、Phase 3.7)。"
            "raw_response / extracted_code は除外、skill_path から物理復元可。"
        ),
    )
    ap.add_argument(
        "--no-metrics", action="store_true",
        help="metrics JSONL 書出しを skip (Phase 3.7)",
    )
    # --- Phase 3.8b router/critic 引数 ---
    ap.add_argument(
        "--router-strategy",
        choices=["none", "asymmetric_debate"],
        default=DEFAULT_ROUTER_STRATEGY,
        help=(
            f"multi-model orchestration strategy (default {DEFAULT_ROUTER_STRATEGY!r}、"
            "Phase 3.8b)。'asymmetric_debate' で初回 attempt のみ proposer (primary "
            "model) + critic を並列実行 (Phase 3.8a NT6 verdict 利用)、critic findings "
            "は router_runs.jsonl に記録のみ (PoC、merge は Phase 3.8c)。"
        ),
    )
    ap.add_argument(
        "--critic-model",
        default=DEFAULT_CRITIC_MODEL,
        help=(
            f"critic 役の Ollama model (default {DEFAULT_CRITIC_MODEL!r}、"
            "MODEL_REGISTRY 参照)。CPU only (num_gpu=0) で動かす想定。"
        ),
    )
    ap.add_argument(
        "--router-metrics-file",
        default=DEFAULT_ROUTER_METRICS_FILE,
        help=(
            f"router 実行の 1 record を append する JSONL パス "
            f"(default {DEFAULT_ROUTER_METRICS_FILE!r}、Phase 3.8b)。"
            "--no-router-metrics で書出し抑止。"
        ),
    )
    ap.add_argument(
        "--no-router-metrics", action="store_true",
        help="router_runs.jsonl 書出しを skip (Phase 3.8b)",
    )
    ap.add_argument(
        "--router-feedback",
        choices=list(ROUTER_FEEDBACK_CHOICES),
        default=DEFAULT_ROUTER_FEEDBACK,
        help=(
            f"critic findings → proposer retry merge loop mode "
            f"(default {DEFAULT_ROUTER_FEEDBACK!r}、Phase 3.8c)。"
            "'none' で Phase 3.8b 互換 (inject なし)、"
            "'on-retry' で attempt 1 router → findings memoize → attempts 2+ で 1 度 inject、"
            "'every-attempt' で 全 attempt router 稼働 + 各 attempt の findings を次回 prompt に inject。"
            "router_strategy='none' のときは無視される。"
        ),
    )
    args = ap.parse_args()

    # --list-models: registry を表示して即 exit (Phase 3.6)
    if args.list_models:
        print("Phase 3.6 MODEL_REGISTRY:")
        for name, meta in MODEL_REGISTRY.items():
            marker = (
                " (PRIMARY default)" if name == PRIMARY_MODEL
                else " (FALLBACK default)" if name == DEFAULT_FALLBACK_MODEL
                else ""
            )
            print(
                f"  {name:30s}  role={meta['role']:8s}  "
                f"size={meta['size_gb']}GB{marker}"
            )
            print(f"    notes: {meta['notes']}")
        return 0

    # --rag-adaptive-schedule を tuple[int, ...] へ parse
    try:
        rag_schedule_parsed: tuple[int, ...] = tuple(
            int(x.strip()) for x in args.rag_adaptive_schedule.split(",")
            if x.strip()
        )
    except ValueError as e:
        print(
            f"[ERROR] --rag-adaptive-schedule parse 失敗: {e} "
            f"(value={args.rag_adaptive_schedule!r}、カンマ区切り int を期待)",
            file=sys.stderr,
        )
        return 2
    if args.rag_adaptive and not rag_schedule_parsed:
        print(
            "[ERROR] --rag-adaptive-schedule が空、最低 1 つの int 必須",
            file=sys.stderr,
        )
        return 2
    if any(n < 0 for n in rag_schedule_parsed):
        print(
            f"[ERROR] --rag-adaptive-schedule に負値: {rag_schedule_parsed}",
            file=sys.stderr,
        )
        return 2
    if args.quality_inner_retries < 0:
        print(
            f"[ERROR] --quality-inner-retries は 0 以上必須 "
            f"(got {args.quality_inner_retries})",
            file=sys.stderr,
        )
        return 2

    # Phase 3.8b: router_metrics_file は --no-router-metrics で None 化
    router_metrics_path: Path | None = (
        None if args.no_router_metrics else Path(args.router_metrics_file)
    )
    result = distill(
        args.theme,
        collection=args.collection,
        model=args.model,
        max_retries=args.max_retries,
        timeout=args.timeout,
        verify=not args.no_verify,
        quality_loop=args.quality_loop,
        quality_max_retries=args.quality_max_retries,
        mypy_bin=args.mypy_bin,
        ask_gemini_bin=args.ask_gemini_bin,
        gemini_timeout=args.gemini_timeout,
        rag_augmented=args.rag_augmented,
        rag_top_k=args.rag_top_k,
        rag_collection=args.rag_collection,
        rag_max_chars=args.rag_max_chars,
        rag_adaptive=args.rag_adaptive,
        rag_schedule=rag_schedule_parsed,
        quality_inner_retries=args.quality_inner_retries,
        enable_fallback=args.enable_7b_fallback,
        fallback_model=args.fallback_model,
        router_strategy=args.router_strategy,
        critic_model=args.critic_model,
        router_metrics_file=router_metrics_path,
        router_feedback=args.router_feedback,
    )

    # Phase 3.7: metrics JSONL 書出し (成功/失敗どちらも記録、--no-metrics で無効化)。
    # distillation の成功手前で書くことで、後段の return 1 / 0 どちらの path でも記録される。
    if not args.no_metrics:
        record = _build_metrics_record(result, args, started_at, system_baseline)
        if _append_metrics_jsonl(Path(args.metrics_file), record):
            print(f"metrics:  appended to {args.metrics_file}", file=sys.stderr)

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
    if result.rag_top_k_history:
        print(f"rag_top_k: {list(result.rag_top_k_history)}")
    if result.inner_retry_history:
        print(f"inner_retry: {list(result.inner_retry_history)}")
    if result.fallback_model_used:
        print(f"fallback: {result.fallback_model_used} (fired at inner retry final)")
    print(f"elapsed:  {result.elapsed_sec:.1f}s")
    if result.error:
        # quality_loop=True で valid=True だが品質残課題ありの warning
        print(f"warning:  {result.error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
