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

# --- Phase 3.2 quality-loop 定数 ---
DEFAULT_QUALITY_MAX_RETRIES = 2  # 品質ループの retry 回数 (L1-L3 retry とは別カウント)
DEFAULT_MYPY_BIN = "mypy"
DEFAULT_ASK_GEMINI_BIN = "ask_gemini"
DEFAULT_GEMINI_REVIEW_TIMEOUT = 90  # full skill review は時間がかかる

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
        # L1+L2+L3 全段 PASS
        print(f"      [OK] L2+L3 verified ({len(callables)} callable(s))")

        if not quality_loop:
            elapsed = time.time() - t0
            return DistillResult(
                theme=theme, n_chunks=len(chunks), source_urls=source_urls,
                raw_response=raw, extracted_code=code, valid=True, error=None,
                skill_path=path, elapsed_sec=elapsed, attempts=attempt,
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
                )
            final_feedback = feedback
            if q_attempt >= quality_max_retries:
                break
            q_attempt += 1
            print(
                f"      [L4+L5] quality retry {q_attempt}/{quality_max_retries}: "
                f"re-distilling with feedback"
            )
            q_prompt = prompt + (
                f"\n\nQUALITY ISSUES FROM PREVIOUS GENERATION:\n{feedback}\n\n"
                f"Regenerate the entire module fixing these specific issues. "
                f"Keep correct functions intact. Output ONLY ```python``` block."
            )
            try:
                raw = call_14b(q_prompt, model=model, timeout=timeout)
            except requests.exceptions.RequestException as e:
                print(
                    f"      [WARN] quality retry: 14B request failed: {e}",
                    file=sys.stderr,
                )
                break
            cleaned = strip_think(raw)
            candidate = extract_code_block(cleaned)
            if not candidate:
                print(
                    "      [WARN] quality retry: no code block, "
                    "abort quality loop",
                    file=sys.stderr,
                )
                break
            s_ok, s_err = validate_python_syntax(candidate)
            if not s_ok:
                print(
                    f"      [WARN] quality retry: syntax err {s_err}, "
                    f"abort quality loop",
                    file=sys.stderr,
                )
                break
            new_path = write_skill(
                theme, candidate, list(source_urls), len(chunks), model,
            )
            imp_ok, imp_err, new_callables = verify_import(new_path)
            if not imp_ok:
                print(
                    f"      [WARN] quality retry: import fail {imp_err}, "
                    f"restoring previous skill and aborting quality loop",
                    file=sys.stderr,
                )
                new_path.write_text(backup_text, encoding="utf-8")
                break
            if not new_callables:
                print(
                    "      [WARN] quality retry: no callables, "
                    "restoring previous skill and aborting quality loop",
                    file=sys.stderr,
                )
                new_path.write_text(backup_text, encoding="utf-8")
                break
            # この retry の skill は L1+L2+L3 PASS、次 retry のバックアップに昇格
            backup_text = new_path.read_text(encoding="utf-8")
            path = new_path
            code = candidate
            callables = new_callables

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
    args = ap.parse_args()

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
    if result.error:
        # quality_loop=True で valid=True だが品質残課題ありの warning
        print(f"warning:  {result.error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
