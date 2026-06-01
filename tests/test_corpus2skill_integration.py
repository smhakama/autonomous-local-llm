"""Phase 3.8b: corpus2skill.py × router integration tests.

These tests stay strictly outside the Qdrant / Ollama dependencies. The
end-to-end smoke that actually runs distill() against a real backend
lives in Commit 3 (manual smoke), not here.

Scope:
    - DistillResult router_* fields are optional and default to None.
    - _build_metrics_record exposes router fields and bumps schema_version.
    - CLI accepts --router-strategy choices, rejects unknown values, and
      defaults to "none" (preserving full backwards compatibility).
    - _run_asymmetric_debate helper builds the right runner shape and
      writes router_runs.jsonl when a metrics path is provided.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import corpus2skill
from corpus2skill import (
    DEFAULT_CRITIC_MODEL,
    DEFAULT_ROUTER_FEEDBACK,
    DEFAULT_ROUTER_METRICS_FILE,
    DEFAULT_ROUTER_STRATEGY,
    METRICS_SCHEMA_VERSION,
    MODEL_REGISTRY,
    ROUTER_FEEDBACK_CHOICES,
    ROUTER_NUM_THREAD,
    ROUTER_STRATEGY_CHOICES,
    DistillResult,
    _build_metrics_record,
    _run_asymmetric_debate,
    _run_sequential_critic_review,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_SCRIPT = REPO_ROOT / "corpus2skill.py"


# -------------------------------------------------------------------------
# DistillResult — new optional router fields
# -------------------------------------------------------------------------


def test_distill_result_router_fields_default_to_none() -> None:
    r = DistillResult(
        theme="x",
        n_chunks=0,
        source_urls=(),
        raw_response="",
        extracted_code="",
        valid=False,
        error=None,
        skill_path=None,
        elapsed_sec=0.0,
        attempts=0,
    )
    assert r.router_strategy is None
    assert r.router_wall_sec is None
    assert r.router_critic_findings_count is None


def test_distill_result_router_fields_round_trip() -> None:
    r = DistillResult(
        theme="x",
        n_chunks=1,
        source_urls=(),
        raw_response="",
        extracted_code="",
        valid=True,
        error=None,
        skill_path=None,
        elapsed_sec=1.0,
        attempts=1,
        router_strategy="asymmetric_debate",
        router_wall_sec=76.8,
        router_critic_findings_count=7,
    )
    assert r.router_strategy == "asymmetric_debate"
    assert r.router_wall_sec == 76.8
    assert r.router_critic_findings_count == 7


# -------------------------------------------------------------------------
# MODEL_REGISTRY — gemma critic entry
# -------------------------------------------------------------------------


def test_model_registry_contains_gemma_critic() -> None:
    assert DEFAULT_CRITIC_MODEL in MODEL_REGISTRY
    entry = MODEL_REGISTRY[DEFAULT_CRITIC_MODEL]
    assert entry["role"] == "critic"


def test_router_num_thread_matches_phase_38a_verdict() -> None:
    """NT6 verdict (Phase 3.8a) is wired in as the router default."""
    assert ROUTER_NUM_THREAD == 6


# -------------------------------------------------------------------------
# _build_metrics_record — schema v3 + router fields
# -------------------------------------------------------------------------


def _make_args() -> argparse.Namespace:
    """Mimic argparse.Namespace with every field _build_metrics_record reads."""
    return argparse.Namespace(
        model="deepseek-r1:14b",
        fallback_model="qwen2.5-coder:7b",
        enable_7b_fallback=False,
        quality_loop=False,
        quality_max_retries=2,
        quality_inner_retries=2,
        rag_augmented=False,
        rag_adaptive=False,
        rag_top_k=2,
        rag_max_chars=1000,
        rag_adaptive_schedule="0,2,3",
        collection="web_brain_clean",
        max_retries=3,
        timeout=600,
        router_strategy="asymmetric_debate",
        critic_model=DEFAULT_CRITIC_MODEL,
        router_metrics_file="metrics/router_runs.jsonl",
        router_feedback=DEFAULT_ROUTER_FEEDBACK,
    )


def test_metrics_schema_version_bumped_to_four() -> None:
    """Phase 3.8c: schema v4 (additive router_feedback_mode + injected_count)."""
    assert METRICS_SCHEMA_VERSION == 4


def test_build_metrics_record_includes_router_config_snapshot() -> None:
    result = DistillResult(
        theme="kubernetes",
        n_chunks=5,
        source_urls=("https://example.invalid/k8s",),
        raw_response="r",
        extracted_code="c",
        valid=True,
        error=None,
        skill_path=Path("/tmp/skill.py"),
        elapsed_sec=80.0,
        attempts=1,
        router_strategy="asymmetric_debate",
        router_wall_sec=76.8,
        router_critic_findings_count=7,
    )
    rec = _build_metrics_record(
        result=result,
        args=_make_args(),
        started_at="2026-06-01T18:00:00+00:00",
        system_baseline={"mem_used_mb": 1234},
    )
    assert rec["router_strategy"] == "asymmetric_debate"
    assert rec["router_wall_sec"] == 76.8
    assert rec["router_critic_findings_count"] == 7
    assert rec["config"]["router_strategy"] == "asymmetric_debate"
    assert rec["config"]["critic_model"] == DEFAULT_CRITIC_MODEL
    assert rec["config"]["router_metrics_file"] == "metrics/router_runs.jsonl"
    assert rec["config"]["router_feedback"] == DEFAULT_ROUTER_FEEDBACK
    assert rec["meta"]["schema_version"] == 4


def test_build_metrics_record_router_fields_default_null() -> None:
    """When router_strategy='none' was used, fields are null in the record."""
    result = DistillResult(
        theme="x",
        n_chunks=1,
        source_urls=(),
        raw_response="",
        extracted_code="",
        valid=True,
        error=None,
        skill_path=None,
        elapsed_sec=1.0,
        attempts=1,
    )
    args = _make_args()
    args.router_strategy = "none"
    rec = _build_metrics_record(
        result=result,
        args=args,
        started_at="2026-06-01T18:00:00+00:00",
        system_baseline=None,
    )
    assert rec["router_strategy"] is None
    assert rec["router_wall_sec"] is None
    assert rec["router_critic_findings_count"] is None
    assert rec["config"]["router_strategy"] == "none"


# -------------------------------------------------------------------------
# CLI — argparse behaviour (subprocess invocation, no Qdrant)
# -------------------------------------------------------------------------


def _run_corpus(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CORPUS_SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


def test_cli_default_router_strategy_is_none() -> None:
    """--theme dummy --list-models runs the early-exit branch with default args."""
    proc = _run_corpus(["--theme", "dummy", "--list-models"])
    assert proc.returncode == 0, proc.stderr
    # gemma critic line must appear in --list-models output
    assert "gemma2:9b-instruct-q4_K_M" in proc.stdout
    assert DEFAULT_ROUTER_STRATEGY == "none"


def test_cli_help_advertises_all_router_flags() -> None:
    proc = _run_corpus(["--help"])
    assert proc.returncode == 0, proc.stderr
    text = proc.stdout
    assert "--router-strategy" in text
    # Phase 3.8c+: choices expand to 3 (none, asymmetric_debate, sequential_critic_review).
    assert "none" in text and "asymmetric_debate" in text
    assert "sequential_critic_review" in text
    assert "--critic-model" in text
    assert "--router-metrics-file" in text
    assert "--no-router-metrics" in text
    # Phase 3.8c: --router-feedback flag with 3 choices.
    assert "--router-feedback" in text
    assert "{none,on-retry,every-attempt}" in text


def test_router_strategy_choices_constant() -> None:
    """Phase 3.8c+: source-of-truth tuple — single edit point on next add."""
    assert ROUTER_STRATEGY_CHOICES == (
        "none", "asymmetric_debate", "sequential_critic_review",
    )


def test_cli_rejects_unknown_router_feedback() -> None:
    proc = _run_corpus(
        ["--theme", "dummy", "--router-feedback", "bogus_mode"]
    )
    assert proc.returncode != 0
    assert "invalid choice" in proc.stderr.lower() or "bogus_mode" in proc.stderr


def test_default_router_feedback_is_every_attempt() -> None:
    """Phase 3.8c default: every-attempt (Gemini second-opinion 推し、研究方針)。"""
    assert DEFAULT_ROUTER_FEEDBACK == "every-attempt"
    assert "every-attempt" in ROUTER_FEEDBACK_CHOICES
    assert "on-retry" in ROUTER_FEEDBACK_CHOICES
    assert "none" in ROUTER_FEEDBACK_CHOICES


def test_cli_rejects_unknown_router_strategy() -> None:
    proc = _run_corpus(
        ["--theme", "dummy", "--router-strategy", "bogus_strategy"]
    )
    assert proc.returncode != 0
    assert "invalid choice" in proc.stderr.lower() or "bogus_strategy" in proc.stderr


def test_default_router_metrics_file_constant() -> None:
    assert DEFAULT_ROUTER_METRICS_FILE == "metrics/router_runs.jsonl"


# -------------------------------------------------------------------------
# _run_asymmetric_debate — runner shape + metrics writer integration
# -------------------------------------------------------------------------


_OLLAMA_RESPONSE_PROPOSER = {
    "response": "```python\ndef helper():\n    pass\n```",
    "prompt_eval_count": 200,
    "eval_count": 500,
    "eval_duration": 50_000_000_000,
    "prompt_eval_duration": 5_000_000_000,
    "total_duration": 56_000_000_000,
    "load_duration": 200_000_000,
}

_OLLAMA_RESPONSE_CRITIC = {
    "response": "- pitfall A\n- pitfall B\n- pitfall C\n",
    "prompt_eval_count": 180,
    "eval_count": 100,
    "eval_duration": 20_000_000_000,
    "prompt_eval_duration": 4_000_000_000,
    "total_duration": 25_000_000_000,
    "load_duration": 150_000_000,
}


def _make_mock_response(payload: dict[str, object]) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _route_by_model(payload_map: dict[str, dict[str, object]]):
    """Return a fake requests.post that picks payload by request body's model."""
    def fake_post(url: str, *, json: dict, timeout: float) -> MagicMock:  # noqa: A002
        model = json["model"]
        if model not in payload_map:
            raise AssertionError(f"unexpected model in test: {model!r}")
        return _make_mock_response(payload_map[model])
    return fake_post


def test_run_asymmetric_debate_writes_router_record(tmp_path: Path) -> None:
    target = tmp_path / "router_runs.jsonl"
    chunks = [
        {"url": "https://example.invalid/a", "heading_path": "Intro", "text": "alpha"},
        {"url": "https://example.invalid/b", "heading_path": "Refs", "text": "beta"},
    ]
    fake_post = _route_by_model({
        "deepseek-r1:14b": _OLLAMA_RESPONSE_PROPOSER,
        DEFAULT_CRITIC_MODEL: _OLLAMA_RESPONSE_CRITIC,
    })

    with patch("router.runners.requests.post", side_effect=fake_post):
        result = _run_asymmetric_debate(
            theme="kubernetes",
            chunks=chunks,
            proposer_prompt="distill this",
            proposer_model="deepseek-r1:14b",
            critic_model=DEFAULT_CRITIC_MODEL,
            timeout=600,
            router_metrics_file=target,
        )

    assert result.strategy_name == "asymmetric_debate"
    assert result.proposer_output.model_id == "deepseek-r1:14b"
    assert result.critic_output.model_id == DEFAULT_CRITIC_MODEL
    assert result.chosen_text.startswith("```python")
    assert result.critic_findings == ("pitfall A", "pitfall B", "pitfall C")

    assert target.exists()
    rec = json.loads(target.read_text(encoding="utf-8").strip())
    assert rec["schema_version"] == 1  # router_runs.jsonl 独自スキーマ
    assert rec["strategy_name"] == "asymmetric_debate"
    assert rec["theme"] == "kubernetes"
    assert rec["critic_findings_count"] == 3
    assert rec["options"]["num_thread"] == ROUTER_NUM_THREAD
    assert rec["options"]["critic_num_gpu"] == 0
    assert rec["options"]["proposer_model"] == "deepseek-r1:14b"


def test_run_asymmetric_debate_no_metrics_file_skips_write(tmp_path: Path) -> None:
    chunks = [{"url": "u", "heading_path": "h", "text": "t"}]
    fake_post = _route_by_model({
        "deepseek-r1:14b": _OLLAMA_RESPONSE_PROPOSER,
        DEFAULT_CRITIC_MODEL: _OLLAMA_RESPONSE_CRITIC,
    })
    with patch("router.runners.requests.post", side_effect=fake_post):
        result = _run_asymmetric_debate(
            theme="x",
            chunks=chunks,
            proposer_prompt="p",
            proposer_model="deepseek-r1:14b",
            critic_model=DEFAULT_CRITIC_MODEL,
            timeout=600,
            router_metrics_file=None,
        )
    assert result.strategy_name == "asymmetric_debate"
    # nothing to assert on disk — the point is that None path is graceful


def test_run_asymmetric_debate_passes_num_thread_six_to_both_runners(
    tmp_path: Path,
) -> None:
    chunks = [{"url": "u", "heading_path": "h", "text": "t"}]
    captured_payloads: list[dict] = []

    def capturing_post(url: str, *, json: dict, timeout: float) -> MagicMock:  # noqa: A002
        captured_payloads.append(json)
        return _make_mock_response(
            _OLLAMA_RESPONSE_PROPOSER
            if json["model"] == "deepseek-r1:14b"
            else _OLLAMA_RESPONSE_CRITIC
        )

    with patch("router.runners.requests.post", side_effect=capturing_post):
        _run_asymmetric_debate(
            theme="x",
            chunks=chunks,
            proposer_prompt="p",
            proposer_model="deepseek-r1:14b",
            critic_model=DEFAULT_CRITIC_MODEL,
            timeout=600,
            router_metrics_file=None,
        )

    assert len(captured_payloads) == 2
    for payload in captured_payloads:
        assert payload["options"]["num_thread"] == ROUTER_NUM_THREAD
        assert payload["stream"] is False
    # critic must run on CPU only
    critic_payload = next(
        p for p in captured_payloads if p["model"] == DEFAULT_CRITIC_MODEL
    )
    assert critic_payload["options"]["num_gpu"] == 0


# -------------------------------------------------------------------------
# Phase 3.8c: distill() router-feedback merge loop integration
# -------------------------------------------------------------------------
#
# Strategy: stub out fetch_chunks_by_theme + QdrantClient + _run_asymmetric_debate
# + call_14b so distill() runs through the retry loop without any I/O. The
# stubbed proposer output is a code block with a syntax error (``def
# broken_``) so every attempt fails L1 and the loop runs to max_retries.
# That gives us 3 distinct proposer prompts to inspect for hint injection.


from router.runners import ModelOutput  # noqa: E402
from router.strategies import RouterResult  # noqa: E402


_BROKEN_CODE_BLOCK = "```python\ndef broken_\n```"


def _make_router_result(findings: tuple[str, ...]) -> RouterResult:
    """Build a RouterResult whose chosen_text fails L1 syntax check."""
    base_output = ModelOutput(
        text=_BROKEN_CODE_BLOCK,
        role="proposer",
        model_id="deepseek-r1:14b",
        prompt_eval_count=10,
        eval_count=20,
        eval_duration_ns=1_000_000_000,
        prompt_eval_duration_ns=100_000_000,
        total_duration_ns=1_100_000_000,
        load_duration_ns=0,
    )
    critic_output = ModelOutput(
        text="\n".join(f"- {f}" for f in findings),
        role="critic",
        model_id=DEFAULT_CRITIC_MODEL,
        prompt_eval_count=10,
        eval_count=20,
        eval_duration_ns=900_000_000,
        prompt_eval_duration_ns=100_000_000,
        total_duration_ns=1_000_000_000,
        load_duration_ns=0,
    )
    return RouterResult(
        strategy_name="asymmetric_debate",
        proposer_output=base_output,
        critic_output=critic_output,
        chosen_text=_BROKEN_CODE_BLOCK,
        critic_findings=findings,
        parallel_wall_sec=1.0,
        started_at="2026-06-01T00:00:00+00:00",
        finished_at="2026-06-01T00:00:01+00:00",
    )


@pytest.fixture
def stub_distill_io(monkeypatch):
    """Patch out QdrantClient + chunks + call_14b so distill() is hermetic.

    Returns the captured single-model prompts list (mutated by the test).
    """
    monkeypatch.setattr(corpus2skill, "QdrantClient", lambda **kw: MagicMock())
    monkeypatch.setattr(
        corpus2skill,
        "fetch_chunks_by_theme",
        lambda *a, **k: [{"url": "u", "heading_path": "h", "text": "t"}],
    )
    captured_single: list[str] = []

    def fake_call_14b(prompt: str, **kwargs: object) -> str:
        captured_single.append(prompt)
        return _BROKEN_CODE_BLOCK

    monkeypatch.setattr(corpus2skill, "call_14b", fake_call_14b)
    return captured_single


def test_router_feedback_none_does_not_inject_hint(
    monkeypatch, stub_distill_io, tmp_path
) -> None:
    """feedback='none' で attempts 2+ の call_14b prompt に hint が含まれない。

    Phase 3.8b 完全互換: router は attempt 1 のみ稼働、findings は memo されるが
    proposer prompt overlay は適用されない。
    """
    captured_proposer: list[str] = []

    def fake_router(*, proposer_prompt, **kw):
        captured_proposer.append(proposer_prompt)
        return _make_router_result(("pitfall ALPHA", "pitfall BETA"))

    monkeypatch.setattr(corpus2skill, "_run_asymmetric_debate", fake_router)

    result = corpus2skill.distill(
        "k8s",
        collection="c",
        max_retries=3,
        timeout=10,
        verify=False,
        quality_loop=False,
        router_strategy="asymmetric_debate",
        critic_model=DEFAULT_CRITIC_MODEL,
        router_metrics_file=None,
        router_feedback="none",
    )

    # All 3 attempts hit L1 syntax fail; attempt 1=router, 2-3=call_14b
    assert not result.valid
    assert result.attempts == 3
    assert len(captured_proposer) == 1  # only attempt 1 used router
    assert len(stub_distill_io) == 2  # attempts 2 + 3 used single-model

    # feedback='none' means NO hint header in any call_14b prompt
    for p in stub_distill_io:
        assert "PRIOR INDEPENDENT REVIEWER" not in p
        assert "pitfall ALPHA" not in p
        assert "pitfall BETA" not in p

    # DistillResult fields reflect "feedback disabled"
    assert result.router_feedback_mode is None
    assert result.router_findings_injected_count == 0
    # Phase 3.8b legacy fields still populated from attempt 1 router run
    assert result.router_strategy == "asymmetric_debate"
    assert result.router_critic_findings_count == 2


def test_router_feedback_on_retry_injects_memoized_findings_each_attempt(
    monkeypatch, stub_distill_io, tmp_path
) -> None:
    """feedback='on-retry' で attempt 1 findings が attempts 2/3 prompt に注入。

    router は attempt 1 のみ稼働 (on-retry semantics)、findings はその後の
    すべての retry に持ち回し。インジェクトの中身は不変 (memoized)。
    """
    captured_proposer: list[str] = []
    router_call_count = {"n": 0}

    def fake_router(*, proposer_prompt, **kw):
        captured_proposer.append(proposer_prompt)
        router_call_count["n"] += 1
        return _make_router_result(("avoid mutable default args",
                                     "use pathlib not os.path"))

    monkeypatch.setattr(corpus2skill, "_run_asymmetric_debate", fake_router)

    result = corpus2skill.distill(
        "k8s",
        collection="c",
        max_retries=3,
        timeout=10,
        verify=False,
        quality_loop=False,
        router_strategy="asymmetric_debate",
        critic_model=DEFAULT_CRITIC_MODEL,
        router_metrics_file=None,
        router_feedback="on-retry",
    )

    # Router runs ONCE (attempt 1 only), attempts 2/3 are single-model
    assert router_call_count["n"] == 1
    assert len(captured_proposer) == 1
    assert len(stub_distill_io) == 2

    # Both call_14b prompts (attempts 2, 3) contain the SAME hint (memoized)
    for p in stub_distill_io:
        assert "PRIOR INDEPENDENT REVIEWER" in p
        assert "- avoid mutable default args" in p
        assert "- use pathlib not os.path" in p

    # Attempt 1's proposer prompt has NO hint (attempt 1 has no prior findings)
    assert "PRIOR INDEPENDENT REVIEWER" not in captured_proposer[0]

    # DistillResult fields reflect 2 injection attempts (2 + 3)
    assert result.router_feedback_mode == "on-retry"
    assert result.router_findings_injected_count == 2


def test_router_feedback_every_attempt_refreshes_findings_each_call(
    monkeypatch, stub_distill_io, tmp_path
) -> None:
    """feedback='every-attempt' で router 毎 attempt 稼働 + findings 毎回 replace。

    各 attempt の critic 出力は次 attempt の proposer prompt にのみ inject
    (累積はしない、最新だけ)。本 test では critic を 3 attempt それぞれ別の
    findings tuple を返すよう仕込み、attempt N+1 の prompt が attempt N の
    findings を反映していることを検証。
    """
    captured_proposer: list[str] = []
    findings_sequence = [
        ("alpha_1", "alpha_2"),
        ("beta_1", "beta_2"),
        ("gamma_1", "gamma_2"),
    ]
    call_idx = {"n": 0}

    def fake_router(*, proposer_prompt, **kw):
        captured_proposer.append(proposer_prompt)
        idx = call_idx["n"]
        call_idx["n"] += 1
        # findings from THIS attempt's critic carry to NEXT attempt's proposer
        return _make_router_result(findings_sequence[idx])

    monkeypatch.setattr(corpus2skill, "_run_asymmetric_debate", fake_router)

    result = corpus2skill.distill(
        "k8s",
        collection="c",
        max_retries=3,
        timeout=10,
        verify=False,
        quality_loop=False,
        router_strategy="asymmetric_debate",
        critic_model=DEFAULT_CRITIC_MODEL,
        router_metrics_file=None,
        router_feedback="every-attempt",
    )

    # Router runs on EVERY attempt (3 calls), call_14b unused
    assert call_idx["n"] == 3
    assert len(captured_proposer) == 3
    assert len(stub_distill_io) == 0

    # Attempt 1: no hint (no prior findings)
    assert "PRIOR INDEPENDENT REVIEWER" not in captured_proposer[0]
    # Attempt 2: hint contains attempt 1's findings (alpha_*)
    assert "PRIOR INDEPENDENT REVIEWER" in captured_proposer[1]
    assert "- alpha_1" in captured_proposer[1]
    assert "- alpha_2" in captured_proposer[1]
    # Attempt 2 prompt MUST NOT contain attempt 2's own findings (beta_*)
    assert "- beta_1" not in captured_proposer[1]
    # Attempt 3: hint contains attempt 2's findings (beta_*), NOT alpha or gamma
    assert "PRIOR INDEPENDENT REVIEWER" in captured_proposer[2]
    assert "- beta_1" in captured_proposer[2]
    assert "- beta_2" in captured_proposer[2]
    assert "- alpha_1" not in captured_proposer[2]
    assert "- gamma_1" not in captured_proposer[2]

    # DistillResult: 2 injection events (attempts 2 + 3)
    assert result.router_feedback_mode == "every-attempt"
    assert result.router_findings_injected_count == 2


# -------------------------------------------------------------------------
# Phase 3.8c+: _run_sequential_critic_review — dispatch + record + ordering
# -------------------------------------------------------------------------


def test_run_sequential_critic_review_writes_router_record(tmp_path: Path) -> None:
    """End-to-end: sequential helper builds the right runner shape, writes a
    record marked with execution_mode='sequential', and produces a
    RouterResult whose strategy_name disambiguates from asymmetric_debate."""
    target = tmp_path / "router_runs.jsonl"
    chunks = [
        {"url": "https://example.invalid/a", "heading_path": "Intro", "text": "alpha"},
        {"url": "https://example.invalid/b", "heading_path": "Refs", "text": "beta"},
    ]
    fake_post = _route_by_model({
        "deepseek-r1:14b": _OLLAMA_RESPONSE_PROPOSER,
        DEFAULT_CRITIC_MODEL: _OLLAMA_RESPONSE_CRITIC,
    })

    with patch("router.runners.requests.post", side_effect=fake_post):
        result = _run_sequential_critic_review(
            theme="kubernetes",
            chunks=chunks,
            proposer_prompt="distill this",
            proposer_model="deepseek-r1:14b",
            critic_model=DEFAULT_CRITIC_MODEL,
            timeout=600,
            router_metrics_file=target,
        )

    assert result.strategy_name == "sequential_critic_review"
    assert result.proposer_output.model_id == "deepseek-r1:14b"
    assert result.critic_output.model_id == DEFAULT_CRITIC_MODEL
    assert result.chosen_text.startswith("```python")
    # 3 bullets in _OLLAMA_RESPONSE_CRITIC → all parsed
    assert result.critic_findings == ("pitfall A", "pitfall B", "pitfall C")

    assert target.exists()
    rec = json.loads(target.read_text(encoding="utf-8").strip())
    assert rec["strategy_name"] == "sequential_critic_review"
    assert rec["theme"] == "kubernetes"
    assert rec["critic_findings_count"] == 3
    # Sequential execution is flagged in the options snapshot.
    assert rec["options"]["execution_mode"] == "sequential"
    assert rec["options"]["num_thread"] == ROUTER_NUM_THREAD
    assert rec["options"]["critic_num_gpu"] == 0


def test_run_sequential_critic_review_injects_proposer_output_into_critic_prompt(
    tmp_path: Path,
) -> None:
    """The whole point of Phase 3.8c+: the critic must see the proposer's
    actual code in its prompt. Captures the per-model request bodies and
    verifies that the critic's prompt contains the proposer's response
    body verbatim."""
    chunks = [{"url": "u", "heading_path": "h", "text": "t"}]
    captured: list[dict] = []

    def capturing_post(url: str, *, json: dict, timeout: float) -> MagicMock:  # noqa: A002
        captured.append(json)
        return _make_mock_response(
            _OLLAMA_RESPONSE_PROPOSER
            if json["model"] == "deepseek-r1:14b"
            else _OLLAMA_RESPONSE_CRITIC
        )

    with patch("router.runners.requests.post", side_effect=capturing_post):
        _run_sequential_critic_review(
            theme="x",
            chunks=chunks,
            proposer_prompt="p",
            proposer_model="deepseek-r1:14b",
            critic_model=DEFAULT_CRITIC_MODEL,
            timeout=600,
            router_metrics_file=None,
        )

    # Exactly 2 calls: proposer first, critic second (sequential ordering).
    assert len(captured) == 2
    assert captured[0]["model"] == "deepseek-r1:14b"
    assert captured[1]["model"] == DEFAULT_CRITIC_MODEL
    # The proposer's response code block must appear inside the critic's
    # prompt — this is the Phase 3.8c+ raison d'être.
    proposer_response_text = _OLLAMA_RESPONSE_PROPOSER["response"]
    critic_prompt = captured[1]["prompt"]
    assert "def helper():" in critic_prompt
    assert "BEGIN PROPOSER CODE" in critic_prompt
    assert "END PROPOSER CODE" in critic_prompt
    # Proposer code body (after stripping markdown fences) should appear in critic prompt
    assert proposer_response_text in critic_prompt
    # critic must still receive options with num_gpu=0 (CPU-only).
    assert captured[1]["options"]["num_gpu"] == 0
