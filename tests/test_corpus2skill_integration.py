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
    DEFAULT_ROUTER_METRICS_FILE,
    DEFAULT_ROUTER_STRATEGY,
    METRICS_SCHEMA_VERSION,
    MODEL_REGISTRY,
    ROUTER_NUM_THREAD,
    DistillResult,
    _build_metrics_record,
    _run_asymmetric_debate,
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
    )


def test_metrics_schema_version_bumped_to_three() -> None:
    assert METRICS_SCHEMA_VERSION == 3


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
    assert rec["meta"]["schema_version"] == 3


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
    assert "{none,asymmetric_debate}" in text
    assert "--critic-model" in text
    assert "--router-metrics-file" in text
    assert "--no-router-metrics" in text


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
