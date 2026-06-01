"""Phase 3.8b: ``router._metrics`` unit tests."""

from __future__ import annotations

import json
from pathlib import Path

from router._metrics import (
    ROUTER_SCHEMA_VERSION,
    append_router_record,
    build_router_record,
)
from router.runners import ModelOutput
from router.strategies import RouterResult


def _sample_result() -> RouterResult:
    proposer = ModelOutput(
        text="proposer output",
        role="proposer",
        model_id="deepseek-r1:14b",
        prompt_eval_count=100,
        eval_count=500,
        eval_duration_ns=10_000_000_000,
        prompt_eval_duration_ns=1_000_000_000,
        total_duration_ns=11_000_000_000,
        load_duration_ns=200_000_000,
    )
    critic = ModelOutput(
        text="- a\n- b\n",
        role="critic",
        model_id="gemma2:9b-instruct-q4_K_M",
        prompt_eval_count=80,
        eval_count=200,
        eval_duration_ns=5_000_000_000,
        prompt_eval_duration_ns=500_000_000,
        total_duration_ns=5_500_000_000,
        load_duration_ns=150_000_000,
    )
    return RouterResult(
        strategy_name="asymmetric_debate",
        proposer_output=proposer,
        critic_output=critic,
        chosen_text=proposer.text,
        critic_findings=("a", "b"),
        parallel_wall_sec=11.2349,
        started_at="2026-06-01T18:00:00+00:00",
        finished_at="2026-06-01T18:00:11+00:00",
    )


def test_build_router_record_schema_v1_shape() -> None:
    result = _sample_result()
    rec = build_router_record(
        result=result,
        theme="kubernetes",
        options={"num_thread": 6},
    )

    assert ROUTER_SCHEMA_VERSION == 1
    assert rec["schema_version"] == 1
    assert rec["strategy_name"] == "asymmetric_debate"
    assert rec["theme"] == "kubernetes"
    assert rec["parallel_wall_sec"] == 11.235  # rounded to 3 decimals
    assert rec["started_at"] == "2026-06-01T18:00:00+00:00"
    assert rec["finished_at"] == "2026-06-01T18:00:11+00:00"

    assert isinstance(rec["outputs"], list)
    assert len(rec["outputs"]) == 2

    proposer_rec = rec["outputs"][0]
    assert proposer_rec["role"] == "proposer"
    assert proposer_rec["model_id"] == "deepseek-r1:14b"
    assert proposer_rec["eval_count"] == 500
    assert proposer_rec["text_len_chars"] == len("proposer output")

    critic_rec = rec["outputs"][1]
    assert critic_rec["role"] == "critic"
    assert critic_rec["eval_count"] == 200
    assert critic_rec["text_len_chars"] == len("- a\n- b\n")

    assert rec["critic_findings_count"] == 2
    assert rec["critic_findings"] == ["a", "b"]
    assert rec["options"] == {"num_thread": 6}
    assert "meta" in rec
    assert "git_commit" in rec["meta"]
    assert "host" in rec["meta"]


def test_build_router_record_handles_none_options() -> None:
    rec = build_router_record(
        result=_sample_result(),
        theme="x",
        options=None,
    )
    assert rec["options"] == {}


def test_build_router_record_record_is_jsonable() -> None:
    rec = build_router_record(
        result=_sample_result(),
        theme="kubernetes",
        options={"num_thread": 6},
    )
    # round-trip through JSON to detect non-serializable values
    s = json.dumps(rec)
    assert json.loads(s) == rec


def test_append_router_record_writes_one_jsonl_line(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "router_runs.jsonl"
    rec1 = {"schema_version": 1, "theme": "a", "x": 1}
    rec2 = {"schema_version": 1, "theme": "b", "x": 2}

    append_router_record(target, rec1)
    append_router_record(target, rec2)

    lines = target.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0]) == rec1
    assert json.loads(lines[1]) == rec2


def test_append_router_record_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c" / "out.jsonl"
    assert not target.parent.exists()
    append_router_record(target, {"hello": "world"})
    assert target.exists()
    assert json.loads(target.read_text()) == {"hello": "world"}


def test_append_router_record_handles_japanese_text(tmp_path: Path) -> None:
    """ensure_ascii=False で日本語が壊れずに書かれる"""
    target = tmp_path / "out.jsonl"
    append_router_record(target, {"text": "壊れない日本語"})
    line = target.read_text(encoding="utf-8").strip()
    assert "壊れない日本語" in line
    assert json.loads(line) == {"text": "壊れない日本語"}
