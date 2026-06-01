"""Phase 3.8b: ``router.strategies`` unit tests.

Parallelism is verified by giving each fake runner a 300 ms sleep and
asserting wall ≈ max(300 ms) rather than 600 ms. A sequential
implementation would take ~600 ms; a broken thread-join would let one
runner finish before the other starts, which we also catch via the
``ordering`` cross-check.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pytest

from router.runners import ModelOutput
from router.strategies import (
    AsymmetricDebateStrategy,
    RouterResult,
    build_critic_prompt,
    parse_critic_findings,
)


@dataclass
class FakeRunner:
    role: str
    model_id: str
    sleep_sec: float = 0.0
    return_text: str = ""
    started_at: float | None = None
    finished_at: float | None = None

    def generate(
        self, prompt: str, *, options: dict[str, Any] | None = None
    ) -> ModelOutput:
        self.started_at = time.monotonic()
        if self.sleep_sec > 0:
            time.sleep(self.sleep_sec)
        self.finished_at = time.monotonic()
        return ModelOutput(
            text=self.return_text,
            role=self.role,
            model_id=self.model_id,
            prompt_eval_count=10,
            eval_count=100,
            eval_duration_ns=int(self.sleep_sec * 1e9),
            prompt_eval_duration_ns=0,
            total_duration_ns=int(self.sleep_sec * 1e9),
            load_duration_ns=0,
        )


# -------------------------------------------------------------------------
# parse_critic_findings + build_critic_prompt
# -------------------------------------------------------------------------


def test_parse_critic_findings_extracts_bullet_lines() -> None:
    text = (
        "- first issue\n"
        "- second issue\n"
        "* third bullet (asterisk)\n"
        "not a bullet line\n"
        "- fourth issue\n"
    )
    findings = parse_critic_findings(text)
    assert findings == (
        "first issue",
        "second issue",
        "third bullet (asterisk)",
        "fourth issue",
    )


def test_parse_critic_findings_handles_empty_text() -> None:
    assert parse_critic_findings("") == ()


def test_parse_critic_findings_handles_preamble() -> None:
    text = (
        "Sure, here are the pitfalls:\n"
        "\n"
        "- gotcha A\n"
        "- gotcha B\n"
    )
    assert parse_critic_findings(text) == ("gotcha A", "gotcha B")


def test_build_critic_prompt_substitutes_fields() -> None:
    p = build_critic_prompt(
        theme="asyncio", n_chunks=3, joined_chunks="--- chunk body ---"
    )
    assert "asyncio" in p
    assert "3 Markdown" in p
    assert "--- chunk body ---" in p
    assert "Output the pitfall list now" in p


# -------------------------------------------------------------------------
# AsymmetricDebateStrategy.route
# -------------------------------------------------------------------------


def test_asymmetric_debate_runs_runners_in_parallel() -> None:
    proposer = FakeRunner(
        role="proposer",
        model_id="big:14b",
        sleep_sec=0.3,
        return_text="```python\npass\n```",
    )
    critic = FakeRunner(
        role="critic",
        model_id="small:9b",
        sleep_sec=0.3,
        return_text="- gotcha 1\n- gotcha 2\n",
    )

    strategy = AsymmetricDebateStrategy()
    t0 = time.monotonic()
    result = strategy.route(
        proposer_prompt="distill ...",
        critic_prompt="critique ...",
        proposer=proposer,
        critic=critic,
    )
    elapsed = time.monotonic() - t0

    # parallel: wall ≈ max(0.3, 0.3) ≈ 0.3, NOT sequential 0.6
    assert elapsed < 0.55, f"expected parallel execution, got wall={elapsed:.3f}s"
    assert result.parallel_wall_sec < 0.55
    # both runners started within a small window of each other (true parallelism)
    assert proposer.started_at is not None
    assert critic.started_at is not None
    assert abs(proposer.started_at - critic.started_at) < 0.05


def test_asymmetric_debate_returns_router_result_with_both_outputs() -> None:
    proposer = FakeRunner(
        role="proposer", model_id="p:1", sleep_sec=0.01, return_text="proposer text"
    )
    critic = FakeRunner(
        role="critic", model_id="c:1", sleep_sec=0.01,
        return_text="- finding A\n- finding B\n",
    )

    result = AsymmetricDebateStrategy().route(
        proposer_prompt="x",
        critic_prompt="y",
        proposer=proposer,
        critic=critic,
    )

    assert isinstance(result, RouterResult)
    assert result.strategy_name == "asymmetric_debate"
    assert result.proposer_output.role == "proposer"
    assert result.proposer_output.model_id == "p:1"
    assert result.critic_output.role == "critic"
    assert result.critic_output.model_id == "c:1"
    assert result.chosen_text == "proposer text"
    assert result.critic_findings == ("finding A", "finding B")
    assert result.parallel_wall_sec > 0
    assert result.started_at and result.finished_at
    assert result.started_at <= result.finished_at


def test_asymmetric_debate_passes_distinct_prompts_to_each_runner() -> None:
    captured: dict[str, str] = {}

    @dataclass
    class CapturingRunner:
        role: str
        model_id: str

        def generate(
            self, prompt: str, *, options: dict[str, Any] | None = None
        ) -> ModelOutput:
            captured[self.role] = prompt
            return ModelOutput(
                text="x",
                role=self.role,
                model_id=self.model_id,
                prompt_eval_count=0,
                eval_count=0,
                eval_duration_ns=0,
                prompt_eval_duration_ns=0,
                total_duration_ns=0,
                load_duration_ns=0,
            )

    AsymmetricDebateStrategy().route(
        proposer_prompt="PROPOSER PROMPT",
        critic_prompt="CRITIC PROMPT",
        proposer=CapturingRunner(role="proposer", model_id="p:1"),
        critic=CapturingRunner(role="critic", model_id="c:1"),
    )

    assert captured["proposer"] == "PROPOSER PROMPT"
    assert captured["critic"] == "CRITIC PROMPT"


def test_asymmetric_debate_propagates_proposer_error() -> None:
    @dataclass
    class FailingRunner:
        role: str
        model_id: str

        def generate(
            self, prompt: str, *, options: dict[str, Any] | None = None
        ) -> ModelOutput:
            raise RuntimeError("boom")

    proposer = FailingRunner(role="proposer", model_id="p:1")
    critic = FakeRunner(
        role="critic", model_id="c:1", sleep_sec=0.01, return_text="- x"
    )

    with pytest.raises(RuntimeError, match="proposer failed"):
        AsymmetricDebateStrategy().route(
            proposer_prompt="x",
            critic_prompt="y",
            proposer=proposer,  # type: ignore[arg-type]
            critic=critic,
        )


def test_asymmetric_debate_propagates_critic_error() -> None:
    @dataclass
    class FailingRunner:
        role: str
        model_id: str

        def generate(
            self, prompt: str, *, options: dict[str, Any] | None = None
        ) -> ModelOutput:
            raise RuntimeError("critic boom")

    proposer = FakeRunner(
        role="proposer", model_id="p:1", sleep_sec=0.01, return_text="ok"
    )
    critic = FailingRunner(role="critic", model_id="c:1")

    with pytest.raises(RuntimeError, match="critic failed"):
        AsymmetricDebateStrategy().route(
            proposer_prompt="x",
            critic_prompt="y",
            proposer=proposer,
            critic=critic,  # type: ignore[arg-type]
        )
