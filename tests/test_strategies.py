"""Phase 3.8b: ``router.strategies`` unit tests.

Parallelism is verified by giving each fake runner a 300 ms sleep and
asserting wall ≈ max(300 ms) rather than 600 ms. A sequential
implementation would take ~600 ms; a broken thread-join would let one
runner finish before the other starts, which we also catch via the
``ordering`` cross-check.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from router.runners import ModelOutput
from router.strategies import (
    DEFAULT_CRITIC_HINT_MAX_LINES,
    AsymmetricDebateStrategy,
    RouterResult,
    SequentialCriticReviewStrategy,
    build_critic_prompt,
    build_critic_review_prompt,
    format_critic_hint,
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
# format_critic_hint (Phase 3.8c)
# -------------------------------------------------------------------------


def test_format_critic_hint_empty_returns_empty() -> None:
    assert format_critic_hint(()) == ""


def test_format_critic_hint_single_finding() -> None:
    out = format_critic_hint(("avoid mutating shared state",))
    assert "PRIOR INDEPENDENT REVIEWER" in out
    assert "- avoid mutating shared state" in out
    # Surrounding blank lines so concatenation with prior prompt is clean.
    assert out.startswith("\n\n")
    assert out.endswith("\n")


def test_format_critic_hint_multi_findings_preserves_order() -> None:
    findings = (
        "do not import asyncio.run inside coroutines",
        "watch for off-by-one in deque slicing",
        "subprocess.run requires text=True for str output",
    )
    out = format_critic_hint(findings)
    idx0 = out.index(findings[0])
    idx1 = out.index(findings[1])
    idx2 = out.index(findings[2])
    assert idx0 < idx1 < idx2
    # Each finding renders as a "- " bullet line.
    for f in findings:
        assert f"- {f}" in out


def test_format_critic_hint_clips_to_max_lines() -> None:
    findings = tuple(f"finding {i}" for i in range(15))
    out = format_critic_hint(findings, max_lines=5)
    # First 5 present, 6th onward absent.
    for i in range(5):
        assert f"- finding {i}" in out
    for i in range(5, 15):
        assert f"- finding {i}" not in out
    # Default clip matches the module-level constant.
    default_out = format_critic_hint(findings)
    expected_default_lines = min(len(findings), DEFAULT_CRITIC_HINT_MAX_LINES)
    assert default_out.count("\n- ") == expected_default_lines


def test_format_critic_hint_preserves_special_chars() -> None:
    findings = (
        "do not use `eval()` on user input",
        "{template} placeholders must be escaped",
        "watch for embedded\nnewlines in returned strings",
    )
    out = format_critic_hint(findings)
    # Backticks/braces/newlines pass through verbatim, no format-substitution
    # collapse and no escaping.
    assert "`eval()`" in out
    assert "{template} placeholders" in out
    assert "embedded\nnewlines" in out


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


# -------------------------------------------------------------------------
# SequentialCriticReviewStrategy (Phase 3.8c+)
# -------------------------------------------------------------------------


def test_critic_review_prompt_template_substitutes_all_fields() -> None:
    p = build_critic_review_prompt(
        theme="asyncio",
        n_chunks=2,
        joined_chunks="--- chunk body ---",
        proposer_code="def helper():\n    pass\n",
    )
    assert "asyncio" in p
    assert "2 Markdown" in p
    assert "--- chunk body ---" in p
    assert "def helper():" in p
    # Anchor markers separate the two payloads so the LLM cannot confuse
    # docs vs. produced code.
    assert "--- BEGIN PROPOSER CODE ---" in p
    assert "--- END PROPOSER CODE ---" in p
    assert "Output the issue list now" in p


@dataclass
class RecordingRunner:
    """FakeRunner variant that captures *every* prompt it receives.

    The sequential strategy mutates the critic prompt between proposer
    and critic, so the test needs to inspect what the critic actually
    saw (not just the template).
    """

    role: str
    model_id: str
    sleep_sec: float = 0.0
    return_text: str = ""
    received_prompts: list[str] = field(default_factory=list)
    started_at: float | None = None
    finished_at: float | None = None

    def generate(
        self, prompt: str, *, options: dict[str, Any] | None = None
    ) -> ModelOutput:
        self.received_prompts.append(prompt)
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


def test_sequential_critic_review_runs_proposer_before_critic() -> None:
    """Ordering invariant: critic must observe proposer.finished_at <=
    its own started_at — i.e., critic cannot start before proposer
    finishes (the whole point of going sequential)."""
    proposer = RecordingRunner(
        role="proposer", model_id="p:1", sleep_sec=0.2,
        return_text="def f(): pass",
    )
    critic = RecordingRunner(
        role="critic", model_id="c:1", sleep_sec=0.2,
        return_text="- issue 1",
    )
    SequentialCriticReviewStrategy().route(
        proposer_prompt="prop",
        critic_prompt_template="review:\n{proposer_code}",
        proposer=proposer,  # type: ignore[arg-type]
        critic=critic,  # type: ignore[arg-type]
    )
    assert proposer.started_at is not None
    assert proposer.finished_at is not None
    assert critic.started_at is not None
    # critic.started_at must be on or after proposer.finished_at
    assert critic.started_at >= proposer.finished_at


def test_sequential_critic_review_wall_equals_sum_not_max() -> None:
    """Sequential semantics: wall ≈ sum(proposer, critic), not max.

    Each runner sleeps 0.2 s; total wall must be at least 0.4 s. A
    parallel implementation would clock in around 0.2 s, so a comfortable
    margin (0.35 s) catches accidental parallelism.
    """
    proposer = RecordingRunner(
        role="proposer", model_id="p:1", sleep_sec=0.2, return_text="x"
    )
    critic = RecordingRunner(
        role="critic", model_id="c:1", sleep_sec=0.2, return_text="- x"
    )
    t0 = time.monotonic()
    result = SequentialCriticReviewStrategy().route(
        proposer_prompt="prop",
        critic_prompt_template="review:\n{proposer_code}",
        proposer=proposer,  # type: ignore[arg-type]
        critic=critic,  # type: ignore[arg-type]
    )
    wall = time.monotonic() - t0
    assert wall >= 0.35, f"wall {wall:.3f}s suggests parallel execution"
    # The strategy's own reported wall should match observed wall within 50 ms.
    assert abs(result.parallel_wall_sec - wall) < 0.05


def test_sequential_critic_review_injects_proposer_output_into_critic_prompt() -> None:
    """The critic must see the proposer's actual code in its prompt."""
    proposer_code = "def reusable_helper():\n    return 42"
    proposer = RecordingRunner(
        role="proposer", model_id="p:1", return_text=proposer_code
    )
    critic = RecordingRunner(
        role="critic", model_id="c:1", return_text="- looks fine"
    )
    template = (
        "Review the following code:\n--- CODE ---\n"
        "{proposer_code}\n--- END ---"
    )
    SequentialCriticReviewStrategy().route(
        proposer_prompt="distill this",
        critic_prompt_template=template,
        proposer=proposer,  # type: ignore[arg-type]
        critic=critic,  # type: ignore[arg-type]
    )
    # Proposer saw only its own prompt, no critic-template leakage.
    assert proposer.received_prompts == ["distill this"]
    # Critic saw the substituted template with proposer's actual output.
    assert len(critic.received_prompts) == 1
    critic_prompt_seen = critic.received_prompts[0]
    assert "def reusable_helper():" in critic_prompt_seen
    assert "return 42" in critic_prompt_seen
    assert "Review the following code" in critic_prompt_seen


def test_sequential_critic_review_returns_router_result_with_findings() -> None:
    """Structural check on RouterResult, including parsed findings."""
    proposer = RecordingRunner(
        role="proposer", model_id="deepseek-r1:14b",
        return_text="def f(): pass",
    )
    critic_text = (
        "- hallucinated `os.walkdir` (should be `os.walk`)\n"
        "- off-by-one in deque slicing on line 12\n"
        "- missing import for `Path`\n"
    )
    critic = RecordingRunner(
        role="critic", model_id="gemma2:9b-instruct-q4_K_M",
        return_text=critic_text,
    )
    result = SequentialCriticReviewStrategy().route(
        proposer_prompt="p",
        critic_prompt_template="t:{proposer_code}",
        proposer=proposer,  # type: ignore[arg-type]
        critic=critic,  # type: ignore[arg-type]
    )
    assert isinstance(result, RouterResult)
    assert result.strategy_name == "sequential_critic_review"
    assert result.chosen_text == "def f(): pass"
    assert result.proposer_output.model_id == "deepseek-r1:14b"
    assert result.critic_output.model_id == "gemma2:9b-instruct-q4_K_M"
    assert result.critic_findings == (
        "hallucinated `os.walkdir` (should be `os.walk`)",
        "off-by-one in deque slicing on line 12",
        "missing import for `Path`",
    )


def test_sequential_critic_review_skips_critic_on_proposer_failure() -> None:
    """If the proposer raises, the critic is NOT called and the error
    propagates. Sequential semantics — there is nothing to review."""

    @dataclass
    class FailingRunner:
        role: str
        model_id: str

        def generate(
            self, prompt: str, *, options: dict[str, Any] | None = None
        ) -> ModelOutput:
            raise RuntimeError("proposer boom")

    critic_called = {"n": 0}

    @dataclass
    class WatchingCritic:
        role: str = "critic"
        model_id: str = "c:1"

        def generate(
            self, prompt: str, *, options: dict[str, Any] | None = None
        ) -> ModelOutput:
            critic_called["n"] += 1
            return ModelOutput(
                text="never reached", role=self.role, model_id=self.model_id,
                prompt_eval_count=0, eval_count=0,
                eval_duration_ns=0, prompt_eval_duration_ns=0,
                total_duration_ns=0, load_duration_ns=0,
            )

    proposer = FailingRunner(role="proposer", model_id="p:1")
    critic = WatchingCritic()

    with pytest.raises(RuntimeError, match="proposer boom"):
        SequentialCriticReviewStrategy().route(
            proposer_prompt="x",
            critic_prompt_template="y:{proposer_code}",
            proposer=proposer,  # type: ignore[arg-type]
            critic=critic,  # type: ignore[arg-type]
        )
    assert critic_called["n"] == 0, "critic must not run after proposer failure"
