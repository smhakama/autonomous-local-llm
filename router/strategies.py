"""Phase 3.8b/3.8c+: router strategies.

PoC #1 = ``AsymmetricDebateStrategy`` (Phase 3.8b) — parallel proposer
+ critic that both receive the *same chunks* (no proposer-output
dependency on critic side) and run concurrently via threading.
Wall ≈ ``max(proposer_total, critic_total)`` because Phase 3.8a measured
``wall_vs_total_max ≈ 1.000`` under the NT6 configuration.

The merge step is intentionally minimal in this PoC: ``chosen_text`` is
the proposer output; the critic output is parsed into a list of findings
and recorded in metrics. Phase 3.8c then feeds those findings back into
proposer retry prompts via ``format_critic_hint``.

PoC #2 = ``SequentialCriticReviewStrategy`` (Phase 3.8c+) — proposer
runs *first*, then a critic re-runs on (chunks + proposer's code) so
findings actually critique the produced module rather than the source
docs in the abstract. Wall ≈ ``proposer_total + critic_total`` (sequential
by construction; the parallel benefit is traded for per-attempt fresh
feedback). Used when Phase 3.8c smoke shows ``every-attempt`` mode with
chunks-only critic produces near-identical findings each iteration.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from .runners import ModelOutput, ModelRunner


CRITIC_PROMPT_TEMPLATE = """You are a code-skill critic. Below are {n_chunks} Markdown chunks of documentation about "{theme}". A separate LLM (the proposer) will read the same chunks and write a single Python module of reusable helper functions for future agents.

Your task: enumerate the common pitfalls, hallucinated imports, off-by-one APIs, deprecated symbols, and code traps a Python distiller might fall into when writing that module from these chunks. Do NOT write Python code yourself. Do NOT critique a specific module (you have not seen one); critique the chunks.

Output format: ONE issue per line, prefixed with "- ", short and specific. 5 to 15 lines. No preamble. No closing summary.

--- BEGIN CHUNKS ---
{joined_chunks}
--- END CHUNKS ---

Output the pitfall list now:"""


_CRITIC_LINE_PATTERN = re.compile(r"^\s*[-*]\s+(.+?)\s*$", re.MULTILINE)


_CRITIC_HINT_HEADER = (
    "PRIOR INDEPENDENT REVIEWER (a separate model that saw the same chunks) "
    "FLAGGED THESE PITFALLS PRE-EMPTIVELY. Avoid them in your output:"
)
DEFAULT_CRITIC_HINT_MAX_LINES = 10


def parse_critic_findings(text: str) -> tuple[str, ...]:
    """Extract ``- ...`` / ``* ...`` bullet lines from critic output.

    Lines that do not start with a bullet marker are ignored — this
    discards stray preamble that LLMs sometimes emit despite the prompt
    asking for bullets only.
    """
    return tuple(m.group(1) for m in _CRITIC_LINE_PATTERN.finditer(text))


def format_critic_hint(
    findings: tuple[str, ...],
    *,
    max_lines: int = DEFAULT_CRITIC_HINT_MAX_LINES,
) -> str:
    """Format critic findings as a prompt-injection hint block.

    Empty findings -> empty string so the caller can unconditionally
    concat without branching. Findings beyond ``max_lines`` are clipped
    to keep the augmented prompt bounded (the critic prompt asks for
    5-15 lines; clipping at 10 keeps the median while protecting
    against runaway critic outputs).

    Special characters in findings (backticks, braces, newlines) are
    preserved verbatim — the hint is plain text, not a format template.
    """
    if not findings:
        return ""
    clipped = findings[:max_lines]
    bullets = "\n".join(f"- {f}" for f in clipped)
    return f"\n\n{_CRITIC_HINT_HEADER}\n{bullets}\n"


def build_critic_prompt(*, theme: str, n_chunks: int, joined_chunks: str) -> str:
    """Format ``CRITIC_PROMPT_TEMPLATE`` with the given chunks payload."""
    return CRITIC_PROMPT_TEMPLATE.format(
        theme=theme, n_chunks=n_chunks, joined_chunks=joined_chunks
    )


@dataclass(frozen=True)
class RouterResult:
    """Immutable outcome of one ``RouterStrategy.route()`` call."""

    strategy_name: str
    proposer_output: ModelOutput
    critic_output: ModelOutput
    chosen_text: str
    critic_findings: tuple[str, ...]
    parallel_wall_sec: float
    started_at: str
    finished_at: str


class RouterStrategy(Protocol):
    name: str

    def route(
        self,
        *,
        proposer_prompt: str,
        critic_prompt: str,
        proposer: ModelRunner,
        critic: ModelRunner,
        options: dict[str, Any] | None = None,
    ) -> RouterResult:
        ...


@dataclass(frozen=True)
class AsymmetricDebateStrategy:
    """Run proposer + critic in parallel on the same chunks.

    Each runner gets its own prompt (proposer_prompt, critic_prompt) but
    they execute concurrently. Errors from either thread are re-raised on
    the caller's thread after both joins so we never leak a half-finished
    run as success.
    """

    name: str = "asymmetric_debate"

    def route(
        self,
        *,
        proposer_prompt: str,
        critic_prompt: str,
        proposer: ModelRunner,
        critic: ModelRunner,
        options: dict[str, Any] | None = None,
    ) -> RouterResult:
        outputs: dict[str, ModelOutput | None] = {"proposer": None, "critic": None}
        errors: dict[str, BaseException | None] = {"proposer": None, "critic": None}

        def _call(key: str, runner: ModelRunner, prompt: str) -> None:
            try:
                outputs[key] = runner.generate(prompt, options=options)
            except BaseException as exc:  # noqa: BLE001 — re-raised below
                errors[key] = exc

        started_at = datetime.now(timezone.utc).isoformat()
        t_start = time.monotonic()
        t_proposer = threading.Thread(
            target=_call,
            args=("proposer", proposer, proposer_prompt),
            name="router-proposer",
        )
        t_critic = threading.Thread(
            target=_call,
            args=("critic", critic, critic_prompt),
            name="router-critic",
        )
        t_proposer.start()
        t_critic.start()
        t_proposer.join()
        t_critic.join()
        wall = time.monotonic() - t_start
        finished_at = datetime.now(timezone.utc).isoformat()

        if errors["proposer"] is not None:
            raise RuntimeError("proposer failed") from errors["proposer"]
        if errors["critic"] is not None:
            raise RuntimeError("critic failed") from errors["critic"]

        proposer_out = outputs["proposer"]
        critic_out = outputs["critic"]
        assert proposer_out is not None
        assert critic_out is not None

        return RouterResult(
            strategy_name=self.name,
            proposer_output=proposer_out,
            critic_output=critic_out,
            chosen_text=proposer_out.text,
            critic_findings=parse_critic_findings(critic_out.text),
            parallel_wall_sec=wall,
            started_at=started_at,
            finished_at=finished_at,
        )


# -------------------------------------------------------------------------
# Phase 3.8c+: SequentialCriticReviewStrategy
# -------------------------------------------------------------------------
#
# Phase 3.8c smoke proved that a chunks-only critic with a stable prompt
# converges to near-identical findings each iteration (top 2 bullets verbatim
# match across 3 modes). The fix is to let the critic actually see what the
# proposer produced and review *that code* against the chunks. Trade-off:
# lose Phase 3.8b's parallel-wall benefit (now wall ≈ proposer + critic
# instead of max), gain genuinely fresh per-attempt feedback.


CRITIC_REVIEW_PROMPT_TEMPLATE = """You are a code-skill critic reviewing a Python module that another LLM (the proposer) just wrote about "{theme}". You have the same {n_chunks} Markdown chunks the proposer saw, AND the proposer's actual code output. Review the code against the chunks.

Your task: identify concrete issues in the proposer's code — hallucinated APIs that don't appear in the chunks, off-by-one logic, mis-implemented invariants, mishandled edge cases, deprecated symbols, or claims the chunks do not support. Cite specific function names or line content from the code. Do NOT rewrite the code. Do NOT critique the chunks in the abstract; critique THIS module.

Output format: ONE issue per line, prefixed with "- ", short and specific. 5 to 15 lines. No preamble. No closing summary.

--- BEGIN CHUNKS ---
{joined_chunks}
--- END CHUNKS ---

--- BEGIN PROPOSER CODE ---
{proposer_code}
--- END PROPOSER CODE ---

Output the issue list now:"""


def build_critic_review_prompt(
    *,
    theme: str,
    n_chunks: int,
    joined_chunks: str,
    proposer_code: str,
) -> str:
    """Format ``CRITIC_REVIEW_PROMPT_TEMPLATE`` with the proposer's code.

    The proposer code is injected verbatim so the critic can refer to
    specific function names / lines. No truncation is applied here — if
    the proposer emits a 50KB module, the caller is responsible for
    deciding whether to clip first (out of scope for the strategy).
    """
    return CRITIC_REVIEW_PROMPT_TEMPLATE.format(
        theme=theme,
        n_chunks=n_chunks,
        joined_chunks=joined_chunks,
        proposer_code=proposer_code,
    )


@dataclass(frozen=True)
class SequentialCriticReviewStrategy:
    """Run proposer first, then critic on (chunks + proposer's code).

    Sequential by construction: critic input depends on proposer output,
    so no parallelism is possible. The ``parallel_wall_sec`` field on
    ``RouterResult`` records the *total sequential wall* here, not a true
    parallel wall — readers should disambiguate via ``strategy_name``.

    On proposer failure the critic is NOT invoked: a failed proposer
    leaves nothing meaningful to review, and we want fast-fail semantics
    so the outer retry loop can move on.
    """

    name: str = "sequential_critic_review"

    def route(
        self,
        *,
        proposer_prompt: str,
        critic_prompt_template: str,
        proposer: ModelRunner,
        critic: ModelRunner,
        options: dict[str, Any] | None = None,
    ) -> RouterResult:
        started_at = datetime.now(timezone.utc).isoformat()
        t_start = time.monotonic()
        proposer_out = proposer.generate(proposer_prompt, options=options)
        critic_prompt = critic_prompt_template.format(
            proposer_code=proposer_out.text
        )
        critic_out = critic.generate(critic_prompt, options=options)
        wall = time.monotonic() - t_start
        finished_at = datetime.now(timezone.utc).isoformat()

        return RouterResult(
            strategy_name=self.name,
            proposer_output=proposer_out,
            critic_output=critic_out,
            chosen_text=proposer_out.text,
            critic_findings=parse_critic_findings(critic_out.text),
            parallel_wall_sec=wall,
            started_at=started_at,
            finished_at=finished_at,
        )
