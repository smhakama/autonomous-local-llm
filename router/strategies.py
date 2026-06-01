"""Phase 3.8b: router strategies.

PoC #1 = ``AsymmetricDebateStrategy`` — parallel proposer + critic that
both receive the *same chunks* (no proposer-output dependency on critic
side) and run concurrently via threading. Wall ≈ ``max(proposer_total,
critic_total)`` because Phase 3.8a measured ``wall_vs_total_max ≈ 1.000``
under the NT6 configuration.

The merge step is intentionally minimal in this PoC: ``chosen_text`` is
the proposer output; the critic output is parsed into a list of findings
and recorded in metrics so a later phase (3.8c) can decide whether to
feed findings back as a retry hint. Keeping the merge dumb here lets us
measure critic *signal* in isolation before adding feedback complexity.
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
