"""Phase 3.8b: ``ModelRunner`` Protocol + Ollama-backed concrete runner.

A ``ModelRunner`` is the single seam through which the router talks to an
LLM backend. Keep it narrow on purpose: ``generate(prompt, options)`` and a
couple of identifying attributes (``role``, ``model_id``). Swapping the
backend (Ollama → vLLM → llama.cpp → a remote API) means writing one new
class that implements this Protocol; nothing in ``router.strategies`` or
``corpus2skill.py`` has to change.

``ModelOutput`` mirrors the relevant fields of Ollama's ``/api/generate``
response so downstream code (metrics, retry loop) does not have to know
the raw HTTP shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import requests


@dataclass(frozen=True)
class ModelOutput:
    """Immutable result of a single ``ModelRunner.generate()`` call.

    ``role`` and ``model_id`` are duplicated from the runner so the
    downstream record (router_runs.jsonl) is self-describing without
    needing to join against runner state.
    """

    text: str
    role: str
    model_id: str
    prompt_eval_count: int
    eval_count: int
    eval_duration_ns: int
    prompt_eval_duration_ns: int
    total_duration_ns: int
    load_duration_ns: int

    @staticmethod
    def from_ollama_response(
        resp: dict[str, Any] | None,
        *,
        role: str,
        model_id: str,
    ) -> "ModelOutput":
        """Construct from an Ollama ``/api/generate`` JSON payload.

        Missing keys default to 0 / empty string — Ollama can omit count
        fields when ``eval_count == 0`` (early EOS, see Phase 3.7e-1).
        """
        resp = resp or {}
        return ModelOutput(
            text=str(resp.get("response", "")),
            role=role,
            model_id=model_id,
            prompt_eval_count=int(resp.get("prompt_eval_count", 0) or 0),
            eval_count=int(resp.get("eval_count", 0) or 0),
            eval_duration_ns=int(resp.get("eval_duration", 0) or 0),
            prompt_eval_duration_ns=int(resp.get("prompt_eval_duration", 0) or 0),
            total_duration_ns=int(resp.get("total_duration", 0) or 0),
            load_duration_ns=int(resp.get("load_duration", 0) or 0),
        )


class ModelRunner(Protocol):
    """Single seam between router strategies and an LLM backend."""

    role: str
    model_id: str

    def generate(
        self, prompt: str, *, options: dict[str, Any] | None = None
    ) -> ModelOutput:
        ...


@dataclass(frozen=True)
class OllamaRunner:
    """Concrete ``ModelRunner`` backed by Ollama ``/api/generate``.

    ``default_options`` is merged with per-call ``options`` (per-call wins
    on key conflict). The intended use is to pin runner-level knobs like
    ``num_thread=6`` (Phase 3.8a NT6 verdict) and ``num_gpu=0`` (gemma
    CPU-only) at construction, then vary ``num_predict`` / ``temperature``
    per call.
    """

    role: str
    model_id: str
    base_url: str = "http://127.0.0.1:11434"
    timeout_sec: float = 600.0
    default_options: dict[str, Any] = field(default_factory=dict)

    def generate(
        self, prompt: str, *, options: dict[str, Any] | None = None
    ) -> ModelOutput:
        merged: dict[str, Any] = dict(self.default_options)
        if options:
            merged.update(options)
        payload: dict[str, Any] = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": merged,
        }
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout_sec,
        )
        resp.raise_for_status()
        return ModelOutput.from_ollama_response(
            resp.json(), role=self.role, model_id=self.model_id
        )
