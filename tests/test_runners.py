"""Phase 3.8b: ``router.runners`` unit tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from router.runners import ModelOutput, OllamaRunner


SAMPLE_RESPONSE: dict[str, object] = {
    "response": "hello world",
    "prompt_eval_count": 12,
    "eval_count": 34,
    "eval_duration": 500_000_000,
    "prompt_eval_duration": 100_000_000,
    "total_duration": 700_000_000,
    "load_duration": 50_000_000,
}


def test_model_output_from_ollama_response_extracts_all_fields() -> None:
    out = ModelOutput.from_ollama_response(
        SAMPLE_RESPONSE, role="proposer", model_id="dummy:latest"
    )
    assert out.text == "hello world"
    assert out.role == "proposer"
    assert out.model_id == "dummy:latest"
    assert out.prompt_eval_count == 12
    assert out.eval_count == 34
    assert out.eval_duration_ns == 500_000_000
    assert out.prompt_eval_duration_ns == 100_000_000
    assert out.total_duration_ns == 700_000_000
    assert out.load_duration_ns == 50_000_000


def test_model_output_from_ollama_response_handles_missing_keys() -> None:
    out = ModelOutput.from_ollama_response({}, role="critic", model_id="x:y")
    assert out.text == ""
    assert out.eval_count == 0
    assert out.eval_duration_ns == 0
    assert out.total_duration_ns == 0


def test_model_output_from_ollama_response_handles_none_payload() -> None:
    out = ModelOutput.from_ollama_response(None, role="critic", model_id="x:y")
    assert out.text == ""
    assert out.eval_count == 0


def test_model_output_is_frozen() -> None:
    out = ModelOutput.from_ollama_response(SAMPLE_RESPONSE, role="r", model_id="m")
    with pytest.raises(Exception):
        out.text = "mutated"  # type: ignore[misc]


def _mock_response(payload: dict[str, object]) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def test_ollama_runner_generate_posts_to_correct_endpoint() -> None:
    runner = OllamaRunner(
        role="proposer",
        model_id="test-model:1b",
        base_url="http://example.invalid:11434",
        default_options={"num_thread": 6, "temperature": 0.1},
    )

    with patch(
        "router.runners.requests.post", return_value=_mock_response(SAMPLE_RESPONSE)
    ) as mock_post:
        out = runner.generate("hi", options={"num_predict": 10})

    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "http://example.invalid:11434/api/generate"
    payload = kwargs["json"]
    assert payload["model"] == "test-model:1b"
    assert payload["prompt"] == "hi"
    assert payload["stream"] is False
    # default options merged with per-call options
    assert payload["options"]["num_thread"] == 6
    assert payload["options"]["temperature"] == 0.1
    assert payload["options"]["num_predict"] == 10
    # caller-passed timeout matches runner default
    assert kwargs["timeout"] == 600.0

    assert isinstance(out, ModelOutput)
    assert out.text == "hello world"
    assert out.role == "proposer"
    assert out.model_id == "test-model:1b"


def test_ollama_runner_per_call_options_override_defaults() -> None:
    runner = OllamaRunner(
        role="critic",
        model_id="m:x",
        default_options={"num_thread": 6, "temperature": 0.1},
    )

    with patch(
        "router.runners.requests.post", return_value=_mock_response(SAMPLE_RESPONSE)
    ) as mock_post:
        runner.generate("hi", options={"temperature": 0.7})

    payload = mock_post.call_args.kwargs["json"]
    assert payload["options"]["temperature"] == 0.7  # overridden
    assert payload["options"]["num_thread"] == 6  # default preserved


def test_ollama_runner_no_options_sends_only_defaults() -> None:
    runner = OllamaRunner(
        role="proposer",
        model_id="m:x",
        default_options={"num_thread": 6},
    )

    with patch(
        "router.runners.requests.post", return_value=_mock_response(SAMPLE_RESPONSE)
    ) as mock_post:
        runner.generate("hi")

    payload = mock_post.call_args.kwargs["json"]
    assert payload["options"] == {"num_thread": 6}


def test_ollama_runner_does_not_mutate_default_options() -> None:
    """Per-call merging must not contaminate the runner's default_options."""
    defaults = {"num_thread": 6}
    runner = OllamaRunner(
        role="proposer",
        model_id="m:x",
        default_options=defaults,
    )

    with patch(
        "router.runners.requests.post", return_value=_mock_response(SAMPLE_RESPONSE)
    ):
        runner.generate("hi", options={"num_predict": 99, "num_thread": 8})

    assert defaults == {"num_thread": 6}, "default_options leaked per-call mutation"
    assert runner.default_options == {"num_thread": 6}


def test_ollama_runner_raise_for_status_propagates() -> None:
    resp = MagicMock()
    resp.raise_for_status.side_effect = RuntimeError("HTTP 500")

    runner = OllamaRunner(role="proposer", model_id="m:x")
    with patch("router.runners.requests.post", return_value=resp):
        with pytest.raises(RuntimeError, match="HTTP 500"):
            runner.generate("hi")
