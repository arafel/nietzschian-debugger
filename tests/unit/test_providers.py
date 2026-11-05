"""Tests for multi-model provider selection.

Network calls are mocked so selection, request formatting, and streaming can be
checked without API keys or a running local model.
"""

from __future__ import annotations

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from nietzschian_debugger.llm import providers
from nietzschian_debugger.llm import client as llm_client


@pytest.fixture(autouse=True)
def _reset_provider():
    # Never let a set provider leak into other tests (which use the Claude path).
    yield
    providers.configure(None)


def test_default_is_anthropic() -> None:
    providers.configure(None)
    assert providers.active() is None
    providers.configure("claude")
    assert providers.active() is None


def test_known_providers_parse() -> None:
    providers.configure("deepseek:deepseek-chat")
    assert providers.active() == "deepseek"
    assert providers.model_name() == "deepseek-chat"

    providers.configure("ollama")  # model omitted -> default
    assert providers.active() == "ollama"
    assert providers.model_name() == "llama3"

    providers.configure("openai:gpt-4o-mini")
    assert providers.active() == "openai"
    assert providers.model_name() == "gpt-4o-mini"


def test_unknown_name_stays_on_anthropic() -> None:
    providers.configure("notaprovider:x")
    assert providers.active() is None


def test_estimate_tokens_is_nonnegative() -> None:
    n = providers.estimate_tokens("system prompt", [{"role": "user", "content": "hello there"}])
    assert isinstance(n, int) and n >= 0


@pytest.fixture
def openai_client(monkeypatch):
    providers.configure("openai:gpt-5.1-codex")
    client = MagicMock()
    monkeypatch.setattr(providers, "_client_obj", client)
    return client


MESSAGES = [
    {"role": "user", "content": "My test fails"},
    {"role": "assistant", "content": "What did you observe?"},
    {"role": "user", "content": "A timeout"},
]


def test_openai_complete_preserves_instructions_and_history(openai_client):
    openai_client.responses.create.return_value = SimpleNamespace(
        status="completed", output_text="What timed out?"
    )
    assert llm_client.get_conversation_model() == "gpt-5.1-codex"
    assert llm_client.get_model_for_file("example.py") == "gpt-5.1-codex"
    assert llm_client.call_llm("Ask one question", MESSAGES, "") == "What timed out?"
    openai_client.responses.create.assert_called_once_with(
        model="gpt-5.1-codex", instructions="Ask one question", input=MESSAGES,
        max_output_tokens=4096, store=False,
    )
    openai_client.chat.completions.create.assert_not_called()


def test_openai_stream_emits_only_text_and_closes(openai_client):
    stream = openai_client.responses.stream.return_value.__enter__.return_value
    stream.__iter__.return_value = iter([
        SimpleNamespace(type="response.created"),
        SimpleNamespace(type="response.reasoning_summary_text.delta", delta="hidden"),
        SimpleNamespace(type="response.output_text.delta", delta="What "),
        SimpleNamespace(type="response.output_text.delta", delta="timed out?"),
        SimpleNamespace(type="response.completed"),
    ])
    stream.get_final_response.return_value = SimpleNamespace(
        status="completed", output_text="What timed out?"
    )
    callbacks = MagicMock()
    assert llm_client.stream_question("Ask", MESSAGES, "gpt-4o-mini", callbacks) == "What timed out?"
    assert [call.args[0] for call in callbacks.on_text.call_args_list] == ["What ", "timed out?"]
    callbacks.on_complete.assert_called_once_with("What timed out?")
    callbacks.on_error.assert_not_called()
    assert openai_client.responses.stream.call_args.kwargs["model"] == "gpt-4o-mini"
    openai_client.responses.stream.return_value.__exit__.assert_called_once()


@pytest.mark.parametrize("event", [
    SimpleNamespace(type="error", message="Request failed"),
    SimpleNamespace(type="response.failed"),
])
def test_openai_stream_failure_does_not_complete(openai_client, event):
    stream = openai_client.responses.stream.return_value.__enter__.return_value
    stream.__iter__.return_value = iter([event])
    stream.get_final_response.return_value = SimpleNamespace(status="failed", output_text="")
    callbacks = MagicMock()
    with pytest.raises(providers.ProviderError):
        providers.stream("Ask", MESSAGES, "", callbacks)
    callbacks.on_error.assert_called_once()
    callbacks.on_complete.assert_not_called()
    openai_client.responses.stream.return_value.__exit__.assert_called_once()


@pytest.mark.parametrize("status,text", [("incomplete", "Partial"), ("completed", "")])
def test_openai_rejects_unusable_responses(openai_client, status, text):
    openai_client.responses.create.return_value = SimpleNamespace(status=status, output_text=text)
    with pytest.raises(providers.ProviderError):
        providers.complete("Ask", MESSAGES, "")


def test_openai_connection_error_calls_error_callback(openai_client):
    error = RuntimeError("Connection failed")
    openai_client.responses.stream.side_effect = error
    callbacks = MagicMock()
    with pytest.raises(RuntimeError, match="Connection failed"):
        providers.stream("Ask", MESSAGES, "", callbacks)
    callbacks.on_error.assert_called_once_with(error)
    callbacks.on_complete.assert_not_called()


def test_compatible_provider_keeps_chat_completions(monkeypatch):
    providers.configure("ollama:llama3.1")
    client = MagicMock()
    monkeypatch.setattr(providers, "_client_obj", client)
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Question?"))]
    )
    assert providers.complete("Ask", MESSAGES, "") == "Question?"
    client.chat.completions.create.assert_called_once_with(
        model="llama3.1", messages=[{"role": "system", "content": "Ask"}, *MESSAGES],
        max_tokens=1024,
    )
    client.responses.create.assert_not_called()


@pytest.mark.parametrize("streaming", [False, True])
def test_real_sdk_with_simulated_http(monkeypatch, streaming):
    """Exercise SDK serialization and SSE parsing without a key or network."""
    import json
    openai = pytest.importorskip("openai")
    import httpx

    response = {
        "id": "resp_test", "object": "response", "created_at": 0,
        "status": "completed", "model": "gpt-4o-mini",
        "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
        "output": [{"id": "msg_test", "type": "message", "role": "assistant",
                    "status": "completed", "content": [
                        {"type": "output_text", "text": "What changed?", "annotations": []}]}],
    }

    def handle(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/responses"
        assert body["store"] is False
        assert body["instructions"] == "Ask"
        assert body["input"] == MESSAGES
        if not streaming:
            return httpx.Response(200, json=response)
        events = [
            {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
            {"type": "response.output_item.added", "output_index": 0,
             "item": {**response["output"][0], "content": []}},
            {"type": "response.content_part.added", "item_id": "msg_test", "output_index": 0,
             "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}},
            {"type": "response.output_text.delta", "item_id": "msg_test", "output_index": 0,
             "content_index": 0, "delta": "What changed?"},
            {"type": "response.completed", "response": response},
        ]
        payload = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=payload)

    providers.configure("openai")
    with openai.OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        monkeypatch.setattr(providers, "_client_obj", client)
        if streaming:
            callbacks = MagicMock()
            assert providers.stream("Ask", MESSAGES, "", callbacks) == "What changed?"
            callbacks.on_text.assert_called_once_with("What changed?")
            callbacks.on_complete.assert_called_once_with("What changed?")
            callbacks.on_error.assert_not_called()
        else:
            assert providers.complete("Ask", MESSAGES, "") == "What changed?"
