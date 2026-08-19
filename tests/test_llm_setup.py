import pytest

from services.llm_setup import invoke_with_fallback


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, response_text: str = "ok"):
        self.response_text = response_text
        self.calls = 0

    def invoke(self, prompt: str):
        self.calls += 1
        return _FakeMessage(self.response_text)


class _RaisingLLM:
    def __init__(self, message: str = "provider unavailable"):
        self.message = message
        self.calls = 0

    def invoke(self, prompt: str):
        self.calls += 1
        raise RuntimeError(self.message)


def test_invoke_with_fallback_uses_first_healthy_provider():
    llm = _FakeLLM("first provider answered")
    content, index = invoke_with_fallback([llm], "prompt")
    assert content == "first provider answered"
    assert index == 0


def test_invoke_with_fallback_skips_failing_provider_and_tries_next():
    failing = _RaisingLLM("insufficient_quota")
    healthy = _FakeLLM("second provider answered")
    content, index = invoke_with_fallback([failing, healthy], "prompt")
    assert content == "second provider answered"
    assert index == 1
    assert failing.calls == 1
    assert healthy.calls == 1


def test_invoke_with_fallback_skips_none_entries():
    healthy = _FakeLLM("answered")
    content, index = invoke_with_fallback([None, healthy], "prompt")
    assert content == "answered"
    assert index == 1


def test_invoke_with_fallback_raises_last_exception_when_all_fail():
    first = _RaisingLLM("first failure")
    second = _RaisingLLM("second failure")
    with pytest.raises(RuntimeError, match="second failure"):
        invoke_with_fallback([first, second], "prompt")


def test_invoke_with_fallback_raises_runtime_error_when_list_empty():
    with pytest.raises(RuntimeError):
        invoke_with_fallback([], "prompt")


def test_invoke_with_fallback_raises_runtime_error_when_all_none():
    with pytest.raises(RuntimeError):
        invoke_with_fallback([None, None], "prompt")


def test_invoke_with_fallback_does_not_call_providers_after_success():
    healthy = _FakeLLM("first works")
    unused = _FakeLLM("should not be called")
    invoke_with_fallback([healthy, unused], "prompt")
    assert unused.calls == 0
