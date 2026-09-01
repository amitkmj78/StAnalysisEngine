from unittest.mock import patch

from services.quant_signal_narrative_service import build_quant_signal_narrative


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, response_text: str):
        self.response_text = response_text

    def invoke(self, prompt: str):
        return _FakeMessage(self.response_text)


class _RaisingLLM:
    def invoke(self, prompt: str):
        raise RuntimeError("provider unavailable")


def test_build_quant_signal_narrative_splits_technical_and_plain_english():
    llm = _FakeLLM(
        "TECHNICAL: RSI is oversold at 32.7, MACD remains bearish below its signal line.\n"
        "PLAIN ENGLISH: The stock has been sold off harder than usual and may be due for a bounce, "
        "though short-term momentum is still pointed down."
    )
    with patch("services.quant_signal_narrative_service._current_indicators_text", return_value="fake indicators"):
        result = build_quant_signal_narrative([llm], "CASY", "BUY", 42.78, 1075.97, 753.58)
    assert result["technical"].startswith("RSI is oversold")
    assert result["plain_english"].startswith("The stock has been sold off")


def test_build_quant_signal_narrative_joins_multiline_sections():
    llm = _FakeLLM(
        "TECHNICAL: RSI is oversold at 32.7.\nMACD remains bearish.\n"
        "PLAIN ENGLISH: Sold off hard.\nMay bounce soon."
    )
    with patch("services.quant_signal_narrative_service._current_indicators_text", return_value="fake indicators"):
        result = build_quant_signal_narrative([llm], "CASY", "BUY", 42.78, 1075.97, 753.58)
    assert result["technical"] == "RSI is oversold at 32.7. MACD remains bearish."
    assert result["plain_english"] == "Sold off hard. May bounce soon."


def test_build_quant_signal_narrative_falls_back_when_format_not_followed():
    llm = _FakeLLM("Just a plain response with no labeled sections at all.")
    with patch("services.quant_signal_narrative_service._current_indicators_text", return_value="fake indicators"):
        result = build_quant_signal_narrative([llm], "CASY", "BUY", 42.78, 1075.97, 753.58)
    assert result["technical"] == "Just a plain response with no labeled sections at all."
    assert result["plain_english"] is None


def test_build_quant_signal_narrative_returns_none_when_all_providers_fail():
    with patch("services.quant_signal_narrative_service._current_indicators_text", return_value="fake indicators"):
        result = build_quant_signal_narrative([_RaisingLLM()], "CASY", "BUY", 42.78, 1075.97, 753.58)
    assert result is None
