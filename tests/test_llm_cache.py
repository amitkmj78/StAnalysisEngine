from web.backend.llm_cache import label_for_llm, ordered_llms, resolve_llm

OPENAI, GROQ, CLAUDE, OLLAMA = "openai-instance", "groq-instance", "claude-instance", "ollama-instance"
LABELS = ["OpenAI · GPT-4.0", "Groq · gpt-oss-20b", "Claude · Sonnet 5", "Local · llama3 (Ollama)"]


def test_resolve_llm_matches_by_label_prefix():
    assert resolve_llm("OpenAI · GPT-4.0", OPENAI, GROQ, CLAUDE, OLLAMA) == OPENAI
    assert resolve_llm("Groq · gpt-oss-20b", OPENAI, GROQ, CLAUDE, OLLAMA) == GROQ
    assert resolve_llm("Claude · Sonnet 5", OPENAI, GROQ, CLAUDE, OLLAMA) == CLAUDE
    assert resolve_llm("Local · llama3 (Ollama)", OPENAI, GROQ, CLAUDE, OLLAMA) == OLLAMA


def test_ordered_llms_puts_preferred_provider_first():
    ordered = ordered_llms("Groq · gpt-oss-20b", OPENAI, GROQ, CLAUDE, OLLAMA, LABELS)
    assert ordered[0] == GROQ
    assert set(ordered) == {OPENAI, GROQ, CLAUDE, OLLAMA}


def test_ordered_llms_falls_back_to_labels_zero_when_no_preference_given():
    ordered = ordered_llms(None, OPENAI, GROQ, CLAUDE, OLLAMA, LABELS)
    assert ordered[0] == OPENAI  # LABELS[0]


def test_ordered_llms_excludes_unconfigured_none_providers():
    ordered = ordered_llms(None, OPENAI, None, None, None, ["OpenAI · GPT-4.0"])
    assert ordered == [OPENAI]


def test_ordered_llms_does_not_duplicate_preferred_provider():
    ordered = ordered_llms("OpenAI · GPT-4.0", OPENAI, GROQ, CLAUDE, OLLAMA, LABELS)
    assert ordered.count(OPENAI) == 1
    assert len(ordered) == 4


def test_label_for_llm_returns_matching_label():
    assert label_for_llm(GROQ, OPENAI, GROQ, CLAUDE, OLLAMA, LABELS) == "Groq · gpt-oss-20b"


def test_label_for_llm_returns_none_when_instance_not_found():
    assert label_for_llm("unknown-instance", OPENAI, GROQ, CLAUDE, OLLAMA, LABELS) is None
