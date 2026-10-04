from services.rag_retrieval import CHUNK_CHARS, Passage, build_context, chunks, rank_passages


def test_chunks_keep_short_paragraphs_and_split_long_ones_on_sentences():
    short = "First paragraph about tariffs.\n\nSecond paragraph about margins."
    assert chunks(short) == ["First paragraph about tariffs.", "Second paragraph about margins."]

    long_para = " ".join(["Revenue grew strongly in services and wearables."] * 60)
    pieces = chunks(long_para)
    assert len(pieces) > 1
    assert all(len(p) <= CHUNK_CHARS + 60 for p in pieces)


def test_a_passage_with_the_question_terms_ranks_first():
    passages = [
        Passage("A", "The company sells phones and laptops in many regions."),
        Passage("B", "Tariffs on imported semiconductors could raise component costs and squeeze margins."),
        Passage("C", "Dividend paid to shareholders of record in August."),
    ]
    top = rank_passages("What do the tariffs mean for margins?", passages, k=2)
    assert top[0].source == "B"


def test_a_question_that_names_a_ticker_prefers_that_tickers_passages():
    same_text = "Revenue growth was driven by services and new product launches."
    passages = [
        Passage("MSFT filing", same_text, "MSFT"),
        Passage("AAPL filing", same_text, "AAPL"),
    ]
    top = rank_passages("What drove AAPL revenue growth?", passages, k=2)
    assert top[0].source == "AAPL filing"


def test_no_matching_terms_returns_nothing():
    passages = [Passage("A", "Dividend paid to shareholders.")]
    assert rank_passages("zebra quantum lattice", passages) == []


def test_build_context_numbers_passages_and_respects_the_budget():
    passages = [Passage("S1", "x" * 100), Passage("S2", "y" * 100), Passage("S3", "z" * 100)]
    context, used = build_context(passages, budget=230)
    assert context.startswith("[1] S1")
    assert [p.source for p in used] == ["S1", "S2"]
    assert "[3]" not in context
