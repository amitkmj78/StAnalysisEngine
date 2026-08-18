from unittest.mock import patch

from services.web_search.client import SearchResponse, SearchResult, _score, search
from services.web_search.extract import extract_content


# ---------------------------------------------------------------------------
# extract_content — pure HTML parsing, no network (httpx.Client is mocked)
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, text: str, content_type: str = "text/html"):
        self.text = text
        self.headers = {"content-type": content_type}

    def raise_for_status(self):
        pass


class _FakeClient:
    def __init__(self, response: _FakeResponse):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url):
        return self._response


ARTICLE_HTML = """
<html>
<head><script>var x = 1;</script></head>
<body>
  <nav>Home | About | Contact</nav>
  <header>Site Header</header>
  <article>
    <p>This is the real article content that a reader actually came for, with enough length to pass the paragraph floor.</p>
    <p>A second paragraph continuing the real story, also long enough to be kept by the extractor.</p>
  </article>
  <footer>Copyright 2026</footer>
</body>
</html>
"""

NO_ARTICLE_TAG_HTML = """
<html><body>
  <div class="nav">Short nav link</div>
  <p>Short.</p>
  <p>This is a long enough paragraph to be picked up by the fallback paragraph-density heuristic when there is no article or main tag present on the page.</p>
</body></html>
"""


def test_extract_content_prefers_article_tag():
    with patch("services.web_search.extract.httpx.Client", return_value=_FakeClient(_FakeResponse(ARTICLE_HTML))):
        text = extract_content("https://example.com/article")
    assert text is not None
    assert "real article content" in text
    assert "Site Header" not in text
    assert "Copyright" not in text


def test_extract_content_falls_back_to_paragraphs_without_article_tag():
    with patch("services.web_search.extract.httpx.Client", return_value=_FakeClient(_FakeResponse(NO_ARTICLE_TAG_HTML))):
        text = extract_content("https://example.com/no-article-tag")
    assert text is not None
    assert "fallback paragraph-density heuristic" in text
    assert "Short nav link" not in text  # too short to pass MIN_PARAGRAPH_CHARS


def test_extract_content_returns_none_for_non_html():
    with patch(
        "services.web_search.extract.httpx.Client",
        return_value=_FakeClient(_FakeResponse("{}", content_type="application/json")),
    ):
        assert extract_content("https://example.com/data.json") is None


def test_extract_content_returns_none_on_request_failure():
    class _RaisingClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            raise ConnectionError("boom")

    with patch("services.web_search.extract.httpx.Client", return_value=_RaisingClient()):
        assert extract_content("https://example.com/unreachable") is None


def test_extract_content_truncates_to_max_chars():
    long_html = "<html><body><article>" + ("<p>" + "word " * 2000 + "</p>") + "</article></body></html>"
    with patch("services.web_search.extract.httpx.Client", return_value=_FakeClient(_FakeResponse(long_html))):
        text = extract_content("https://example.com/long", max_chars=100)
    assert text is not None
    assert len(text) == 100


# ---------------------------------------------------------------------------
# _score — pure keyword-overlap scoring
# ---------------------------------------------------------------------------

def test_score_is_higher_for_title_match_than_content_only_match():
    title_match = _score("apple earnings", "Apple Earnings Report", "some unrelated body text")
    content_only_match = _score("apple earnings", "Unrelated Title", "apple earnings mentioned here")
    assert title_match > content_only_match


def test_score_is_zero_for_no_overlap():
    assert _score("apple earnings", "Banana Recipe", "how to bake bread") == 0.0


def test_score_is_capped_at_one():
    score = _score("ab", "ab ab ab", "ab ab ab")
    assert score <= 1.0


def test_score_handles_empty_query_terms():
    # every query term <=2 chars gets filtered out -> no terms to score on
    assert _score("a is to", "anything", "anything") == 0.0


# ---------------------------------------------------------------------------
# search() — end-to-end wiring, with backend.py's network call mocked
# ---------------------------------------------------------------------------

def test_search_returns_empty_response_when_backend_finds_nothing():
    with patch("services.web_search.client.ddg_search", return_value=[]):
        resp = search("no results for this query")
    assert isinstance(resp, SearchResponse)
    assert resp.results == []
    assert resp.response_time_ms >= 0


def test_search_ranks_results_by_score_and_respects_include_raw_content():
    hits = [
        {"title": "Irrelevant", "href": "https://example.com/a", "body": "nothing to do with the query here"},
        {"title": "apple earnings beat", "href": "https://example.com/b", "body": "apple earnings details"},
    ]
    with (
        patch("services.web_search.client.ddg_search", return_value=hits),
        patch("services.web_search.client.extract_content", return_value=None),
    ):
        resp = search("apple earnings", max_results=2, include_raw_content=False)

    assert len(resp.results) == 2
    assert all(isinstance(r, SearchResult) for r in resp.results)
    # Higher-relevance result ("apple earnings beat") should be ranked first.
    assert resp.results[0].url == "https://example.com/b"
    assert resp.results[0].score >= resp.results[1].score
    # include_raw_content=False -> raw_content stripped from every result.
    assert all(r.raw_content is None for r in resp.results)


def test_search_falls_back_to_snippet_when_extraction_fails():
    hits = [{"title": "T", "href": "https://example.com/a", "body": "fallback snippet text"}]
    with (
        patch("services.web_search.client.ddg_search", return_value=hits),
        patch("services.web_search.client.extract_content", return_value=None),
    ):
        resp = search("query", max_results=1)
    assert resp.results[0].content == "fallback snippet text"
