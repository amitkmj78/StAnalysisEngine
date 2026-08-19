from unittest.mock import MagicMock, patch

from services.web_search import backend
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
    with patch("services.web_search.client.discover", return_value=([], False)):
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
        patch("services.web_search.client.discover", return_value=(hits, False)),
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
        patch("services.web_search.client.discover", return_value=(hits, False)),
        patch("services.web_search.client.extract_content", return_value=None),
    ):
        resp = search("query", max_results=1)
    assert resp.results[0].content == "fallback snippet text"


def test_search_uses_hit_body_directly_when_content_already_extracted():
    hits = [{"title": "T", "href": "https://example.com/a", "body": "full crawlsearch article text"}]
    with (
        patch("services.web_search.client.discover", return_value=(hits, True)),
        patch("services.web_search.client.extract_content") as mock_extract,
    ):
        resp = search("query", max_results=1, include_raw_content=True)
    mock_extract.assert_not_called()
    assert resp.results[0].content == "full crawlsearch article text"
    assert resp.results[0].raw_content == "full crawlsearch article text"


# ---------------------------------------------------------------------------
# backend.discover() — CrawlSearch first, then Brave, then DuckDuckGo
# ---------------------------------------------------------------------------

def test_discover_uses_crawlsearch_first_when_configured():
    with (
        patch.object(backend, "CRAWLSEARCH_API_URL", "http://localhost:8100"),
        patch.object(backend, "BRAVE_SEARCH_API_KEY", "fake-key"),
        patch.object(backend, "crawlsearch_search", return_value=[{"title": "T", "href": "u", "body": "full text"}]) as mock_cs,
        patch.object(backend, "brave_search") as mock_brave,
        patch.object(backend, "ddg_search") as mock_ddg,
    ):
        hits, already_extracted = backend.discover("query", 5)
    mock_cs.assert_called_once_with("query", 5)
    mock_brave.assert_not_called()
    mock_ddg.assert_not_called()
    assert hits == [{"title": "T", "href": "u", "body": "full text"}]
    assert already_extracted is True


def test_discover_falls_through_to_brave_when_crawlsearch_returns_nothing():
    with (
        patch.object(backend, "CRAWLSEARCH_API_URL", "http://localhost:8100"),
        patch.object(backend, "BRAVE_SEARCH_API_KEY", "fake-key"),
        patch.object(backend, "crawlsearch_search", return_value=[]),
        patch.object(backend, "brave_search", return_value=[{"title": "T", "href": "u", "body": "b"}]) as mock_brave,
    ):
        hits, already_extracted = backend.discover("query", 5)
    mock_brave.assert_called_once_with("query", 5)
    assert already_extracted is False


def test_discover_uses_brave_when_api_key_configured():
    with (
        patch.object(backend, "CRAWLSEARCH_API_URL", None),
        patch.object(backend, "BRAVE_SEARCH_API_KEY", "fake-key"),
        patch.object(backend, "brave_search", return_value=[{"title": "T", "href": "u", "body": "b"}]) as mock_brave,
        patch.object(backend, "ddg_search") as mock_ddg,
    ):
        hits, already_extracted = backend.discover("query", 5)
    mock_brave.assert_called_once_with("query", 5)
    mock_ddg.assert_not_called()
    assert hits == [{"title": "T", "href": "u", "body": "b"}]
    assert already_extracted is False


def test_discover_falls_back_to_ddg_when_no_brave_key():
    with (
        patch.object(backend, "CRAWLSEARCH_API_URL", None),
        patch.object(backend, "BRAVE_SEARCH_API_KEY", None),
        patch.object(backend, "brave_search") as mock_brave,
        patch.object(backend, "ddg_search", return_value=[]) as mock_ddg,
    ):
        backend.discover("query", 5)
    mock_ddg.assert_called_once_with("query", 5)
    mock_brave.assert_not_called()


def test_crawlsearch_search_maps_response_fields():
    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {
        "results": [{"title": "Apple News", "url": "https://example.com/a", "content": "Full article text"}]
    }
    with (
        patch.object(backend, "CRAWLSEARCH_API_URL", "http://localhost:8100"),
        patch("services.web_search.backend.httpx.post", return_value=fake_response),
    ):
        hits = backend.crawlsearch_search("apple", 5)
    assert hits == [{"title": "Apple News", "href": "https://example.com/a", "body": "Full article text"}]


def test_crawlsearch_search_returns_empty_list_on_failure():
    with patch("services.web_search.backend.httpx.post", side_effect=Exception("boom")):
        hits = backend.crawlsearch_search("apple", 5)
    assert hits == []


def test_brave_search_maps_response_fields():
    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {
        "web": {"results": [{"title": "Apple News", "url": "https://example.com/a", "description": "Some snippet"}]}
    }
    with patch("services.web_search.backend.httpx.get", return_value=fake_response):
        hits = backend.brave_search("apple", 5)
    assert hits == [{"title": "Apple News", "href": "https://example.com/a", "body": "Some snippet"}]


def test_brave_search_returns_empty_list_on_failure():
    with patch("services.web_search.backend.httpx.get", side_effect=Exception("boom")):
        hits = backend.brave_search("apple", 5)
    assert hits == []


def test_brave_search_skips_results_missing_url():
    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {"web": {"results": [{"title": "No URL", "description": "x"}]}}
    with patch("services.web_search.backend.httpx.get", return_value=fake_response):
        hits = backend.brave_search("apple", 5)
    assert hits == []
