from unittest.mock import patch

import pytest

from services import market_news_service as mod
from services.market_news_service import get_hot_market_news


@pytest.fixture(autouse=True)
def _clear_news_cache():
    # get_hot_market_news keeps its own last-known-good result in module
    # state (not a plain @ttl_cache -- see its docstring for why) --
    # without resetting it, every test after the first would just see
    # the first test's cached result regardless of its own mocks.
    mod._cached_result = None
    mod._cached_at = 0.0
    yield
    mod._cached_result = None
    mod._cached_at = 0.0


def _yahoo_entry(item_id, title, url, published_at="2026-09-04T12:00:00Z", provider="Reuters"):
    return {
        "id": item_id,
        "content": {
            "id": item_id,
            "title": title,
            "clickThroughUrl": {"url": url},
            "provider": {"displayName": provider},
            "pubDate": published_at,
        },
    }


def test_uses_yahoo_news_when_available():
    entries = [_yahoo_entry("1", "Fed holds rates steady", "https://example.com/1")]
    with patch("services.market_news_service.fetch_with_backoff", return_value=entries), patch(
        "services.market_news_service.ddg_search"
    ) as mock_ddg:
        result = get_hot_market_news()
    assert result["source"] == "yahoo"
    assert len(result["items"]) == 1
    assert result["items"][0]["title"] == "Fed holds rates steady"
    assert result["items"][0]["source"] == "Reuters"
    mock_ddg.assert_not_called()


def test_dedupes_the_same_story_across_multiple_tickers():
    entries = [_yahoo_entry("dup-1", "Market rallies on jobs data", "https://example.com/dup")]
    with patch("services.market_news_service.fetch_with_backoff", return_value=entries):
        result = get_hot_market_news()
    # Same id returned for every MARKET_TICKERS lookup -- should collapse to one.
    assert len(result["items"]) == 1


def test_falls_back_to_duckduckgo_when_yahoo_returns_nothing():
    with patch("services.market_news_service.fetch_with_backoff", return_value=[]), patch(
        "services.market_news_service.ddg_search",
        return_value=[{"title": "Stocks slide", "href": "https://example.com/ddg", "body": ""}],
    ):
        result = get_hot_market_news()
    assert result["source"] == "duckduckgo"
    assert result["items"] == [
        {"title": "Stocks slide", "url": "https://example.com/ddg", "source": "DuckDuckGo", "published_at": None}
    ]


def test_falls_back_to_duckduckgo_when_yahoo_errors():
    with patch("services.market_news_service.fetch_with_backoff", side_effect=Exception("rate limited")), patch(
        "services.market_news_service.ddg_search",
        return_value=[{"title": "Stocks slide", "href": "https://example.com/ddg", "body": ""}],
    ):
        result = get_hot_market_news()
    assert result["source"] == "duckduckgo"
    assert len(result["items"]) == 1


def test_a_real_result_is_reused_within_the_success_ttl():
    entries = [_yahoo_entry("1", "Fed holds rates steady", "https://example.com/1")]
    with patch("services.market_news_service.fetch_with_backoff", return_value=entries) as mock_fetch:
        get_hot_market_news()
        get_hot_market_news()
    # Second call reused the cached result -- one fetch, not two.
    assert mock_fetch.call_count == len(mod.MARKET_TICKERS)


def test_an_empty_result_is_retried_almost_immediately_not_cached_for_the_full_ttl():
    """Both sources failing must not poison the ticker for the full
    5-minute success TTL -- a retry a few seconds later should hit the
    network again, not replay the same empty result."""
    with patch("services.market_news_service.fetch_with_backoff", return_value=[]), patch(
        "services.market_news_service.ddg_search", return_value=[]
    ):
        first = get_hot_market_news()
    assert first["items"] == []

    # Simulate EMPTY_RETRY_SECONDS having elapsed (far short of the
    # 5-minute success TTL) -- a retry now should hit the network again
    # rather than replay the cached empty result.
    mod._cached_at -= mod.EMPTY_RETRY_SECONDS + 1

    entries = [_yahoo_entry("1", "Market rallies", "https://example.com/1")]
    with patch("services.market_news_service.fetch_with_backoff", return_value=entries):
        second = get_hot_market_news()
    assert len(second["items"]) == 1
