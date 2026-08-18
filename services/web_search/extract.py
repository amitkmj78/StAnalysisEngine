"""
Real content extraction — fetches a search result's URL and pulls out
clean, readable article text. This is the actual "built from scratch"
half of services/web_search: ddgs only ever gives a one-line snippet
(backend.py's `body` field), the same as any search engine's results
page. Getting real article content is the part Tavily and similar
"AI search" products charge for; this does it ourselves with
BeautifulSoup and a simple, honest heuristic — not a full readability
algorithm, but good enough for feeding an LLM prompt.
"""

import logging
import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

MAX_CONTENT_CHARS = 4000
REQUEST_TIMEOUT_SECONDS = 5.0
STRIP_TAGS = ["script", "style", "nav", "header", "footer", "aside", "form", "iframe", "noscript"]
MIN_PARAGRAPH_CHARS = 40
USER_AGENT = "Mozilla/5.0 (compatible; StAnalysisEngineBot/1.0; +internal research tool)"


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_content(url: str, max_chars: int = MAX_CONTENT_CHARS) -> Optional[str]:
    """
    Returns cleaned article text, or None on any failure (unreachable
    URL, non-HTML content, empty/unparseable page) — a caller must fall
    back to the raw search-snippet in that case (handled in client.py),
    never propagate the failure into the whole search response.
    """
    try:
        with httpx.Client(
            timeout=REQUEST_TIMEOUT_SECONDS,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
    except Exception as e:
        logger.info("Web search: failed to fetch %s: %s", url, e)
        return None

    if "html" not in resp.headers.get("content-type", ""):
        return None

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        logger.info("Web search: failed to parse %s: %s", url, e)
        return None

    for tag_name in STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    main = soup.find("article") or soup.find("main")
    if main is not None:
        text = main.get_text(separator=" ")
    else:
        # No semantic article/main tag — fall back to the page's <p> tags.
        # A real article body is almost always the dense cluster of
        # paragraph text; nav links, ads, and footers rarely form long
        # <p> blocks, so a length floor is a cheap, honest filter.
        paragraphs = [p.get_text(separator=" ") for p in soup.find_all("p")]
        text = " ".join(p for p in paragraphs if len(p.strip()) > MIN_PARAGRAPH_CHARS)

    text = _clean_text(text)
    return text[:max_chars] if text else None
