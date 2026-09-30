import hashlib
import hmac
import json
from unittest.mock import MagicMock, patch

import httpx

from services.webhook_service import generate_webhook_secret, send_webhook


def _response(status_code=200):
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.raise_for_status.return_value = None
    return response


def test_generate_webhook_secret_is_long_and_unique():
    a, b = generate_webhook_secret(), generate_webhook_secret()
    assert len(a) >= 32
    assert a != b


@patch("services.webhook_service.httpx.post")
def test_send_webhook_signs_the_exact_request_body(mock_post):
    mock_post.return_value = _response(200)
    payload = {"ticker": "AAPL", "alert_type": "cost_drop", "values": {"pct_change": -12.3}, "link": "https://x/stock/AAPL"}

    result = send_webhook("https://example.com/hook", "my-secret", payload)

    assert result is True
    _, kwargs = mock_post.call_args
    sent_body = kwargs["content"]
    expected_signature = hmac.new(b"my-secret", sent_body, hashlib.sha256).hexdigest()
    assert kwargs["headers"]["X-Signature"] == expected_signature
    # A receiver must be able to reconstruct the exact same signature
    # from the payload alone -- confirms the signed bytes really are the
    # JSON-serialized payload, not some other representation.
    assert json.loads(sent_body) == payload


@patch("services.webhook_service.httpx.post")
def test_send_webhook_returns_false_on_non_2xx(mock_post):
    response = _response(500)
    response.raise_for_status.side_effect = httpx.HTTPStatusError("boom", request=MagicMock(), response=response)
    mock_post.return_value = response
    assert send_webhook("https://example.com/hook", "secret", {"a": 1}) is False


@patch("services.webhook_service.httpx.post")
def test_send_webhook_fails_open_on_connection_error(mock_post):
    # Never raises -- a broken receiver must not blow up the alert
    # pipeline that's trying to deliver a different, already-successful
    # notification (same posture as email_service._send_email).
    mock_post.side_effect = httpx.ConnectError("refused")
    assert send_webhook("https://example.com/hook", "secret", {"a": 1}) is False


@patch("services.webhook_service.httpx.post")
def test_send_webhook_fails_open_on_timeout(mock_post):
    mock_post.side_effect = httpx.TimeoutException("timed out")
    assert send_webhook("https://example.com/hook", "secret", {"a": 1}) is False
