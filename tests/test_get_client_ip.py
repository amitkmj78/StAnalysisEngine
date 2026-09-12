from unittest.mock import MagicMock

from web.backend.auth import get_client_ip


def _request(headers: dict, client_host: str | None = "127.0.0.1"):
    req = MagicMock()
    req.headers = headers
    req.client = MagicMock(host=client_host) if client_host is not None else None
    return req


def test_prefers_x_forwarded_for():
    req = _request({"x-forwarded-for": "203.0.113.7, 10.0.0.1"})
    assert get_client_ip(req) == "203.0.113.7"


def test_x_forwarded_for_single_ip():
    req = _request({"x-forwarded-for": "203.0.113.7"})
    assert get_client_ip(req) == "203.0.113.7"


def test_falls_back_to_x_real_ip():
    req = _request({"x-real-ip": "198.51.100.5"})
    assert get_client_ip(req) == "198.51.100.5"


def test_x_forwarded_for_wins_over_x_real_ip():
    req = _request({"x-forwarded-for": "203.0.113.7", "x-real-ip": "198.51.100.5"})
    assert get_client_ip(req) == "203.0.113.7"


def test_falls_back_to_request_client_host():
    req = _request({}, client_host="192.168.1.50")
    assert get_client_ip(req) == "192.168.1.50"


def test_none_when_nothing_available():
    req = _request({}, client_host=None)
    assert get_client_ip(req) is None


def test_blank_x_forwarded_for_falls_through():
    req = _request({"x-forwarded-for": "  "}, client_host="192.168.1.50")
    assert get_client_ip(req) == "192.168.1.50"
