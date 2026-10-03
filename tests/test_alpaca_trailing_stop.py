import httpx
import pytest

from services import alpaca_trading_client as client


class _Recorder:
    def __init__(self, status=200, body=None):
        self.calls = []
        self.status = status
        self.body = body if body is not None else {}

    def __call__(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return httpx.Response(self.status, json=self.body, request=httpx.Request(method, url))


def test_trailing_stop_payload_is_sell_gtc_with_trail_percent(monkeypatch):
    rec = _Recorder(body={"id": "abc"})
    monkeypatch.setattr(client.httpx, "request", rec, raising=False)
    monkeypatch.setattr(client.httpx, "post", lambda url, **kw: rec("POST", url, **kw))

    out = client.submit_trailing_stop(
        "K", "S", client_order_id="cid-1", ticker="AAPL", qty=10, trail_percent=5.5
    )

    assert out == {"id": "abc"}
    method, url, kwargs = rec.calls[0]
    assert method == "POST" and url.endswith("/orders")
    payload = kwargs["json"]
    assert payload["type"] == "trailing_stop"
    assert payload["side"] == "sell"
    assert payload["time_in_force"] == "gtc"
    assert payload["trail_percent"] == "5.5"
    assert payload["client_order_id"] == "cid-1"


def test_trailing_stop_rejection_raises_definite_error(monkeypatch):
    rec = _Recorder(status=422, body={"message": "insufficient qty"})
    monkeypatch.setattr(client.httpx, "post", lambda url, **kw: rec("POST", url, **kw))
    with pytest.raises(client.AlpacaTradingError) as exc:
        client.submit_trailing_stop("K", "S", client_order_id="c", ticker="X", qty=1, trail_percent=3)
    assert exc.value.status_code == 422
    assert "insufficient qty" in exc.value.message


def test_cancel_order_calls_delete_and_succeeds_on_204(monkeypatch):
    rec = _Recorder(status=204)
    monkeypatch.setattr(client.httpx, "delete", lambda url, **kw: rec("DELETE", url, **kw))
    assert client.cancel_order("K", "S", "order-9") is None
    assert rec.calls[0][1].endswith("/orders/order-9")


def test_cancel_of_closed_order_surfaces_error(monkeypatch):
    rec = _Recorder(status=422, body={"message": "order is already in a terminal state"})
    monkeypatch.setattr(client.httpx, "delete", lambda url, **kw: rec("DELETE", url, **kw))
    with pytest.raises(client.AlpacaTradingError):
        client.cancel_order("K", "S", "order-9")
