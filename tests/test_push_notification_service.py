from unittest.mock import MagicMock, patch

import pywebpush

import services.push_notification_service as push_service

_SUBSCRIPTION = {"endpoint": "https://push.example.com/abc", "keys": {"p256dh": "key1", "auth": "key2"}}


def test_is_configured_false_without_vapid_keys(monkeypatch):
    monkeypatch.setattr(push_service, "VAPID_PUBLIC_KEY", None)
    monkeypatch.setattr(push_service, "VAPID_PRIVATE_KEY", None)
    assert push_service.is_configured() is False


def test_is_configured_true_with_both_keys(monkeypatch):
    monkeypatch.setattr(push_service, "VAPID_PUBLIC_KEY", "pub")
    monkeypatch.setattr(push_service, "VAPID_PRIVATE_KEY", "priv")
    assert push_service.is_configured() is True


def test_send_push_skips_when_not_configured(monkeypatch):
    monkeypatch.setattr(push_service, "VAPID_PUBLIC_KEY", None)
    monkeypatch.setattr(push_service, "VAPID_PRIVATE_KEY", None)
    sent, should_remove = push_service.send_push(_SUBSCRIPTION, "Title", "Body")
    assert (sent, should_remove) == (False, False)


def test_send_push_success(monkeypatch):
    monkeypatch.setattr(push_service, "VAPID_PUBLIC_KEY", "pub")
    monkeypatch.setattr(push_service, "VAPID_PRIVATE_KEY", "priv")
    with patch("pywebpush.webpush", return_value=None) as mock_webpush:
        sent, should_remove = push_service.send_push(_SUBSCRIPTION, "Title", "Body", url="/stock/AAPL")
    assert (sent, should_remove) == (True, False)
    kwargs = mock_webpush.call_args.kwargs
    assert kwargs["subscription_info"] == _SUBSCRIPTION
    assert kwargs["vapid_private_key"] == "priv"
    assert "Title" in kwargs["data"]


def test_send_push_expired_subscription_signals_removal(monkeypatch):
    monkeypatch.setattr(push_service, "VAPID_PUBLIC_KEY", "pub")
    monkeypatch.setattr(push_service, "VAPID_PRIVATE_KEY", "priv")
    response = MagicMock(status_code=410)
    exc = pywebpush.WebPushException("Gone", response=response)
    with patch("pywebpush.webpush", side_effect=exc):
        sent, should_remove = push_service.send_push(_SUBSCRIPTION, "Title", "Body")
    assert (sent, should_remove) == (False, True)


def test_send_push_transient_failure_does_not_signal_removal(monkeypatch):
    monkeypatch.setattr(push_service, "VAPID_PUBLIC_KEY", "pub")
    monkeypatch.setattr(push_service, "VAPID_PRIVATE_KEY", "priv")
    response = MagicMock(status_code=500)
    exc = pywebpush.WebPushException("Server error", response=response)
    with patch("pywebpush.webpush", side_effect=exc):
        sent, should_remove = push_service.send_push(_SUBSCRIPTION, "Title", "Body")
    assert (sent, should_remove) == (False, False)


def test_send_push_never_raises_on_unexpected_error(monkeypatch):
    monkeypatch.setattr(push_service, "VAPID_PUBLIC_KEY", "pub")
    monkeypatch.setattr(push_service, "VAPID_PRIVATE_KEY", "priv")
    with patch("pywebpush.webpush", side_effect=ConnectionError("network down")):
        sent, should_remove = push_service.send_push(_SUBSCRIPTION, "Title", "Body")
    assert (sent, should_remove) == (False, False)
