from datetime import date

import pytest

from services.stripe_service import StripeNotConfiguredError, create_checkout_session, create_portal_session
from services.subscription_access_service import compute_free_tier_target_date


# ---------------------------------------------------------------------------
# compute_free_tier_target_date (RS-2 free-tier lag window)
# ---------------------------------------------------------------------------

def test_no_request_defaults_to_lagged_latest():
    target, is_lagged = compute_free_tier_target_date(
        requested_date=None,
        latest_date=date(2026, 8, 13),
        record_start_date=date(2026, 1, 1),
        lag_days=7,
    )
    assert target == date(2026, 8, 6)
    assert is_lagged is True


def test_requesting_too_recent_a_date_gets_capped():
    target, is_lagged = compute_free_tier_target_date(
        requested_date=date(2026, 8, 12),
        latest_date=date(2026, 8, 13),
        record_start_date=date(2026, 1, 1),
        lag_days=7,
    )
    assert target == date(2026, 8, 6)
    assert is_lagged is True


def test_requesting_an_already_old_date_passes_through():
    target, is_lagged = compute_free_tier_target_date(
        requested_date=date(2026, 1, 5),
        latest_date=date(2026, 8, 13),
        record_start_date=date(2026, 1, 1),
        lag_days=7,
    )
    assert target == date(2026, 1, 5)
    assert is_lagged is True


def test_clamped_to_record_start_for_a_young_record():
    # Record only 3 days old — a 7-day lag would push the cutoff before
    # the record even started; free tier should see day 1, not nothing.
    target, is_lagged = compute_free_tier_target_date(
        requested_date=None,
        latest_date=date(2026, 8, 13),
        record_start_date=date(2026, 8, 11),
        lag_days=7,
    )
    assert target == date(2026, 8, 11)
    assert is_lagged is True


def test_no_publications_yet_passes_through_unchanged():
    target, is_lagged = compute_free_tier_target_date(
        requested_date=None,
        latest_date=None,
        record_start_date=None,
        lag_days=7,
    )
    assert target is None
    assert is_lagged is False


# ---------------------------------------------------------------------------
# stripe_service — fails clearly, never crashes, when unconfigured
# ---------------------------------------------------------------------------

def test_checkout_session_raises_clearly_when_unconfigured(monkeypatch):
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    with pytest.raises(StripeNotConfiguredError):
        create_checkout_session("user-1", "a@b.com", "https://x/success", "https://x/cancel")


def test_portal_session_raises_clearly_when_unconfigured(monkeypatch):
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    with pytest.raises(StripeNotConfiguredError):
        create_portal_session("cus_123", "https://x/return")


def test_checkout_session_raises_clearly_when_price_id_missing(monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_fake")
    monkeypatch.delenv("STRIPE_PRICE_ID", raising=False)
    with pytest.raises(StripeNotConfiguredError):
        create_checkout_session("user-1", "a@b.com", "https://x/success", "https://x/cancel")
