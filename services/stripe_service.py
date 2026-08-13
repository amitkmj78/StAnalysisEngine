"""
Horizon 1 (docs/signal-licensing-whitelabel-requirements.md.pdf, RS-2)
payment layer — a thin wrapper over Stripe Checkout (subscribe) and the
Stripe Customer Portal (self-service manage/cancel, satisfying RS-2's
"access continues to period end" without any custom billing UI).

No Stripe keys are configured anywhere yet, by design (see
HORIZON1_SUBSCRIPTIONS_ENABLED_KEY in web/backend/app_settings.py) — every
function here raises StripeNotConfiguredError with a clear message rather
than crashing when STRIPE_SECRET_KEY/STRIPE_PRICE_ID/STRIPE_WEBHOOK_SECRET
are unset, the same fail-clear pattern services/email_service.py uses for
GMAIL_SENDER_EMAIL/GMAIL_APP_PASSWORD.
"""

import os

import stripe


class StripeNotConfiguredError(Exception):
    """Raised instead of letting the stripe SDK fail on a missing/invalid
    api_key — callers (the subscriptions router) turn this into a clear
    503, not an unhandled 500."""


def _require_api_key() -> None:
    secret_key = os.environ.get("STRIPE_SECRET_KEY")
    if not secret_key:
        raise StripeNotConfiguredError(
            "STRIPE_SECRET_KEY is not set — Horizon 1 payment endpoints are built but not yet configured."
        )
    stripe.api_key = secret_key


def _require_price_id() -> str:
    price_id = os.environ.get("STRIPE_PRICE_ID")
    if not price_id:
        raise StripeNotConfiguredError(
            "STRIPE_PRICE_ID is not set — create the recurring subscription price in the Stripe "
            "dashboard first and set this to its price_... id."
        )
    return price_id


def create_checkout_session(user_id: str, email: str, success_url: str, cancel_url: str) -> str:
    """Returns the Stripe-hosted Checkout URL to redirect the subscriber
    to. client_reference_id/metadata carry user_id through to the webhook
    so checkout.session.completed can be matched back to our own user
    without ever handling card data ourselves (RS-2, NFR-05)."""
    _require_api_key()
    price_id = _require_price_id()
    session = stripe.checkout.Session.create(
        mode="subscription",
        customer_email=email,
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=success_url,
        cancel_url=cancel_url,
        client_reference_id=user_id,
        metadata={"user_id": user_id},
        subscription_data={"metadata": {"user_id": user_id}},
    )
    return session.url


def create_portal_session(stripe_customer_id: str, return_url: str) -> str:
    """Returns the Stripe-hosted Customer Portal URL — subscribers manage
    payment method and cancel here directly; Stripe's default portal
    behavior already keeps access through the current period end, which
    is exactly RS-2's "self-service cancellation with access continuing
    to period end" requirement, with no custom cancellation flow to build
    or get wrong."""
    _require_api_key()
    session = stripe.billing_portal.Session.create(customer=stripe_customer_id, return_url=return_url)
    return session.url


def verify_and_parse_webhook(payload: bytes, sig_header: str) -> stripe.Event:
    """Verifies the Stripe-Signature header against STRIPE_WEBHOOK_SECRET
    and returns the parsed event, or raises. Never trust an unverified
    webhook body — anyone could POST a fake "subscription active" event
    otherwise."""
    _require_api_key()
    webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET")
    if not webhook_secret:
        raise StripeNotConfiguredError("STRIPE_WEBHOOK_SECRET is not set — cannot verify webhook signatures yet.")
    return stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
