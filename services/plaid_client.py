"""
Thin wrapper over the official `plaid-python` SDK. Real network I/O (role
analogous to data_service.py's yfinance wrappers, not a pure function) --
web/backend/plaid_sync.py and routers/plaid_integration.py are the only
callers. Kept separate from services/plaid_holdings.py (the pure mapping
logic) so that module stays importable/unit-testable without the plaid
SDK or any network access.

PLAID_CLIENT_ID/PLAID_SECRET are read lazily (os.getenv, not
os.environ[...]) so the app can start up and every other feature keeps
working before Plaid is configured -- a caller here gets a clear
RuntimeError only when it actually tries to talk to Plaid, not an
import-time crash.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

import plaid
from plaid.api import plaid_api
from plaid.model.country_code import CountryCode
from plaid.model.investments_holdings_get_request import InvestmentsHoldingsGetRequest
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from plaid.model.item_remove_request import ItemRemoveRequest
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
from plaid.model.products import Products
from plaid.model.webhook_verification_key_get_request import WebhookVerificationKeyGetRequest

APP_CLIENT_NAME = "StAnalysisEngine"


class PlaidNotConfiguredError(RuntimeError):
    pass


class PlaidItemLoginRequiredError(RuntimeError):
    """The linked item needs the user to re-authenticate (Link update
    mode) -- a real Plaid ITEM_LOGIN_REQUIRED error, not a transient
    network failure. Callers (plaid_sync.py) must not delete/clear a
    portfolio's positions on this; only mark the item's status."""


def _plaid_environment() -> Any:
    env = os.getenv("PLAID_ENV", "sandbox").strip().lower()
    if env == "sandbox":
        return plaid.Environment.Sandbox
    if env == "production":
        return plaid.Environment.Production
    raise PlaidNotConfiguredError(
        f"PLAID_ENV must be 'sandbox' or 'production', got {env!r} "
        "(Plaid retired the separate 'development' environment)."
    )


@lru_cache(maxsize=1)
def get_plaid_client() -> plaid_api.PlaidApi:
    client_id = os.getenv("PLAID_CLIENT_ID")
    secret = os.getenv("PLAID_SECRET")
    if not client_id or not secret:
        raise PlaidNotConfiguredError(
            "PLAID_CLIENT_ID/PLAID_SECRET are not set -- Plaid isn't configured on this server yet."
        )
    configuration = plaid.Configuration(
        host=_plaid_environment(),
        api_key={"clientId": client_id, "secret": secret},
    )
    return plaid_api.PlaidApi(plaid.ApiClient(configuration))


def _plaid_error_code(exc: plaid.ApiException) -> str | None:
    try:
        return json.loads(exc.body).get("error_code")
    except Exception:
        return None


def create_link_token(user_id: str) -> str:
    request = LinkTokenCreateRequest(
        products=[Products("investments")],
        client_name=APP_CLIENT_NAME,
        country_codes=[CountryCode("US")],
        language="en",
        user=LinkTokenCreateRequestUser(client_user_id=str(user_id)),
    )
    response = get_plaid_client().link_token_create(request)
    return response["link_token"]


def exchange_public_token(public_token: str) -> tuple[str, str]:
    """Returns (access_token, plaid_item_id)."""
    request = ItemPublicTokenExchangeRequest(public_token=public_token)
    response = get_plaid_client().item_public_token_exchange(request)
    return response["access_token"], response["item_id"]


def fetch_holdings(access_token: str) -> dict:
    """Raises PlaidItemLoginRequiredError on ITEM_LOGIN_REQUIRED, or
    plaid.ApiException for any other Plaid-side failure -- callers
    (plaid_sync.py) decide what to do with each, but neither is silently
    swallowed here."""
    request = InvestmentsHoldingsGetRequest(access_token=access_token)
    try:
        response = get_plaid_client().investments_holdings_get(request)
    except plaid.ApiException as e:
        if _plaid_error_code(e) == "ITEM_LOGIN_REQUIRED":
            raise PlaidItemLoginRequiredError(str(e)) from e
        raise
    return {
        "accounts": response["accounts"],
        "holdings": response["holdings"],
        "securities": response["securities"],
    }


def remove_item(access_token: str) -> None:
    """Best-effort -- callers should still delete the local plaid_items
    row even if this raises (the user's intent to disconnect shouldn't
    be blocked by a Plaid-side hiccup)."""
    request = ItemRemoveRequest(access_token=access_token)
    get_plaid_client().item_remove(request)


@lru_cache(maxsize=32)
def get_webhook_verification_key(key_id: str) -> dict:
    """Cached per key_id -- Plaid's own docs describe these as long-lived
    and rotated infrequently; re-fetching on every webhook would be an
    unnecessary Plaid API call per delivery."""
    request = WebhookVerificationKeyGetRequest(key_id=key_id)
    response = get_plaid_client().webhook_verification_key_get(request)
    return dict(response["key"])
