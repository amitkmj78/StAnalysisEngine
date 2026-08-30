"""
In-process source of truth for which live-quote provider
services.data_service's get_latest_price/get_extended_hours_price use.

Admin-tunable (web/backend/routers/admin_settings.py), persisted in
app_settings (PRICE_DATA_PROVIDER_KEY) so the choice survives a restart —
but the value actually consulted on every price call is this plain
module-level variable, not a database read. Two reasons:

1. data_service.py's functions are synchronous and called from dozens of
   places across sync and async code; app_settings reads go through
   asyncpg's async pool, so threading an awaited settings lookup into
   every one of those call sites would be a much larger refactor for a
   value that changes maybe a few times a year.
2. This app runs as a single uvicorn process (no --workers), so a plain
   module-level global is already the same pattern this codebase uses for
   other process-wide state (e.g. services/cache_utils.py's caches) and is
   trivially consistent: the admin's POST handler updates this variable
   directly, in the same process that will serve the very next request.

web/backend/main.py's startup hook seeds this from the persisted setting
on boot; the admin endpoint updates both the DB (for the next restart)
and this variable (for right now) on every change.
"""

PRICE_PROVIDERS = ("yahoo", "alpaca")

_current_provider = "yahoo"


def get_price_provider() -> str:
    return _current_provider


def set_price_provider(provider: str) -> None:
    if provider not in PRICE_PROVIDERS:
        raise ValueError(f"provider must be one of {PRICE_PROVIDERS}, got {provider!r}")
    global _current_provider
    _current_provider = provider
