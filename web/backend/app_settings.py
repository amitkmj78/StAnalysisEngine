from web.backend.db import service_conn

VERIFY_PREDICTIONS_ENABLED_KEY = "verify_predictions_enabled"
# Defaults OFF deliberately: CMP-01/Q-01/Q-02 require counsel confirmation
# that unpaid, impersonal publication carries no registration requirement
# *before* the first real publication happens. Deploying the publication
# pipeline must not itself start publishing — an explicit admin opt-in does.
PUBLISH_SIGNALS_ENABLED_KEY = "publish_signals_enabled"


async def get_setting_bool(key: str, default: bool) -> bool:
    async with service_conn() as conn:
        value = await conn.fetchval("SELECT value FROM app_settings WHERE key = $1", key)
    if value is None:
        return default
    return value.lower() == "true"


async def set_setting_bool(key: str, value: bool) -> None:
    async with service_conn() as conn:
        await conn.execute(
            """
            INSERT INTO app_settings (key, value, updated_at)
            VALUES ($1, $2, now())
            ON CONFLICT (key) DO UPDATE SET value = $2, updated_at = now()
            """,
            key, "true" if value else "false",
        )
