from web.backend.db import service_conn

VERIFY_PREDICTIONS_ENABLED_KEY = "verify_predictions_enabled"


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
