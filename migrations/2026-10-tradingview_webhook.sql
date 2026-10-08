-- ALX-4: a per-user secret token embedded in the webhook URL TradingView
-- posts its alerts to (POST /api/v1/webhooks/tradingview/{token}) --
-- TradingView alerts have no custom-header signing capability, so the
-- token in the path IS the authentication, same role webhook_secret
-- plays for the outbound ALR-3 webhook, just inbound here.
alter table user_notification_settings add column if not exists tradingview_webhook_token text unique;
