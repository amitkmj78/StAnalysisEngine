import logging
import os
import smtplib
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

APP_URL = os.environ.get("APP_URL", "http://54.242.22.56")

WELCOME_SUBJECT = "You're approved — welcome to StAnalysisEngine"

WELCOME_BODY_TEMPLATE = """Hi,

Your StAnalysisEngine account has been approved. Sign in here: {app_url}/login

StAnalysisEngine is an AI-assisted stock analysis toolkit. A few places to start:

- Predict — AI price forecasts and buy/sell/hold signals for any ticker, with a running accuracy
  record you can check anytime.
- Stock Finder — screen and rank stocks across configurable universes by return, volatility, and
  other metrics.
- Portfolio — track your real positions, P&L, and after-hours prices in one place.
- Track Record — a public, independently-verifiable daily top-picks ledger, published
  automatically and never edited after the fact.
- Watchlist, Trade Journal, Entry Strategy, Monthly Investing Plan, Index Fund Finder — tools
  for the rest of the research-to-trade workflow.

Thanks,
StAnalysisEngine
"""


def send_welcome_email(to_email: str) -> bool:
    """
    Best-effort: a delivery failure must never block the admin's approval
    action itself, so this always returns a bool rather than raising —
    callers log a failed send but don't fail the request over it. Skips
    sending (and returns False) if GMAIL_SENDER_EMAIL / GMAIL_APP_PASSWORD
    aren't configured, e.g. in local dev where nobody wants test emails
    going out. GMAIL_APP_PASSWORD must be a Gmail App Password, not the
    account's real password.
    """
    sender = os.environ.get("GMAIL_SENDER_EMAIL")
    app_password = os.environ.get("GMAIL_APP_PASSWORD")
    if not sender or not app_password:
        logger.warning(
            "Welcome email skipped for %s: GMAIL_SENDER_EMAIL/GMAIL_APP_PASSWORD not configured", to_email
        )
        return False

    msg = MIMEText(WELCOME_BODY_TEMPLATE.format(app_url=APP_URL))
    msg["Subject"] = WELCOME_SUBJECT
    msg["From"] = sender
    msg["To"] = to_email

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(sender, app_password)
            server.sendmail(sender, [to_email], msg.as_string())
        return True
    except Exception as e:
        logger.warning("Welcome email failed to send to %s: %s", to_email, e)
        return False
