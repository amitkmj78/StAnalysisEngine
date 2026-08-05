import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

APP_URL = os.environ.get("APP_URL", "http://54.242.22.56")

WELCOME_SUBJECT = "You're approved — welcome to StAnalysisEngine"

_FEATURES = [
    ("Predict", "AI price forecasts and buy/sell/hold signals for any ticker, with a running "
                 "accuracy record you can check anytime."),
    ("Stock Finder", "Screen and rank stocks across configurable universes by return, "
                      "volatility, and other metrics."),
    ("Portfolio", "Track your real positions, P&L, and after-hours prices in one place."),
    ("Track Record", "A public, independently-verifiable daily top-picks ledger, published "
                      "automatically and never edited after the fact."),
    ("More tools", "Watchlist, Trade Journal, Entry Strategy, Monthly Investing Plan, and "
                    "Index Fund Finder round out the research-to-trade workflow."),
]

WELCOME_TEXT_TEMPLATE = """Hi,

Your StAnalysisEngine account has been approved. Sign in here: {app_url}/login

StAnalysisEngine is an AI-assisted stock analysis toolkit. A few places to start:

{feature_lines}

Thanks,
StAnalysisEngine
"""

_HTML_FEATURE_ROW = """
              <tr>
                <td style="padding:10px 0;border-top:1px solid #e2e8f0;">
                  <p style="margin:0;font-family:Arial,Helvetica,sans-serif;font-size:14px;line-height:1.5;color:#0f172a;">
                    <strong>{name}</strong><br>
                    <span style="color:#475569;">{description}</span>
                  </p>
                </td>
              </tr>"""

WELCOME_HTML_TEMPLATE = """\
<!doctype html>
<html>
  <body style="margin:0;padding:0;background-color:#f1f5f9;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#f1f5f9;padding:32px 16px;">
      <tr>
        <td align="center">
          <table role="presentation" width="560" cellpadding="0" cellspacing="0" style="max-width:560px;width:100%;background-color:#ffffff;border-radius:12px;overflow:hidden;border:1px solid #e2e8f0;">
            <tr>
              <td style="background-color:#0f172a;padding:24px 32px;">
                <p style="margin:0;font-family:Arial,Helvetica,sans-serif;font-size:18px;font-weight:bold;color:#ffffff;letter-spacing:0.2px;">
                  StAnalysisEngine
                </p>
              </td>
            </tr>
            <tr>
              <td style="padding:32px;">
                <p style="margin:0 0 8px;font-family:Arial,Helvetica,sans-serif;font-size:20px;font-weight:bold;color:#0f172a;">
                  You're approved
                </p>
                <p style="margin:0 0 24px;font-family:Arial,Helvetica,sans-serif;font-size:14px;line-height:1.6;color:#475569;">
                  Your account is ready to go. Sign in below to start using it.
                </p>
                <table role="presentation" cellpadding="0" cellspacing="0">
                  <tr>
                    <td style="border-radius:8px;background-color:#0f172a;">
                      <a href="{app_url}/login" style="display:inline-block;padding:12px 24px;font-family:Arial,Helvetica,sans-serif;font-size:14px;font-weight:bold;color:#ffffff;text-decoration:none;">
                        Sign in to StAnalysisEngine
                      </a>
                    </td>
                  </tr>
                </table>
                <p style="margin:28px 0 4px;font-family:Arial,Helvetica,sans-serif;font-size:13px;font-weight:bold;text-transform:uppercase;letter-spacing:0.4px;color:#64748b;">
                  A few places to start
                </p>
                <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
{feature_rows}
                </table>
              </td>
            </tr>
            <tr>
              <td style="padding:20px 32px;background-color:#f8fafc;border-top:1px solid #e2e8f0;">
                <p style="margin:0;font-family:Arial,Helvetica,sans-serif;font-size:12px;color:#94a3b8;">
                  StAnalysisEngine · AI-assisted stock analysis
                </p>
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
"""


def _render_welcome_email(app_url: str) -> tuple[str, str]:
    """Returns (plain_text, html) for the welcome email body."""
    text = WELCOME_TEXT_TEMPLATE.format(
        app_url=app_url,
        feature_lines="\n".join(f"- {name} — {desc}" for name, desc in _FEATURES),
    )
    html = WELCOME_HTML_TEMPLATE.format(
        app_url=app_url,
        feature_rows="".join(_HTML_FEATURE_ROW.format(name=name, description=desc) for name, desc in _FEATURES),
    )
    return text, html


def send_welcome_email(to_email: str) -> bool:
    """
    Best-effort: a delivery failure must never block the admin's approval
    action itself, so this always returns a bool rather than raising —
    callers log a failed send but don't fail the request over it. Skips
    sending (and returns False) if GMAIL_SENDER_EMAIL / GMAIL_APP_PASSWORD
    aren't configured, e.g. in local dev where nobody wants test emails
    going out. GMAIL_APP_PASSWORD must be a Gmail App Password, not the
    account's real password.

    Sends multipart/alternative (plain text + HTML) — the HTML part is
    what most clients render, the plain text part is a fallback for
    clients/screen readers that don't render HTML.
    """
    sender = os.environ.get("GMAIL_SENDER_EMAIL")
    app_password = os.environ.get("GMAIL_APP_PASSWORD")
    if not sender or not app_password:
        logger.warning(
            "Welcome email skipped for %s: GMAIL_SENDER_EMAIL/GMAIL_APP_PASSWORD not configured", to_email
        )
        return False

    text_body, html_body = _render_welcome_email(APP_URL)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = WELCOME_SUBJECT
    msg["From"] = sender
    msg["To"] = to_email
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(sender, app_password)
            server.sendmail(sender, [to_email], msg.as_string())
        return True
    except Exception as e:
        logger.warning("Welcome email failed to send to %s: %s", to_email, e)
        return False
