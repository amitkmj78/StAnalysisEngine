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


def _send_email(to_email: str, subject: str, text_body: str, html_body: str | None = None) -> bool:
    """
    Shared SMTP mechanics for every outbound email this app sends. Skips
    sending (and returns False) if GMAIL_SENDER_EMAIL / GMAIL_APP_PASSWORD
    aren't configured, e.g. in local dev where nobody wants test emails
    going out. GMAIL_APP_PASSWORD must be a Gmail App Password, not the
    account's real password. Best-effort: never raises — a delivery
    failure must never block whatever triggered the send (an admin
    action, a scheduled job), so callers get a bool and log accordingly.
    """
    sender = os.environ.get("GMAIL_SENDER_EMAIL")
    app_password = os.environ.get("GMAIL_APP_PASSWORD")
    if not sender or not app_password:
        logger.warning("Email skipped for %s (%s): GMAIL_SENDER_EMAIL/GMAIL_APP_PASSWORD not configured", to_email, subject)
        return False

    if html_body:
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))
    else:
        msg = MIMEText(text_body, "plain")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_email

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(sender, app_password)
            server.sendmail(sender, [to_email], msg.as_string())
        return True
    except Exception as e:
        logger.warning("Email failed to send to %s (%s): %s", to_email, subject, e)
        return False


def send_welcome_email(to_email: str) -> bool:
    """
    Sends multipart/alternative (plain text + HTML) — the HTML part is
    what most clients render, the plain text part is a fallback for
    clients/screen readers that don't render HTML. See _send_email for
    the fail-open behavior when mail isn't configured.
    """
    text_body, html_body = _render_welcome_email(APP_URL)
    return _send_email(to_email, WELCOME_SUBJECT, text_body, html_body)


RESET_SUBJECT = "Reset your StAnalysisEngine password"

RESET_TEXT_TEMPLATE = """Hi,

Someone (hopefully you) requested a password reset for this account.

Reset your password: {reset_link}

This link expires in 1 hour and can only be used once. If you didn't request this, no action is
needed — your password will not be changed.

Thanks,
StAnalysisEngine
"""

RESET_HTML_TEMPLATE = """\
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
                  Reset your password
                </p>
                <p style="margin:0 0 24px;font-family:Arial,Helvetica,sans-serif;font-size:14px;line-height:1.6;color:#475569;">
                  Someone (hopefully you) requested a password reset for this account. This link expires
                  in 1 hour and can only be used once.
                </p>
                <table role="presentation" cellpadding="0" cellspacing="0">
                  <tr>
                    <td style="border-radius:8px;background-color:#0f172a;">
                      <a href="{reset_link}" style="display:inline-block;padding:12px 24px;font-family:Arial,Helvetica,sans-serif;font-size:14px;font-weight:bold;color:#ffffff;text-decoration:none;">
                        Reset password
                      </a>
                    </td>
                  </tr>
                </table>
                <p style="margin:24px 0 0;font-family:Arial,Helvetica,sans-serif;font-size:13px;line-height:1.6;color:#94a3b8;">
                  If you didn't request this, no action is needed — your password will not be changed.
                </p>
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


def send_password_reset_email(to_email: str, reset_link: str) -> bool:
    """
    Same fail-open behavior as every other email here: if SMTP isn't
    configured (local dev) or delivery fails, this returns False rather
    than raising — the forgot-password endpoint must still return its
    generic "check your email" response either way, so the response
    itself never reveals whether the send actually succeeded.
    """
    text_body = RESET_TEXT_TEMPLATE.format(reset_link=reset_link)
    html_body = RESET_HTML_TEMPLATE.format(reset_link=reset_link)
    return _send_email(to_email, RESET_SUBJECT, text_body, html_body)


def send_admin_alert_email(to_email: str, subject: str, body_text: str) -> bool:
    """
    NFR-01/02: a plain-text operational alert (publication delayed/missing,
    etc.) — no branded HTML needed for an internal ops email to the admin.
    Same fail-open behavior as every other email here: a broken mail
    config must never block or crash the job that's trying to report a
    *different* problem.
    """
    return _send_email(to_email, subject, body_text)


def send_rankings_email(to_email: str, target_date: str, signals: list[dict]) -> bool:
    """
    Horizon 1 (RS-3) — the day's current rankings, sent only to active
    paid subscribers (caller's responsibility to check that; this
    function just sends whatever list it's given). Plain text: this is
    impersonal research content, identical for every recipient, not a
    marketing email — RS-1's impersonality constraint applies here too,
    so this must never be templated per-recipient beyond the greeting.
    """
    subject = f"StAnalysisEngine rankings — {target_date}"
    lines = [f"Rankings for {target_date}:", ""]
    for s in signals:
        lines.append(f"  #{s['rank']}  {s['ticker']}  ({s['trailing_return_pct']:+.2f}%)")
    lines.append("")
    lines.append(f"Full history and methodology: {APP_URL}/track-record")
    lines.append("")
    lines.append(
        "This is impersonal research — the same content sent to every subscriber, describing what "
        "the model ranked and why. It is not individualized advice and not a recommendation to buy "
        "or sell anything. Past performance does not indicate future results."
    )
    return _send_email(to_email, subject, "\n".join(lines))
