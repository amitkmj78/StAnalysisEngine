"""PLT-6: TOTP-based two-factor login. A thin, pure wrapper around pyotp --
no DB/FastAPI imports, so it's trivially unit-testable with synthetic
secrets and can never accidentally touch storage itself. The caller
(web/backend/routers/auth.py) owns encrypting the secret at rest
(web/backend/crypto_utils.py's Fernet helper, the same one paper-trading/
Plaid tokens already use) and deciding when totp_enabled actually flips.
"""

import pyotp
import qrcode
import qrcode.image.pil
from io import BytesIO
import base64

ISSUER = "StAnalysisEngine"


def generate_secret() -> str:
    return pyotp.random_base32()


def provisioning_uri(secret: str, email: str) -> str:
    return pyotp.TOTP(secret).provisioning_uri(name=email, issuer_name=ISSUER)


def verify_code(secret: str, code: str) -> bool:
    """valid_window=1 tolerates one 30s step of clock drift either way --
    the standard, narrow tolerance most TOTP implementations use (not a
    wide window that would make stolen-code replay meaningfully easier)."""
    if not code or not code.strip().isdigit():
        return False
    return pyotp.TOTP(secret).verify(code.strip(), valid_window=1)


def qr_code_data_uri(uri: str) -> str:
    """A PNG data: URI the frontend can drop straight into an <img src>,
    so enrollment needs no separate QR-rendering dependency client-side."""
    img = qrcode.make(uri, image_factory=qrcode.image.pil.PilImage)
    buf = BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
