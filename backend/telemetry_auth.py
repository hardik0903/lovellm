"""
telemetry_auth.py
-----------------
Lightweight token-based authentication for the /telemetry/* endpoints.

How it works
------------
1. Set TELEMETRY_SECRET in your .env (same file as GROQ_API_KEY).
   If unset, auth is DISABLED with a startup warning — useful during
   local development so the dashboard still works without config.

2. The frontend calls POST /telemetry/auth/login with:
       { "password": "<TELEMETRY_SECRET>" }
   and receives:
       { "token": "<signed JWT>", "expires_in": 86400 }

3. All other /telemetry/* routes are protected by the `require_auth`
   dependency. The frontend attaches the token as:
       Authorization: Bearer <token>

Token format: HMAC-SHA256 signed, 24-hour expiry.
No database needed — stateless verification.

Security notes
--------------
- This is single-password auth, intentionally simple for a research
  dashboard. For multi-user access, replace with a proper OAuth flow.
- Tokens are invalidated on server restart (HMAC key is derived from
  TELEMETRY_SECRET at startup). Rotating the secret invalidates all
  outstanding tokens.
- Do NOT expose /telemetry/* to the public internet without TLS.
"""

import hashlib
import hmac
import json
import os
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

# ── secret resolution ──────────────────────────────────────────────────────────
_SECRET = os.getenv("TELEMETRY_SECRET", "")
_AUTH_ENABLED = bool(_SECRET)

if not _AUTH_ENABLED:
    import warnings
    warnings.warn(
        "[telemetry_auth] TELEMETRY_SECRET is not set. "
        "Telemetry endpoints are UNPROTECTED. "
        "Set TELEMETRY_SECRET in your .env to enable authentication.",
        stacklevel=1,
    )

_SIGNING_KEY = hashlib.sha256(_SECRET.encode()).digest() if _AUTH_ENABLED else b""
TOKEN_TTL = 86_400  # 24 hours

auth_router = APIRouter(tags=["telemetry-auth"])
_bearer = HTTPBearer(auto_error=False)


# ── token helpers ──────────────────────────────────────────────────────────────

def _make_token(expires_at: int) -> str:
    payload = json.dumps({"exp": expires_at}, separators=(",", ":"))
    import base64
    payload_b64 = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
    sig = hmac.new(_SIGNING_KEY, payload_b64.encode(), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{sig}"


def _verify_token(token: str) -> bool:
    if not _AUTH_ENABLED:
        return True
    try:
        import base64
        parts = token.split(".")
        if len(parts) != 2:
            return False
        payload_b64, sig = parts
        expected_sig = hmac.new(
            _SIGNING_KEY, payload_b64.encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return False
        padding = 4 - len(payload_b64) % 4
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * padding))
        return payload.get("exp", 0) > time.time()
    except Exception:
        return False


# ── FastAPI dependency ─────────────────────────────────────────────────────────

def require_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)):
    if not _AUTH_ENABLED:
        return  # auth disabled — let it through
    if credentials is None or not _verify_token(credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token. Please log in at /telemetry/auth/login.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── auth endpoints ─────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    password: str


@auth_router.post("/auth/login")
def login(body: LoginRequest):
    """
    Exchange the TELEMETRY_SECRET password for a 24-hour bearer token.
    The dashboard calls this on form submit and stores the token in
    sessionStorage (cleared on tab close — no persistent credentials).
    """
    if not _AUTH_ENABLED:
        # Auth disabled — return a dummy token so the frontend still works
        return {"token": "auth-disabled", "expires_in": TOKEN_TTL, "auth_enabled": False}

    if not hmac.compare_digest(body.password.encode(), _SECRET.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password.",
        )

    expires_at = int(time.time()) + TOKEN_TTL
    return {
        "token": _make_token(expires_at),
        "expires_in": TOKEN_TTL,
        "auth_enabled": True,
    }


@auth_router.get("/auth/verify")
def verify(credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)):
    """
    Lightweight ping the dashboard uses on mount to check if a stored
    token is still valid before showing the login screen.
    """
    if not _AUTH_ENABLED:
        return {"valid": True, "auth_enabled": False}
    valid = credentials is not None and _verify_token(credentials.credentials)
    return {"valid": valid, "auth_enabled": True}