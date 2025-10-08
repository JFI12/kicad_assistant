# companion/tools/oauth.py
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from pathlib import Path
import os, httpx
from urllib.parse import urlencode

router = APIRouter(tags=["auth"])

# ---------------------------------------------------------------------------
# Paths and token storage
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]   # companion/
WORKSPACE = BASE_DIR / "workspace"
TOKENS_DIR = WORKSPACE / "tokens"
TOKENS_DIR.mkdir(parents=True, exist_ok=True)
DIGIKEY_TOKEN_FILE = TOKENS_DIR / "digikey.json"

# ---------------------------------------------------------------------------
# Digi-Key config (from .env or environment)
# ---------------------------------------------------------------------------
DIGIKEY_CLIENT_ID = os.environ.get("DIGIKEY_CLIENT_ID", "YOUR_CLIENT_ID")
DIGIKEY_CLIENT_SECRET = os.environ.get("DIGIKEY_CLIENT_SECRET", "YOUR_SECRET")
DIGIKEY_REDIRECT_URI = os.environ.get("DIGIKEY_REDIRECT_URI", "http://localhost:8000/api/oauth/digikey/callback")

DIGIKEY_AUTH_URL = "https://api.digikey.com/v1/oauth2/authorize"
DIGIKEY_TOKEN_URL = "https://api.digikey.com/v1/oauth2/token"
DIGIKEY_SCOPES = ["product.read", "libraries.read"]

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/api/oauth/digikey/start",
    operation_id="digikey_start_auth"
)
def digikey_start():
    """
    Step 1: Called by frontend. Returns Digi-Key login URL.
    Frontend should open this in a popup window.
    """
    params = {
        "response_type": "code",
        "client_id": DIGIKEY_CLIENT_ID,
        "redirect_uri": DIGIKEY_REDIRECT_URI,
        "scope": " ".join(DIGIKEY_SCOPES),
        "state": "nonce-123",  # TODO: randomize and track in session
    }
    return JSONResponse({"auth_url": f"{DIGIKEY_AUTH_URL}?{urlencode(params)}"})


@router.get(
    "/api/oauth/digikey/callback",
    operation_id="digikey_callback_auth"
)
async def digikey_callback(code: str, state: str):
    """
    Step 2: Digi-Key redirects here after user login.
    We exchange `code` for access/refresh tokens and save them.
    """
    async with httpx.AsyncClient() as client:
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": DIGIKEY_REDIRECT_URI,
            "client_id": DIGIKEY_CLIENT_ID,
            "client_secret": DIGIKEY_CLIENT_SECRET,
        }
        r = await client.post(DIGIKEY_TOKEN_URL, data=data)
        if r.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Token exchange failed: {r.text}")

        tokens = r.json()
        DIGIKEY_TOKEN_FILE.write_text(r.text, encoding="utf-8")

    # After success, redirect frontend to a "login complete" page
    return RedirectResponse(url="/oauth-complete")


@router.get(
    "/api/oauth/digikey/status",
    operation_id="digikey_status_auth"
)
def digikey_status():
    """
    Check whether we have a token saved.
    """
    if DIGIKEY_TOKEN_FILE.exists() and DIGIKEY_TOKEN_FILE.stat().st_size > 0:
        return {"logged_in": True, "token_path": str(DIGIKEY_TOKEN_FILE.resolve())}
    return {"logged_in": False}
