# companion/services/digikey_api.py
from pathlib import Path
import os, json, httpx, time

BASE_DIR = Path(__file__).resolve().parents[1]   # companion/
WORKSPACE = BASE_DIR / "workspace"
TOKENS_DIR = WORKSPACE / "tokens"
TOKENS_DIR.mkdir(parents=True, exist_ok=True)
DIGIKEY_TOKEN_FILE = TOKENS_DIR / "digikey.json"

# Config from environment
CLIENT_ID = os.environ.get("DIGIKEY_CLIENT_ID", "YOUR_CLIENT_ID")
CLIENT_SECRET = os.environ.get("DIGIKEY_CLIENT_SECRET", "YOUR_SECRET")
REDIRECT_URI = os.environ.get("DIGIKEY_REDIRECT_URI", "http://localhost:8000/api/oauth/digikey/callback")

TOKEN_URL = "https://api.digikey.com/v1/oauth2/token"
PART_SEARCH_URL = "https://api.digikey.com/Search/v3/Products"
# (note: Digi-Key library endpoints are not always public; you may need to rely on GitHub KiCad libs as fallback)

# ---------------------------------------------------------------------------
# Token management
# ---------------------------------------------------------------------------

def _load_tokens():
    if not DIGIKEY_TOKEN_FILE.exists():
        return None
    return json.loads(DIGIKEY_TOKEN_FILE.read_text(encoding="utf-8"))

def _save_tokens(tokens: dict):
    DIGIKEY_TOKEN_FILE.write_text(json.dumps(tokens, indent=2), encoding="utf-8")

def _refresh_if_needed(tokens: dict) -> dict:
    """
    If token is expired, refresh it.
    """
    if not tokens:
        return None

    expires_at = tokens.get("expires_at")
    now = int(time.time())

    if expires_at and now < expires_at - 30:  # still valid
        return tokens

    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        return tokens

    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }
    r = httpx.post(TOKEN_URL, data=data)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to refresh Digi-Key token: {r.text}")

    new_tokens = r.json()
    # Add computed expiry
    if "expires_in" in new_tokens:
        new_tokens["expires_at"] = now + int(new_tokens["expires_in"])
    _save_tokens(new_tokens)
    return new_tokens

def get_access_token() -> str | None:
    tokens = _load_tokens()
    if not tokens:
        return None
    tokens = _refresh_if_needed(tokens)
    return tokens.get("access_token")

# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

async def search_parts(query: str, filters: dict | None = None) -> list[dict]:
    """
    Search Digi-Key for parts.
    Returns a list of {mpn, description, vendor_url}.
    """
    token = get_access_token()
    if not token:
        return []

    headers = {"Authorization": f"Bearer {token}"}
    params = {"Keywords": query, "RecordCount": 5}
    if filters:
        params.update(filters)

    async with httpx.AsyncClient() as client:
        r = await client.get(PART_SEARCH_URL, headers=headers, params=params)
        if r.status_code != 200:
            raise RuntimeError(f"Digi-Key search failed: {r.text}")
        data = r.json()

    results = []
    for prod in data.get("Products", []):
        results.append({
            "mpn": prod.get("ManufacturerPartNumber"),
            "description": prod.get("Description"),
            "vendor_url": prod.get("ProductUrl")
        })
    return results


async def get_kicad_assets(mpn: str) -> dict:
    """
    Placeholder: Digi-Key doesn’t have a public API for symbols/footprints.
    You can fetch from their GitHub KiCad libraries instead:
    https://github.com/Digi-Key/digikey-kicad-library
    """
    # TODO: implement GitHub fetch (symbols + footprints)
    # For now, return None and let parts.py fall back.
    return {
        "symbol_kicad_sym": None,
        "footprint_kicad_mod": None,
        "footprint_name": None
    }
