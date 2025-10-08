# companion/tools/fetch_url.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any
import os
import httpx
import base64
from urllib.parse import urlparse

router = APIRouter(tags=["tools"])

# Environment-driven safeguards
MAX_FETCH_BYTES = int(os.environ.get("MAX_FETCH_BYTES", 5_000_000))  # 5 MB default
OUTBOUND_ALLOWLIST = os.environ.get("OUTBOUND_ALLOWLIST", "")  # e.g. "gitlab.com,raw.githubusercontent.com"
ALLOWED_HOSTS = {h.strip().lower() for h in OUTBOUND_ALLOWLIST.split(",") if h.strip()}

class FetchUrlReq(BaseModel):
    url: HttpUrl
    timeout_sec: Optional[float] = 30.0
    headers: Optional[Dict[str, Any]] = None

@router.post("/tools/fetch_url")
async def fetch_url(req: FetchUrlReq):
    """
    Fetch remote content safely:
    - Optional outbound host allow-list via OUTBOUND_ALLOWLIST env.
    - Enforces max download size via MAX_FETCH_BYTES env.
    - Returns text if it's text-like, otherwise base64-encoded bytes.
    """
    parsed = urlparse(str(req.url))
    host = (parsed.netloc or "").lower()

    if ALLOWED_HOSTS and host not in ALLOWED_HOSTS:
        raise HTTPException(
            status_code=400,
            detail=f"Host '{host}' not allowed. Set OUTBOUND_ALLOWLIST to allow."
        )

    # Restrict to http/https
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="Only http/https URLs are allowed.")

    try:
        async with httpx.AsyncClient(timeout=req.timeout_sec) as client:
            # Stream so we can enforce size limits
            async with client.stream("GET", str(req.url), headers=req.headers) as r:
                r.raise_for_status()
                content_type = r.headers.get("content-type", "application/octet-stream")
                # Accumulate up to MAX_FETCH_BYTES
                chunks = []
                total = 0
                async for chunk in r.aiter_bytes():
                    total += len(chunk)
                    if total > MAX_FETCH_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Remote content exceeds MAX_FETCH_BYTES={MAX_FETCH_BYTES}"
                        )
                    chunks.append(chunk)
                data = b"".join(chunks)

        # Heuristic: treat as text if content-type starts with text/ or JSON or KiCad symbol files
        is_probably_text = (
            content_type.startswith("text/")
            or "json" in content_type
            or str(req.url).lower().endswith(".kicad_sym")
        )

        if is_probably_text:
            try:
                text = data.decode("utf-8")
                return {
                    "url": str(req.url),
                    "status": 200,
                    "content_type": content_type,
                    "is_binary": False,
                    "size": len(data),
                    "content": text,
                    "encoding": "utf-8",
                }
            except UnicodeDecodeError:
                # Fallback to base64 if it isn't valid UTF-8
                pass

        # Binary (or undecodable text): return base64
        b64 = base64.b64encode(data).decode("ascii")
        return {
            "url": str(req.url),
            "status": 200,
            "content_type": content_type,
            "is_binary": True,
            "size": len(data),
            "content_b64": b64,
        }

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Network error: {e}")
