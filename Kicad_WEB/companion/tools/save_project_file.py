# companion/tools/save_project_file.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import os
import base64

router = APIRouter(tags=["tools"])

# Root where files are saved. Prefer setting an absolute path.
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()

class SaveProjectFileReq(BaseModel):
    # Relative path (e.g., "ProjectSymbols/MCU_ST_STM32C0.kicad_sym")
    path: str
    # Text content if is_binary=False, or base64 if is_binary=True
    content: str
    is_binary: Optional[bool] = False
    make_dirs: Optional[bool] = True

@router.post("/tools/save_project_file")
async def save_project_file(req: SaveProjectFileReq):
    """
    Save a file under PROJECT_ROOT safely:
    - Ensures the path is relative (no absolute paths, no '..' traversal).
    - Supports text or base64-encoded binary writes.
    """
    rel = Path(req.path)

    if rel.is_absolute():
        raise HTTPException(status_code=400, detail="Path must be relative, not absolute.")
    if any(p == ".." for p in rel.parts):
        raise HTTPException(status_code=400, detail="Path traversal is not allowed.")

    out_path = (PROJECT_ROOT / rel).resolve()

    # Ensure the resolved path is still under PROJECT_ROOT
    try:
        out_path.relative_to(PROJECT_ROOT)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path escapes PROJECT_ROOT.")

    if req.make_dirs:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if req.is_binary:
            data = base64.b64decode(req.content)
            out_path.write_bytes(data)
            written = len(data)
        else:
            text = req.content
            out_path.write_text(text, encoding="utf-8")
            written = len(text.encode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write file: {e}")

    return {
        "ok": True,
        "path": str(out_path),
        "bytes_written": written,
        "project_root": str(PROJECT_ROOT),
    }
