# companion/tools/project_metadata.py
from fastapi import APIRouter
from pathlib import Path
from typing import Any, Dict
import os
import json

router = APIRouter(tags=["project"])

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()
META_FILE = PROJECT_ROOT / "ProjectSymbols" / "metadata.json"

@router.get("/project/metadata")
async def get_project_metadata() -> Dict[str, Any]:
    if META_FILE.is_file():
        try:
            return json.loads(META_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}
