# companion/tools/symbol_metadata.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import re
import json

router = APIRouter(tags=["tools"])

class ExtractReq(BaseModel):
    kicad_sym_text: str
    # Optional hint to pick a specific symbol entry name, if the file has multiple
    prefer_symbol_name: Optional[str] = None

# Simple helpers to pull (property "Name" "Value") pairs from a symbol S-expression
PROP_RE = re.compile(r'\(property\s+"([^"]+)"\s+"([^"]*)"', re.IGNORECASE)
# Try to capture symbol names like: (symbol "Lib:PartName" ...)
SYMBOL_NAME_RE = re.compile(r'\(symbol\s+"([^"]+)"', re.IGNORECASE)

def extract_properties(sym_text: str) -> Dict[str, str]:
    props: Dict[str, str] = {}
    for m in PROP_RE.finditer(sym_text):
        key, val = m.group(1), m.group(2)
        props[key] = val
    return props

def pick_symbol_block(sym_text: str, prefer_name: Optional[str]) -> str:
    """
    If multiple (symbol "...") blocks exist, pick the one that matches prefer_name when provided,
    otherwise return the whole text (properties still get extracted reasonably).
    """
    blocks = re.split(r'\n(?=\(symbol\s+")', sym_text)
    if prefer_name:
        for b in blocks:
            m = SYMBOL_NAME_RE.search(b)
            if m and prefer_name.lower() in m.group(1).lower():
                return b
    # default: return the first symbol block if present
    if len(blocks) > 1:
        return blocks[0]
    return sym_text

def guess_mpn(props: Dict[str, str], symbol_name: Optional[str]) -> Optional[str]:
    """
    Best-effort MPN from:
    - symbol name (e.g., 'MCU_ST_STM32C0:STM32C011F4Px' -> STM32C011F4)
    - Description or ki_keywords tokens
    """
    # From symbol_name: try to pull uppercase letters+digits chunk
    if symbol_name:
        # Try things like "Lib:STM32C011F4Px" -> STM32C011F4
        tail = symbol_name.split(":")[-1]
        m = re.search(r'[A-Z0-9]{6,}', tail.upper())
        if m:
            return m.group(0)

    for key in ("Description", "ki_keywords"):
        val = props.get(key, "")
        m = re.search(r'\b[A-Z0-9]{6,}\b', val.upper())
        if m:
            return m.group(0)

    return None

@router.post("/tools/extract_symbol_metadata")
async def extract_symbol_metadata(req: ExtractReq):
    if not req.kicad_sym_text:
        raise HTTPException(status_code=400, detail="kicad_sym_text is required")

    block = pick_symbol_block(req.kicad_sym_text, req.prefer_symbol_name)
    props = extract_properties(block)
    sym_name_match = SYMBOL_NAME_RE.search(block)
    symbol_name = sym_name_match.group(1) if sym_name_match else None

    meta: Dict[str, Any] = {
        "symbol_name": symbol_name,
        "datasheet": props.get("Datasheet"),
        "description": props.get("Description"),
        "keywords": props.get("ki_keywords"),
        "fp_filters": props.get("ki_fp_filters"),
    }
    meta["mpn"] = guess_mpn(props, symbol_name)

    return {"metadata": meta}
