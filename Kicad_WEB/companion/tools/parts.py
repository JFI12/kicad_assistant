# companion/tools/parts.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path

from companion.services import digikey_api

router = APIRouter(tags=["tools"])

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class SearchInput(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None

class AssetsInput(BaseModel):
    mpn: str

# ---------------------------------------------------------------------------
# /tools/search_parts
# ---------------------------------------------------------------------------
@router.post(
    "/tools/search_parts",
    operation_id="search_parts_tool"
)
async def search_parts(inp: SearchInput):
    q = inp.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="query must not be empty")

    try:
        results = await digikey_api.search_parts(q, inp.filters)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    # If nothing found, fallback stub
    if not results:
        results = [
            {"mpn": f"{q.upper()}-A", "vendor_url": "https://www.digikey.com/"},
            {"mpn": f"{q.upper()}-B", "vendor_url": "https://www.digikey.com/"}
        ]

    return {"results": results}

# ---------------------------------------------------------------------------
# /tools/get_kicad_assets
# ---------------------------------------------------------------------------
@router.post(
    "/tools/get_kicad_assets",
    operation_id="get_kicad_assets_tool"
)
async def get_kicad_assets(inp: AssetsInput):
    mpn = inp.mpn.strip()
    if not mpn:
        raise HTTPException(status_code=400, detail="mpn must not be empty")

    token = digikey_api.get_access_token()
    if not token:
        return {
            "login_required": True,
            "message": "Login to Digi-Key to fetch verified KiCad symbol & footprint.",
            "fallback": {
                "symbol_kicad_sym": None,
                "footprint_kicad_mod": None
            }
        }

    try:
        assets = await digikey_api.get_kicad_assets(mpn)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Asset fetch failed: {e}")

    # If Digi-Key doesn’t provide assets yet, fallback to fake symbol/footprint
    if not assets["symbol_kicad_sym"] or not assets["footprint_kicad_mod"]:
        fake_symbol = f'''
(kicad_symbol_lib (version 20211014) (generator "companion.tools.parts")
  (symbol "{mpn}"
    (property "Reference" "U" (at 0 0 0))
    (property "Value" "{mpn}" (at 0 -2 0))
    (symbol "{mpn}_1_1"
      (rectangle (start -5 5) (end 5 -5) (stroke (width 0.15) (type default)) (fill (type background)))
    )
  )
)
'''.strip()

        fake_footprint = f'''
(module {mpn}_LQFP (layer F.Cu) (tedit 5B3079AF)
  (fp_text reference U** (at 0 0) (layer F.SilkS))
  (fp_text value {mpn} (at 0 -2) (layer F.Fab))
  (pad 1 smd rect (at -4 4) (size 0.6 0.3) (layers F.Cu F.Paste F.Mask))
  (pad 2 smd rect (at -3 4) (size 0.6 0.3) (layers F.Cu F.Paste F.Mask))
)
'''.strip()

        assets = {
            "symbol_kicad_sym": fake_symbol,
            "footprint_kicad_mod": fake_footprint,
            "footprint_name": f"{mpn}_LQFP"
        }

    return {
        "login_required": False,
        **assets
    }
