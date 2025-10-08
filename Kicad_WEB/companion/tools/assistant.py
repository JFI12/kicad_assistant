# companion/tools/assistant.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# ✅ Importera och anropa dina verktyg direkt (ingen HTTP-self-call)
from companion.tools.parts import (
    SearchInput,
    AssetsInput,
    search_parts as tool_search_parts,           # async
    get_kicad_assets as tool_get_kicad_assets,   # async
)
from companion.tools.kicad import (
    AddSymInput,
    AddFptInput,
    add_symbol_to_project as tool_add_symbol,          # async
    add_footprint_to_project as tool_add_footprint,    # async
)
from companion.tools.pinout import (
    PinoutInput,
    generate_pinout_excel as tool_generate_pinout_excel,   # async
)
from companion.tools.schematic import (
    InsertWithFootprintReq,
    schematic_insert_with_footprint as tool_insert_with_footprint,  # sync
)
from companion.services import digikey_api

router = APIRouter(tags=["assistant"])


class AddPartReq(BaseModel):
    query_or_mpn: str
    x: float = 0
    y: float = 0
    ref_prefix: str = "U"
    annotate: bool = True  # kör annotate_references i slutet


@router.post("/assistant/add_part", operation_id="assistant_add_part")
async def assistant_add_part(req: AddPartReq):
    """
    One-shot:
      - sök (om text) eller använd given MPN
      - hämta symbol + footprint (kräver Digi-Key inlogg)
      - skriv in i projektets bibliotek
      - infoga symbol i schemat + sätt Footprint + (valfritt) annotera
      - generera pinout Excel
    """
    q = req.query_or_mpn.strip()
    if not q:
        raise HTTPException(400, "query_or_mpn must not be empty")

    # 1) välj MPN (sök om det ser ut som en fri text)
    mpn = q
    if " " in q:
        try:
            sr = await tool_search_parts(SearchInput(query=q))
            results = sr.get("results", []) if isinstance(sr, dict) else sr
        except Exception as e:
            raise HTTPException(500, f"search_parts failed: {e}")
        if not results:
            raise HTTPException(404, f"Hittade inga delar för '{q}'")
        mpn = results[0]["mpn"]

    # 2) kolla Digi-Key inloggning
    token = digikey_api.get_access_token()
    if not token:
        return {"login_required": True, "message": "Logga in på Digi-Key först (/api/oauth/digikey/start)."}

    # 3) hämta assets
    try:
        assets = await tool_get_kicad_assets(AssetsInput(mpn=mpn))
    except Exception as e:
        raise HTTPException(500, f"get_kicad_assets failed: {e}")

    if isinstance(assets, dict) and assets.get("login_required"):
        return {"login_required": True, "message": "Digi-Key kräver inloggning."}

    symbol_text = assets["symbol_kicad_sym"]
    footprint_text = assets["footprint_kicad_mod"]
    fp_name = assets.get("footprint_name", f"{mpn}_pkg")

    if not symbol_text or not footprint_text:
        raise HTTPException(500, "Saknar symbol eller footprint i assets-responsen.")

    # 4) skriv in i projektbiblioteket
    try:
        await tool_add_symbol(AddSymInput(symbol_kicad_sym=symbol_text, library_name="ProjectSymbols"))
    except Exception as e:
        raise HTTPException(500, f"add_symbol_to_project failed: {e}")

    try:
        await tool_add_footprint(AddFptInput(footprint_kicad_mod=footprint_text, footprint_name=fp_name))
    except Exception as e:
        raise HTTPException(500, f"add_footprint_to_project failed: {e}")

    # 5) infoga i schemat + footprint + (valfritt) annotera (sync-funktion)
    try:
        ins = tool_insert_with_footprint(
            InsertWithFootprintReq(
                lib_id=f"ProjectSymbols:{mpn}",
                x=req.x,
                y=req.y,
                ref_prefix=req.ref_prefix,
                value=mpn,
                footprint_lib="ProjectFootprints",
                footprint_name=fp_name,
                annotate=req.annotate,
            )
        )
    except Exception as e:
        raise HTTPException(500, f"insert_with_footprint failed: {e}")

    # 6) pinout excel (icke-kritisk)
    try:
        pin = await tool_generate_pinout_excel(PinoutInput(mpn=mpn))
        pinout_path = pin.get("excel_path") if isinstance(pin, dict) else None
    except Exception:
        pinout_path = None

    return {
        "ok": True,
        "mpn": mpn,
        "uuid": ins["uuid"],
        "reference": ins["reference"],
        "footprint": ins["footprint"],
        "pinout_excel": pinout_path,
        "annotate_counts": ins.get("annotate_counts"),
    }
