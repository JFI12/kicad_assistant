# companion/tools/pinout.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from pathlib import Path
import os

import pandas as pd  # pip install pandas openpyxl

router = APIRouter(tags=["tools"])

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # companion/
PROJECT_ROOT = Path(os.environ.get("KICAD_PROJECT_ROOT", str(BASE_DIR / "workspace")))
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

ARTIFACTS = PROJECT_ROOT / "Artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
PinType = Literal["Power", "Ground", "GPIO", "Analog", "Crystal", "Reset", "Boot", "NRF", "VREF", "USB", "SWD", "Other"]

class Pin(BaseModel):
    number: int = Field(..., description="Physical pin number as on package")
    name: str = Field(..., description="Pin name, e.g., VDD, PA0, NRST")
    type: PinType = "Other"
    domain: Optional[str] = Field(None, description="Power domain, e.g., 'VDD', 'VDDA', 'VBAT'")
    default_net: Optional[str] = None
    notes: Optional[str] = None

class PinoutInput(BaseModel):
    mpn: str
    datasheet_url: Optional[str] = None
    pins: Optional[List[Pin]] = None  # If omitted, we create a starter template

# ---------------------------------------------------------------------------
# Heuristics for quick decoupling/notes
# ---------------------------------------------------------------------------
def _suggest_defaults(pin: Pin) -> Pin:
    n = pin.name.upper()
    out = pin.model_copy()

    # Power / ground quick labels
    if any(k in n for k in ["GND", "VSS"]):
        out.type = "Ground"
        out.default_net = "GND"
    elif n.startswith("VDD") or n in {"VDDA", "VDDD", "VDDIO", "VBAT", "VCC", "VREF+"}:
        out.type = "Power"
        out.default_net = "3V3" if "BAT" not in n else "VBAT"

        # Add simple decoupling guidance
        if out.notes:
            out.notes += " | "
        out.notes = (out.notes or "") + "Place 0.1µF close to pin; add 1µF per domain."
        if out.name.upper() in {"VDDA", "VREF+"}:
            out.notes += " Consider analog filtering per datasheet."
    elif n in {"NRST", "RESET", "RST"}:
        out.type = "Reset"
        out.notes = (out.notes or "") + " Pull-up resistor to VDD; add reset button header."
    elif n in {"PH0", "PH1", "OSC_IN", "OSC_OUT"}:
        out.type = "Crystal"
        out.notes = (out.notes or "") + " Add crystal + load capacitors per datasheet."
    elif n in {"SWDIO", "SWCLK", "SWO"}:
        out.type = "SWD"
        out.notes = (out.notes or "") + " Provide SWD header."
    elif n.startswith("PA") or n.startswith("PB") or n.startswith("PC") or n.startswith("PD") or n.startswith("PE") or n.startswith("PF") or n.startswith("PG") or n.startswith("PH") or n.startswith("PI"):
        out.type = "GPIO"

    return out

# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@router.post(
    "/tools/generate_pinout_excel",
    operation_id="generate_pinout_excel_tool"
)
async def generate_pinout_excel(inp: PinoutInput):
    mpn = inp.mpn.strip()
    if not mpn:
        raise HTTPException(status_code=400, detail="mpn must not be empty")

    # Build rows
    rows = []
    if inp.pins and len(inp.pins) > 0:
        for p in inp.pins:
            p2 = _suggest_defaults(p)
            rows.append({
                "PinNumber": p2.number,
                "PinName": p2.name,
                "Type": p2.type,
                "PowerDomain": p2.domain or "",
                "DefaultNet": p2.default_net or "",
                "Notes": p2.notes or ""
            })
    else:
        # starter template (you can overwrite with a real parsed list later)
        stubs = [
            Pin(number=1, name="VDD"),
            Pin(number=2, name="VSS"),
            Pin(number=3, name="NRST"),
            Pin(number=4, name="PA0"),
            Pin(number=5, name="PA1"),
            Pin(number=6, name="PB0"),
            Pin(number=7, name="PH0"),  # crystal in
            Pin(number=8, name="PH1"),  # crystal out
        ]
        for p in stubs:
            p2 = _suggest_defaults(p)
            rows.append({
                "PinNumber": p2.number,
                "PinName": p2.name,
                "Type": p2.type,
                "PowerDomain": p2.domain or "",
                "DefaultNet": p2.default_net or "",
                "Notes": p2.notes or ""
            })

    df = pd.DataFrame(rows, columns=["PinNumber", "PinName", "Type", "PowerDomain", "DefaultNet", "Notes"])

    out_path = ARTIFACTS / f"{mpn}_pinout.xlsx"
    df.to_excel(out_path, index=False)

    meta = {
        "ok": True,
        "excel_path": str(out_path.resolve()),
        "rows": len(df),
        "datasheet_url": inp.datasheet_url or None
    }
    return meta
