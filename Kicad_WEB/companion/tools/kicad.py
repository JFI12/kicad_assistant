# companion/tools/kicad.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import os
import re

router = APIRouter(tags=["tools"])

# ---------------------------------------------------------------------------
# Workspace / project paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # companion/
PROJECT_ROOT = Path(os.environ.get("KICAD_PROJECT_ROOT", str(BASE_DIR / "workspace")))
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

ARTIFACTS = PROJECT_ROOT / "Artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

FP_PRETTY = PROJECT_ROOT / "footprints.pretty"
FP_PRETTY.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class AddSymInput(BaseModel):
    symbol_kicad_sym: str
    library_name: str = "ProjectSymbols"

class AddFptInput(BaseModel):
    footprint_kicad_mod: str
    footprint_name: str

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_symbol_name(kicad_sym_text: str) -> str:
    """Naive parse: find (symbol "Name") and return Name."""
    m = re.search(r'\(symbol\s+"([^"]+)"', kicad_sym_text)
    return m.group(1).split(":")[-1] if m else "Unknown"

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/tools/add_symbol_to_project",
    operation_id="add_symbol_to_project_tool"
)
async def add_symbol_to_project(inp: AddSymInput):
    """Append a symbol into the project .kicad_sym and update sym-lib-table."""
    lib = PROJECT_ROOT / f"{inp.library_name}.kicad_sym"

    # Ensure file exists with proper header
    if not lib.exists():
        lib.write_text('(kicad_symbol_lib (version 20211014) (generator "companion"))\n', encoding="utf-8")

    # Append the symbol definition
    with lib.open("a", encoding="utf-8") as f:
        f.write("\n")
        f.write(inp.symbol_kicad_sym.strip())
        f.write("\n")

    # Ensure sym-lib-table includes this library
    sym_table = PROJECT_ROOT / "sym-lib-table"
    entry = f'(lib (name {inp.library_name})(type Legacy)(uri {lib.as_posix()})(options "")(descr ""))\n'

    if sym_table.exists():
        txt = sym_table.read_text(encoding="utf-8")
        if entry not in txt:
            sym_table.write_text(txt.rstrip() + "\n" + entry, encoding="utf-8")
    else:
        sym_table.write_text(f'(sym_lib_table\n  {entry})\n', encoding="utf-8")

    symbol_name = extract_symbol_name(inp.symbol_kicad_sym)
    return {
        "ok": True,
        "library_ref": f"{inp.library_name}:{symbol_name}",
        "file": str(lib.resolve())
    }



@router.post(
    "/tools/add_footprint_to_project",
    operation_id="add_footprint_to_project_tool"
)
async def add_footprint_to_project(inp: AddFptInput):
    """Write a footprint .kicad_mod into footprints.pretty and update fp-lib-table."""
    mod_path = FP_PRETTY / f"{inp.footprint_name}.kicad_mod"
    mod_path.write_text(inp.footprint_kicad_mod, encoding="utf-8")

    fp_table = PROJECT_ROOT / "fp-lib-table"
    entry = f'(lib (name ProjectFootprints)(type KiCad)\n  (uri {FP_PRETTY.as_posix()})(options "")(descr ""))\n'

    if fp_table.exists():
        txt = fp_table.read_text(encoding="utf-8")
        if "ProjectFootprints" not in txt:
            fp_table.write_text(txt.rstrip() + "\n" + entry, encoding="utf-8")
    else:
        fp_table.write_text(f'(fp_lib_table\n  {entry})\n', encoding="utf-8")

    return {
        "ok": True,
        "footprint_lib": "ProjectFootprints",
        "footprint_path": str(mod_path.resolve())
    }
