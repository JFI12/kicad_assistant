# companion/tools/schematic.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, Tuple
import os, uuid
from sexpdata import loads, dumps, Symbol

router = APIRouter(tags=["schematic-tools"])

# Samma workspace-upplägg som i app.py
BASE_DIR = Path(__file__).resolve().parents[1]  # companion/
PROJECT_ROOT = Path(os.environ.get("KICAD_PROJECT_ROOT", str(BASE_DIR / "workspace")))
WS = PROJECT_ROOT  # där vi också lagt current.txt i app.py

def _read_current_sch() -> Path:
    p = (BASE_DIR / "workspace" / "current.txt") if (BASE_DIR / "workspace" / "current.txt").exists() \
        else (WS / "current.txt")
    if not p.exists():
        raise HTTPException(404, "No project open. POST /project/open with a .kicad_sch path")
    sch = Path(p.read_text(encoding="utf-8"))
    if not sch.exists():
        raise HTTPException(404, f"Schematic not found: {sch}")
    return sch

def _load_sch(path: Path):
    return loads(path.read_text(encoding="utf-8"))

def _save_sch(path: Path, sexpr):
    path.write_text(dumps(sexpr), encoding="utf-8")

def _ensure_property_list(node, name: str, value: str):
    """Ensure (property "name" "value") exists/updated on a symbol node."""
    if not (isinstance(node, list) and node and node[0] == Symbol("symbol")):
        return
    # find existing property
    for i, child in enumerate(node[1:]):
        if isinstance(child, list) and child and child[0] == Symbol("property"):
            if len(child) >= 3 and child[1] == name:
                # update value
                new = [Symbol("property"), name, value]
                node[i+1] = new
                return
    # add fresh
    node.append([Symbol("property"), name, value])

def _make_symbol_node(lib_id: str, at_xy: Tuple[float,float], ref_prefix: str, value_text: Optional[str]) -> list:
    """Create a minimal KiCad 8 symbol instance node."""
    x, y = at_xy
    ref = f"{ref_prefix}?"  # låt KiCad annotera senare
    symbol_uuid = str(uuid.uuid4())

    symbol = [Symbol("symbol")]
    symbol.append([Symbol("lib_id"), lib_id])
    symbol.append([Symbol("at"), float(x), float(y)])
    symbol.append([Symbol("uuid"), symbol_uuid])

    # Required KiCad properties
    symbol.append([Symbol("property"), "Reference", ref])
    symbol.append([Symbol("property"), "Value", value_text or lib_id.split(":")[-1]])

    # Minimal grafik (en instans behöver inte rectangle här – KiCad ritar från lib)
    return symbol

class AddSymbolReq(BaseModel):
    lib_id: str = Field(..., description='Ex. "ProjectSymbols:STM32F407VGT6"')
    x: float = 0
    y: float = 0
    ref_prefix: str = "U"
    value: Optional[str] = None

@router.post("/schematic/add_symbol_instance", operation_id="schematic_add_symbol_instance")
def schematic_add_symbol_instance(req: AddSymbolReq):
    sch_path = _read_current_sch()
    root = _load_sch(sch_path)

    # KiCad .kicad_sch top-level: (kicad_sch (version ...) (generator ...) (paper ...) (symbol ...) (symbol ...) ...)
    if not (isinstance(root, list) and root and root[0] == Symbol("kicad_sch")):
        raise HTTPException(400, "Invalid schematic S-expression")

    sym_node = _make_symbol_node(req.lib_id, (req.x, req.y), req.ref_prefix, req.value)
    root.append(sym_node)

    _save_sch(sch_path, root)

    # Returnera UUID + ref som sattes
    ref = None
    for child in sym_node:
        if isinstance(child, list) and child and child[0] == Symbol("property"):
            if len(child) >= 3 and child[1] == "Reference":
                ref = child[2]
    uid = None
    for child in sym_node:
        if isinstance(child, list) and child and child[0] == Symbol("uuid"):
            uid = child[1]

    return {"ok": True, "uuid": uid, "reference": ref, "lib_id": req.lib_id}

class SetFootprintReq(BaseModel):
    uuid: str
    footprint_lib: str = "ProjectFootprints"   # libnamn i fp-lib-table
    footprint_name: str = Field(..., description="Ex. STM32F407VGT6_LQFP")

@router.post("/schematic/set_footprint_property", operation_id="schematic_set_footprint_property")
def schematic_set_footprint_property(req: SetFootprintReq):
    sch_path = _read_current_sch()
    root = _load_sch(sch_path)

    found = False
    def visit(node):
        nonlocal found
        if isinstance(node, list) and node and node[0] == Symbol("symbol"):
            # hitta uuid
            for child in node:
                if isinstance(child, list) and child and child[0] == Symbol("uuid"):
                    if child[1] == req.uuid:
                        # sätt (property "Footprint" "Lib:Name")
                        fp_value = f"{req.footprint_lib}:{req.footprint_name}"
                        _ensure_property_list(node, "Footprint", fp_value)
                        found = True
                        return
        # gå neråt
        if isinstance(node, list):
            for child in node[1:]:
                visit(child)

    visit(root)
    if not found:
        raise HTTPException(404, f"Symbol uuid {req.uuid} not found")

    _save_sch(sch_path, root)
    return {"ok": True}

# --- Extras för smidigt flöde -----------------------------------------------
from typing import Dict
import re

@router.post("/schematic/annotate_references", operation_id="schematic_annotate_references")
def schematic_annotate_references():
    """
    Ersätter U?/R?/C?/L?/D?/Q?/Y?/J?/TP? med U1, R1, C1, ... i aktuellt schema.
    Enkel sekventiell annotering per prefix.
    """
    sch_path = _read_current_sch()
    root = _load_sch(sch_path)

    counters: Dict[str, int] = {"U":0,"R":0,"C":0,"L":0,"D":0,"Q":0,"Y":0,"J":0,"TP":0}
    ref_re = re.compile(r"^(U|R|C|L|D|Q|Y|J|TP)\?$", re.IGNORECASE)

    def try_annotate(node):
        if isinstance(node, list) and node and node[0] == Symbol("symbol"):
            for i, child in enumerate(node):
                if isinstance(child, list) and child and child[0] == Symbol("property"):
                    if len(child) >= 3 and child[1] == "Reference":
                        ref_val = child[2]
                        m = ref_re.match(str(ref_val))
                        if m:
                            prefix = m.group(1).upper()
                            counters[prefix] = counters.get(prefix, 0) + 1
                            node[i] = [Symbol("property"), "Reference", f"{prefix}{counters[prefix]}"]
        if isinstance(node, list):
            for ch in node[1:]:
                try_annotate(ch)

    try_annotate(root)
    _save_sch(sch_path, root)
    return {"ok": True, "assigned": counters}


class InsertWithFootprintReq(BaseModel):
    lib_id: str = Field(..., description='Ex. "ProjectSymbols:STM32F407VGT6"')
    x: float = 0
    y: float = 0
    ref_prefix: str = "U"
    value: Optional[str] = None
    footprint_lib: str = "ProjectFootprints"
    footprint_name: str = Field(..., description="Ex. STM32F407VGT6_LQFP")
    annotate: bool = True  # kör annotate_references i slutet

@router.post("/schematic/insert_with_footprint", operation_id="schematic_insert_with_footprint")
def schematic_insert_with_footprint(req: InsertWithFootprintReq):
    """
    Infogar symbol i schemat och sätter Footprint i ett enda anrop.
    (Optionellt kör annotering.)
    """
    # 1) lägg in symbolen
    add_res = schematic_add_symbol_instance(AddSymbolReq(
        lib_id=req.lib_id,
        x=req.x,
        y=req.y,
        ref_prefix=req.ref_prefix,
        value=req.value
    ))
    uid = add_res["uuid"]

    # 2) sätt footprint-property
    set_res = schematic_set_footprint_property(SetFootprintReq(
        uuid=uid,
        footprint_lib=req.footprint_lib,
        footprint_name=req.footprint_name
    ))

    # 3) (valfritt) annotera
    assigned = None
    if req.annotate:
        ann = schematic_annotate_references()
        assigned = ann.get("assigned")

    return {
        "ok": True,
        "uuid": uid,
        "reference": add_res["reference"],
        "lib_id": req.lib_id,
        "footprint": f"{req.footprint_lib}:{req.footprint_name}",
        "annotate_counts": assigned
    }
