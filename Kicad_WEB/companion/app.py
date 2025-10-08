# companion/app.py — KiCad 8/9 friendly, GUI-only (no kicad-cli)
from __future__ import annotations

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pathlib import Path
from uuid import uuid4
from sexpdata import Symbol, loads
from typing import Optional, Tuple, List, Dict
import os, zipfile, io, shutil, datetime, re, math, tempfile, json, time

# 3rd party: kicad-skip
from skip import Schematic  # pip install kicad-skip

app = FastAPI(title="KiCad API", version="0.2.0")
APP = app  # alias for uvicorn compatibility

# ---------------------------------------------------------------------------
# Defaults (match KiCad standard libs; you can override via request payload)
DEFAULT_FOOTPRINT = {
    "R": "Resistor_SMD:R_0603_1608Metric",
    "C": "Capacitor_SMD:C_0603_1608Metric",
    "L": "Inductor_SMD:L_0603_1608Metric",
    "D": "Diode_SMD:D_SOD-123",
    "Q": "Package_TO_SOT_SMD:SOT-23",
    "U": None,  # ICs vary; leave empty if symbol doesn't define it
}

# ---------------------------------------------------------------------------
# CORS
VITE_ORIGIN = os.environ.get("VITE_ORIGIN", "http://localhost:5174")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _to_local_path(p: str) -> Path:
    s = (p or "").strip()
    if os.name == "nt" and len(s) >= 2 and s[1] == ":":
        return Path(s).expanduser().resolve()
    return Path(s).expanduser().resolve()

# ---------------------------------------------------------------------------
# Workspace & paths
BASE_DIR = Path(__file__).parent
WS = BASE_DIR / "workspace"
CACHE = BASE_DIR / "cache"
PROJECT_ROOT = Path(os.environ.get("KICAD_PROJECT_ROOT", str(WS)))
ARTIFACTS = PROJECT_ROOT / "Artifacts"
FP_PRETTY = PROJECT_ROOT / "footprints.pretty"
BACKUPS = WS / "backups"
for d in (WS, CACHE, PROJECT_ROOT, ARTIFACTS, FP_PRETTY, BACKUPS):
    d.mkdir(parents=True, exist_ok=True)

app.mount("/Artifacts", StaticFiles(directory=str(ARTIFACTS), html=False), name="artifacts")

# ---------------------------------------------------------------------------
# KiCad lib tables (KiCad 8 and 9) — merge project+global
APPDATA = Path(os.environ.get("APPDATA", "")) / "kicad"
GLOBAL_LIB_TABLES = [
    APPDATA / "9.0" / "sym-lib-table",
    APPDATA / "8.0" / "sym-lib-table",
]
SYM_INDEX = CACHE / "sym_index.json"

_env_var_pat = re.compile(r"\$\{([^}]+)\}")
def _expand_env(s: str) -> str:
    return _env_var_pat.sub(lambda m: os.environ.get(m.group(1), m.group(0)), s)

def _read_sym_lib_table(path: Path) -> List[dict]:
    """Return [{'name': 'Device', 'uri': 'file:///.../Device.kicad_sym'}]"""
    if not path or not path.exists():
        return []
    try:
        sexp = loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return []
    out = []
    if isinstance(sexp, list) and sexp and str(sexp[0]) == "sym-lib-table":
        for node in sexp[1:]:
            if isinstance(node, list) and node and node[0] == Symbol("lib"):
                rec = {"name": None, "uri": None}
                for kv in node[1:]:
                    if isinstance(kv, list) and kv:
                        if kv[0] == Symbol("name"): rec["name"] = kv[1]
                        elif kv[0] == Symbol("uri"):  rec["uri"]  = kv[1]
                if rec["name"] and rec["uri"]:
                    out.append(rec)
    return out

def _read_all_global_sym_lib_tables() -> List[dict]:
    libs = []
    for p in GLOBAL_LIB_TABLES:
        libs += _read_sym_lib_table(p)
    return libs

def _project_sym_lib_table_for(sch_path: Path) -> Path | None:
    # project-local sym-lib-table can sit near .kicad_pro or schematic folder
    for pro in sch_path.parent.glob("*.kicad_pro"):
        p = pro.parent / "sym-lib-table"
        if p.exists():
            return p
    p2 = sch_path.parent / "sym-lib-table"
    return p2 if p2.exists() else None

def _parse_kicad_sym_symbols(sym_file: Path) -> List[dict]:
    """Return [{'name': 'R', 'desc': 'Resistor', 'keywords': 'res R'}]"""
    try:
        sexp = loads(sym_file.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return []
    out = []
    if isinstance(sexp, list) and sexp and sexp[0] == Symbol("kicad_symbol_lib"):
        for node in sexp[1:]:
            if isinstance(node, list) and node and node[0] == Symbol("symbol"):
                if len(node) >= 2 and isinstance(node[1], str):
                    full = node[1]
                    desc = ""
                    keywords = ""
                    for sub in node[2:]:
                        if isinstance(sub, list) and sub and sub[0] == Symbol("property") and len(sub) >= 3:
                            if sub[1] == "Description" and isinstance(sub[2], str):
                                desc = sub[2]
                            if sub[1] == "Keywords" and isinstance(sub[2], str):
                                keywords = sub[2]
                    out.append({"name": full.split(":")[-1], "desc": desc, "keywords": keywords})
    return out

def _all_libs_for_current(sch_path: Path) -> List[dict]:
    libs = []
    proj_tbl = _project_sym_lib_table_for(sch_path)
    if proj_tbl:
        libs += _read_sym_lib_table(proj_tbl)
    libs += _read_all_global_sym_lib_tables()
    return libs

def _build_symbol_index_for_current() -> dict:
    sch_path = _read_current()
    libs = _all_libs_for_current(sch_path)

    index = []
    for lib in libs:
        name = str(lib["name"])
        uri = _expand_env(str(lib["uri"]))
        if uri.lower().startswith("file://"):
            uri = uri[7:]
        symfile = Path(uri)
        if not symfile.exists():
            continue
        for sym in _parse_kicad_sym_symbols(symfile):
            index.append({
                "lib": name,
                "lib_id": f"{name}:{sym['name']}",
                "name": sym["name"],
                "desc": sym.get("desc", ""),
                "keywords": sym.get("keywords", "")
            })
    SYM_INDEX.parent.mkdir(parents=True, exist_ok=True)
    SYM_INDEX.write_text(
        json.dumps({"items": index, "ts": datetime.datetime.utcnow().isoformat()}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return {"items": index}

@app.post("/libs/scan")
def libs_scan():
    data = _build_symbol_index_for_current()
    return {"ok": True, "count": len(data["items"])}

@app.get("/libs/search")
def libs_search(q: str = "", limit: int = 100):
    """
    Search by:
      - full lib_id:  'Device:R'
      - or plain name: 'R', 'C', '74HC04'  (returns first N lib_ids that contain that name)
    """
    if not SYM_INDEX.exists():
        items = _build_symbol_index_for_current()["items"]
    else:
        items = json.loads(SYM_INDEX.read_text(encoding="utf-8")).get("items", [])

    qn = (q or "").strip().lower()
    if not qn:
        return items[:max(1, min(500, limit))]

    # If query contains ":", treat as lib_id-ish search; else match on name/keywords/desc too.
    if ":" in qn:
        hits = [it for it in items if qn in it["lib_id"].lower()]
    else:
        hits = [
            it for it in items
            if qn == it["name"].lower() or
               qn in it["name"].lower() or
               qn in it["keywords"].lower() or
               qn in it["desc"].lower()
        ]

    # Prefer standard KiCad libs like 'Device', 'Connector', etc. — simple boost
    def score(it):
        lib = it["lib"].lower()
        base = 0
        if it["name"].lower() == qn:
            base += 10
        if lib in ("device", "connector", "power", "regulator", "transistor", "logic"):
            base += 2
        return -base

    hits.sort(key=score)
    return hits[:max(1, min(500, limit))]

# ---------------------------------------------------------------------------
# Models
class OpenProjectReq(BaseModel):
    path: str

class MoveReq(BaseModel):
    uuid: str
    dx: float = Field(..., description="delta x in schematic units")
    dy: float = Field(..., description="delta y in schematic units")

class PlaceReq(BaseModel):
    lib_id: str                 # e.g. "Device:R" or result from /libs/search
    x: float
    y: float
    rotation: float = 0.0
    ref_prefix: str = "U"
    footprint: Optional[str] = None  # optional explicit KLC footprint name

# ---------------------------------------------------------------------------
# Current file helpers
def _read_current() -> Path:
    p = WS / "current.txt"
    if not p.exists():
        raise HTTPException(404, "No project open. POST /project/open with a .kicad_sch path")
    sch = Path(p.read_text(encoding="utf-8")).expanduser().resolve()
    if not sch.exists():
        raise HTTPException(404, f"Schematic not found: {sch}")
    return sch

def _save_sch(path: Path, schem: "Schematic"):
    try:
        if path.exists():
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            bak_name = f"{path.stem}-{ts}{path.suffix}.bak"
            shutil.copy2(path, BACKUPS / bak_name)
    except Exception as e:
        print(f"[WARN] backup failed for {path}: {e}")

    # atomic write into the same directory
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as tmpf:
        tmpname = tmpf.name
        schem.write(tmpname)
        tmpf.flush()
        os.fsync(tmpf.fileno())

    # replace then *force* a newer mtime (Windows can have coarse granularity)
    os.replace(tmpname, str(path))
    try:
        bump = time.time() + 3.0  # +3s
        os.utime(str(path), (bump, bump))
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Minimal blank schematic for auto-init
BLANK_SCH = """\
(kicad_sch
  (version 20231120)
  (generator "eeschema")
  (generator_version "8.0")
  (uuid "{root_uuid}")
  (paper "A4")
  (title_block
    (title "{title}")
    (company "")
    (rev "1")
    (comment 1 "Generated by API")
  )
  (sheet_instances (path "/" (page "1")))
)
"""

def _load_or_init_sch(path: Path) -> "Schematic":
    try:
        return Schematic(str(path))
    except Exception:
        text = BLANK_SCH.format(title=path.stem, root_uuid=str(uuid4()))
        path.write_text(text, encoding="utf-8")
        return Schematic(str(path))

# ---------------------------------------------------------------------------
# Debug
@app.get("/debug/peek")
def debug_peek():
    sch_path = _read_current()
    text = Path(sch_path).read_text(encoding="utf-8", errors="ignore")
    sym_count = text.count("\n  (symbol\n") + text.count("(symbol \n") + text.count("(symbol (")
    tail = "\n".join(text.splitlines()[-80:])
    return {
        "path": str(sch_path),
        "mtime": datetime.datetime.fromtimestamp(Path(sch_path).stat().st_mtime).isoformat(),
        "symbol_guess_count": sym_count,
        "tail": tail,
    }

# ---------------------------------------------------------------------------
# Tree helpers
def _strip_symbol_instances(sexpr):
    remove_idx = []
    for i, item in enumerate(sexpr[1:], start=1):
        if isinstance(item, list) and item and item[0] == Symbol("symbol_instances"):
            remove_idx.append(i)
    for i in reversed(remove_idx):
        sexpr.pop(i)

def _ensure_root_uuid(sexpr):
    for item in sexpr[1:]:
        if isinstance(item, list) and item and item[0] == Symbol("uuid"):
            return item[1]
    order = [Symbol("version"), Symbol("generator"), Symbol("generator_version")]
    idx = 1
    for i, item in enumerate(sexpr[1:], start=1):
        if isinstance(item, list) and item and item[0] in order:
            idx = i + 1
    ru = str(uuid4())
    sexpr.insert(idx, [Symbol("uuid"), ru])
    return ru

def _ensure_sheet_instances(sexpr):
    for item in sexpr[1:]:
        if isinstance(item, list) and item and item[0] == Symbol("sheet_instances"):
            return
    sexpr.append([Symbol("sheet_instances"), [Symbol("path"), "/", [Symbol("page"), "1"]]])

def _ensure_lib_symbols_section(sexpr):
    # (lib_symbols ...) near top
    for item in sexpr[1:]:
        if isinstance(item, list) and item and item[0] == Symbol("lib_symbols"):
            return item
    idx = 1
    for i, item in enumerate(sexpr[1:], start=1):
        if isinstance(item, list) and item and item[0] in {Symbol("paper"), Symbol("title_block")}:
            idx = i + 1
    node = [Symbol("lib_symbols")]
    sexpr.insert(idx, node)
    return node

def _lib_symbols_has(lib_syms_node, lib_id: str) -> bool:
    for it in lib_syms_node[1:]:
        if isinstance(it, list) and it and it[0] == Symbol("symbol") and len(it) >= 2 and it[1] == lib_id:
            return True
    return False

def _insert_index_after_headers(root_list):
    for i, item in enumerate(root_list[1:], start=1):
        if isinstance(item, list) and item and item[0] == Symbol("sheet_instances"):
            return i
    header_tags = {Symbol("paper"), Symbol("title_block"), Symbol("lib_symbols")}
    idx = 1
    for i, item in enumerate(root_list[1:], start=1):
        if isinstance(item, list) and item and item[0] in header_tags:
            idx = i + 1
    return idx

def _prop(name, value, x, y, rot=0.0, hide=False, size=1.27):
    eff = [Symbol("effects"), [Symbol("font"), [Symbol("size"), float(size), float(size)]]]
    if hide:
        eff.append([Symbol("hide"), Symbol("yes")])
    return [Symbol("property"), name, value, [Symbol("at"), float(x), float(y), float(rot)], eff]

def _next_ref(sexpr, prefix: str) -> str:
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$", re.I)
    maxn = 0
    def visit(node):
        nonlocal maxn
        if isinstance(node, list) and node:
            if node[0] == Symbol("symbol"):
                for it in node[1:]:
                    if isinstance(it, list) and it and it[0] == Symbol("property"):
                        if len(it) >= 3 and it[1] == "Reference" and isinstance(it[2], str):
                            m = pat.match(it[2])
                            if m:
                                maxn = max(maxn, int(m.group(1)))
            for child in node[1:]:
                visit(child)
    visit(sexpr)
    return f"{prefix}{maxn+1}"

def _count_symbols_with_prefix(tree, prefix: str) -> int:
    count = 0
    def visit(n):
        nonlocal count
        if isinstance(n, list) and n:
            if n[0] == Symbol("symbol"):
                for it in n[1:]:
                    if isinstance(it, list) and it and it[0] == Symbol("property") and len(it) >= 3 and it[1] == "Reference":
                        ref = it[2]
                        if isinstance(ref, str) and ref.upper().startswith(prefix.upper()):
                            count += 1
                        break
            for ch in n[1:]:
                visit(ch)
    visit(tree)
    return count

def _rotate_offset(dx: float, dy: float, rot_deg: float):
    a = math.radians(rot_deg % 360.0)
    ca, sa = math.cos(a), math.sin(a)
    return (dx * ca - dy * sa, dx * sa + dy * ca)

# ---------------------------------------------------------------------------
# Resolve lib symbol
def _resolve_lib_file_for_libname(sch_path: Path, libname: str) -> Path | None:
    libs = _all_libs_for_current(sch_path)
    for lib in libs:
        if str(lib["name"]) == libname:
            uri = _expand_env(str(lib["uri"]))
            if uri.lower().startswith("file://"):
                uri = uri[7:]
            p = Path(uri)
            if p.exists():
                return p
    return None

def _load_symbol_sexpr_from_kicad_sym(symfile: Path, lib_id: str):
    from sexpdata import loads, Symbol as S
    try:
        sexp = loads(symfile.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None
    if not (isinstance(sexp, list) and sexp and sexp[0] == S("kicad_symbol_lib")):
        return None
    for node in sexp[1:]:
        if isinstance(node, list) and node and node[0] == S("symbol"):
            if len(node) >= 2 and isinstance(node[1], str) and node[1] == lib_id:
                return node
    return None

def _ensure_cached_from_library(sexpr, sch_path: Path, lib_id: str) -> bool:
    lib_syms = _ensure_lib_symbols_section(sexpr)
    if _lib_symbols_has(lib_syms, lib_id):
        return True
    if ":" not in lib_id:
        return False
    libname = lib_id.split(":", 1)[0]
    symfile = _resolve_lib_file_for_libname(sch_path, libname)
    if not symfile:
        return False
    node = _load_symbol_sexpr_from_kicad_sym(symfile, lib_id)
    if not node:
        return False
    lib_syms.append(node)
    return True

# ---------------------------------------------------------------------------
# Root & project mgmt
@app.get("/")
def root():
    return {"status": "ok", "try": ["POST /project/open", "POST /libs/scan", "GET /libs/search?q=R"]}

@app.post("/project/open")
def open_project(body: OpenProjectReq):
    p = _to_local_path(body.path).resolve()
    if not p.exists():
        raise HTTPException(404, "File not found")
    (WS / "current.txt").write_text(str(p), encoding="utf-8")
    return {"ok": True, "path": str(p)}

@app.post("/project/new")
def new_project(name: str = Form(...), set_current: bool = True):
    safe = "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip() or "project"
    proj_dir = WS / "projects" / safe
    proj_dir.mkdir(parents=True, exist_ok=True)

    sch_path = proj_dir / f"{safe}.kicad_sch"
    if sch_path.exists():
        sch_path = proj_dir / f"{safe}-{uuid4().hex[:6]}.kicad_sch"

    sch_text = BLANK_SCH.format(title=name, root_uuid=str(uuid4()))
    sch_path.write_text(sch_text, encoding="utf-8")

    if set_current:
        (WS / "current.txt").write_text(str(sch_path.resolve()), encoding="utf-8")

    return {"ok": True, "project_dir": str(proj_dir.resolve()), "schematic": str(sch_path.resolve()), "set_current": set_current}

@app.post("/project/upload")
async def upload_project(file: UploadFile = File(...), set_current: bool = True):
    up_id = str(uuid4())
    dest_root = WS / "uploads" / up_id
    dest_root.mkdir(parents=True, exist_ok=True)

    filename = file.filename or "upload.bin"
    data = await file.read()

    if filename.lower().endswith(".zip"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                z.extractall(dest_root)
        except zipfile.BadZipFile:
            raise HTTPException(400, "Zip is corrupt")
        sch_candidates = list(dest_root.rglob("*.kicad_sch"))
        if not sch_candidates:
            raise HTTPException(400, "No .kicad_sch found in zip")
        sch_path = sch_candidates[0]
    elif filename.lower().endswith(".kicad_sch"):
        sch_path = dest_root / filename
        sch_path.write_bytes(data)
    else:
        raise HTTPException(400, "Only .kicad_sch or .zip supported")

    if set_current:
        (WS / "current.txt").write_text(str(sch_path.resolve()), encoding="utf-8")

    return {"ok": True, "stored_under": str(dest_root.resolve()), "schematic": str(sch_path.resolve()), "set_current": set_current}

# ---------------------------------------------------------------------------
# Symbol list
@app.get("/symbols/list")
def symbols_list():
    sch_path = _read_current()
    schem = _load_or_init_sch(sch_path)
    sexpr = schem.tree
    _strip_symbol_instances(sexpr)
    _ensure_root_uuid(sexpr)
    _ensure_sheet_instances(sexpr)
    out = []

    def get_prop(node, prop_name):
        if isinstance(node, list) and node and node[0] == Symbol("property"):
            if len(node) >= 3 and node[1] == prop_name:
                return node[2]
        return None

    def visit(node):
        if isinstance(node, list) and node:
            if node[0] == Symbol("symbol"):
                rec = {"uuid": None, "lib_id": None, "ref": None, "value": None, "at": None}
                for item in node[1:]:
                    if isinstance(item, list) and item:
                        tag = item[0]
                        if tag == Symbol("uuid"):    rec["uuid"] = item[1]
                        elif tag == Symbol("lib_id"): rec["lib_id"] = item[1]
                        elif tag == Symbol("at"):     rec["at"] = item[1:4]
                        elif tag == Symbol("property"):
                            ref = get_prop(item, "Reference")
                            val = get_prop(item, "Value")
                            if ref: rec["ref"] = ref
                            if val: rec["value"] = val
                if rec["uuid"] or rec["ref"] or rec["lib_id"]:
                    out.append(rec)
            for child in node[1:]:
                visit(child)
    visit(sexpr)
    return out

# ---------------------------------------------------------------------------
# Geometry endpoint — pin anchors + approx bbox from library symbol
def _symbol_geometry_from_node(sym_node) -> Dict:
    """Extract simple pin anchors and an approximate bbox in symbol-local coords."""
    pins = []
    minx = miny = float("inf")
    maxx = maxy = float("-inf")

    def upd(x, y):
        nonlocal minx, miny, maxx, maxy
        minx = min(minx, x); miny = min(miny, y)
        maxx = max(maxx, x); maxy = max(maxy, y)

    for sub in sym_node[2:]:
        if not (isinstance(sub, list) and sub):
            continue
        tag = sub[0]
        if tag == Symbol("pin"):
            # (pin type shape (at x y rot) (length L) (name "..." ...) (number "..." ...))
            at = None; length = 0.0
            for it in sub[1:]:
                if isinstance(it, list) and it:
                    if it[0] == Symbol("at"):
                        at = it
                    elif it[0] == Symbol("length") and len(it) >= 2:
                        try: length = float(it[1])
                        except Exception: pass
            if at and len(at) >= 4:
                x, y, rot = float(at[1]), float(at[2]), float(at[3])
                # pin endpoint from anchor along rot
                dx = math.cos(math.radians(rot)) * length
                dy = math.sin(math.radians(rot)) * length
                pins.append({"x": x, "y": y, "rot": rot, "length": length, "end": {"x": x+dx, "y": y+dy}})
                upd(x, y); upd(x+dx, y+dy)

        # very rough bbox from rectangles/lines if present
        if tag in (Symbol("rectangle"),):
            # (rectangle (start x y) (end x y) ...)
            sx = sy = ex = ey = None
            for it in sub[1:]:
                if isinstance(it, list) and it:
                    if it[0] == Symbol("start") and len(it) >= 3:
                        sx, sy = float(it[1]), float(it[2])
                    elif it[0] == Symbol("end") and len(it) >= 3:
                        ex, ey = float(it[1]), float(it[2])
            if sx is not None and ex is not None:
                upd(sx, sy); upd(ex, ey)

    if minx == float("inf"):
        minx = miny = -1.0
        maxx = maxy = 1.0

    return {"pins": pins, "bbox": {"min": [minx, miny], "max": [maxx, maxy]}}

@app.get("/libs/geom")
def libs_geom(lib_id: str):
    """Return library symbol pin anchors + approx bbox (symbol-local coords)."""
    sch_path = _read_current()
    if ":" not in lib_id:
        raise HTTPException(400, "lib_id must look like 'Lib:Name'")
    libname = lib_id.split(":", 1)[0]
    symfile = _resolve_lib_file_for_libname(sch_path, libname)
    if not symfile:
        raise HTTPException(404, f"Library '{libname}' not found in project/global tables")
    node = _load_symbol_sexpr_from_kicad_sym(symfile, lib_id)
    if not node:
        raise HTTPException(404, f"Symbol '{lib_id}' not found in {symfile.name}")
    return _symbol_geometry_from_node(node)

# ---------------------------------------------------------------------------
# Place symbol
@app.post("/schematic/place_symbol")
def place_symbol(body: PlaceReq):
    sch_path = _read_current()
    schem = _load_or_init_sch(sch_path)
    sexpr = schem.tree

    if not (isinstance(sexpr, list) and sexpr and sexpr[0] == Symbol("kicad_sch")):
        raise HTTPException(500, "Not a valid .kicad_sch root")

    # Hygiene
    _strip_symbol_instances(sexpr)
    root_uuid = _ensure_root_uuid(sexpr)
    _ensure_sheet_instances(sexpr)

    # Ensure the symbol's library definition is cached in (lib_symbols)
    if not _ensure_cached_from_library(sexpr, sch_path, body.lib_id):
        # Try to auto-resolve from plain name if user sent "R" etc.
        if ":" not in body.lib_id:
            # attempt first hit from index
            items = libs_search(q=body.lib_id, limit=1)
            if items:
                lib_id = items[0]["lib_id"]
                if not _ensure_cached_from_library(sexpr, sch_path, lib_id):
                    raise HTTPException(404, f"Library symbol not found: {body.lib_id}")
                resolved_lib_id = lib_id
            else:
                raise HTTPException(404, f"Library symbol not found: {body.lib_id}")
        else:
            raise HTTPException(404, f"Library symbol not found: {body.lib_id}")
    else:
        resolved_lib_id = body.lib_id

    # Inputs
    in_x, in_y, rot = float(body.x), float(body.y), float(body.rotation)

    # Auto-annotate
    prefix = (body.ref_prefix or "U").strip()[:3]
    ref = _next_ref(sexpr, prefix)
    val = resolved_lib_id.split(":")[-1]

    # Spread parts with same prefix so they don't stack
    n_existing = _count_symbols_with_prefix(sexpr, prefix)
    x = in_x + 5.08 * n_existing
    y = in_y

    # Footprint: explicit > default map > empty
    fp_text = body.footprint
    if not fp_text:
        key = (body.ref_prefix or "U").strip().upper()[:1]
        fp_text = DEFAULT_FOOTPRINT.get(key)

    # Text offsets (rotate with part)
    REF_OFF = (2.54, -1.270)
    VAL_OFF = (2.54,  1.270)
    rdx, rdy = _rotate_offset(*REF_OFF, rot)
    vdx, vdy = _rotate_offset(*VAL_OFF, rot)

    new_uuid = str(uuid4())

    sym = [
        Symbol("symbol"),
        [Symbol("lib_id"), resolved_lib_id],
        [Symbol("at"), x, y, rot],
        [Symbol("unit"), 1],
        [Symbol("exclude_from_sim"), Symbol("no")],
        [Symbol("in_bom"), Symbol("yes")],
        [Symbol("on_board"), Symbol("yes")],
        [Symbol("dnp"), Symbol("no")],
        [Symbol("fields_autoplaced"), Symbol("yes")],
        [Symbol("uuid"), new_uuid],

        # Reference / Value
        [Symbol("property"), "Reference", ref,
            [Symbol("at"), x + rdx, y + rdy, rot],
            [Symbol("effects"), [Symbol("font"), [Symbol("size"), 1.27, 1.27]], [Symbol("justify"), Symbol("left")]]
        ],
        [Symbol("property"), "Value", val,
            [Symbol("at"), x + vdx, y + vdy, rot],
            [Symbol("effects"), [Symbol("font"), [Symbol("size"), 1.27, 1.27]], [Symbol("justify"), Symbol("left")]]
        ],

        # Footprint actually set here:
        _prop("Footprint", fp_text or "", x - 1.778, y, 90.0, hide=True),
        _prop("Datasheet", "~",           x,           y,  0.0, hide=True),
    ]

    # Instance path (tie to root sheet)
    sym.append(
        [Symbol("instances"),
            [Symbol("project"), "",
                [Symbol("path"), f"/{root_uuid}",
                    [Symbol("reference"), ref],
                    [Symbol("unit"), 1]
                ]
            ]
        ]
    )

    # Insert before (sheet_instances)
    insert_at = _insert_index_after_headers(sexpr)
    sexpr.insert(insert_at, sym)

    _save_sch(sch_path, schem)
    return {
        "ok": True,
        "uuid": new_uuid,
        "ref": ref,
        "lib_id": resolved_lib_id,
        "x": x, "y": y, "rotation": rot,
        "footprint": fp_text,
    }

# ---------------------------------------------------------------------------
# Move symbol
@app.post("/schematic/move_symbol")
def move_symbol(body: MoveReq):
    sch_path = _read_current()
    schem = _load_or_init_sch(sch_path)
    sexpr = schem.tree

    def is_sym(node):
        return isinstance(node, list) and node and node[0] == Symbol("symbol")

    def find(tag, node):
        for i, item in enumerate(node):
            if isinstance(item, list) and item and item[0] == Symbol(tag):
                return i, item
        return None, None

    moved = False
    def move_in(node):
        nonlocal moved
        if is_sym(node):
            _, uuid_node = find("uuid", node)
            if uuid_node and uuid_node[1] == body.uuid:
                at_i, at_node = find("at", node)
                if at_node:
                    x0, y0 = float(at_node[1]), float(at_node[2])
                    a = at_node[3] if len(at_node) > 3 else None
                    node[at_i] = [Symbol("at"), x0 + body.dx, y0 + body.dy] + ([a] if a is not None else [])
                    moved = True
                    return
        for child in (node[1:] if isinstance(node, list) else []):
            if isinstance(child, list):
                move_in(child)

    move_in(sexpr)
    if not moved:
        raise HTTPException(404, "uuid not found")

    _save_sch(sch_path, schem)
    return {"ok": True}

# ---------------------------------------------------------------------------
# Status
@app.get("/api/status")
def status():
    return {
        "ok": True,
        "vite_origin": VITE_ORIGIN,
        "project_root": str(PROJECT_ROOT.resolve()),
        "artifacts": str(ARTIFACTS.resolve()),
        "footprints_pretty": str(FP_PRETTY.resolve()),
        "workspace": str(WS.resolve()),
    }

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("companion.app:APP", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
