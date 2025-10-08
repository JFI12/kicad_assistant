# pcb_companion_api.py — FastAPI backend for PCB auto-placement planning (KiCad 9–friendly)
# Goals
# - Accept a natural-language prompt + optional rule toggles
# - Inspect a .kicad_pcb (or live board info sent from the plugin)
# - Produce a placement PLAN: [{ref, x_mm, y_mm, rot_deg, side}, ...]
# - Encode practical defaults: antenna near edge, PSU near edge, decouplers near VCC pins,
#   spacing corridors between highly connected ICs, and minimize weighted interconnect distances.
#
# Notes
# - This service does NOT write the .kicad_pcb. The KiCad action plugin applies the plan.
# - Dependencies purposely light; simple geo + connectivity heuristics only.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Set
from pathlib import Path
import math, json, re, os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    import pcbnew  # KiCad Python (optional on server)
except Exception:
    pcbnew = None  # type: ignore


# ----------------------------------------------------------------------------
# App
app = FastAPI(title="KiCad API", version="0.2.4")
APP = app  # alias for uvicorn: python -m uvicorn companion.app2:APP --reload


# ----------------------------------------------------------------------------
# LLM system instructions (single source of truth)
SYSTEM_INSTRUCTIONS = """You are a KiCad PCB placement assistant.
From the user's instruction and a board snapshot (outline bbox + footprints with refs/positions),
produce a placement PLAN.

Output strictly a JSON object:
{
  "text": "short explanation of your choices",
  "plan": [
    {"ref":"U1","x_mm":42.0,"y_mm":17.5,"rot_deg":0,"side":"top"},
    ...
  ]
}

Rules:
- Only move components that exist in the snapshot.
- Units: millimeters (x_mm, y_mm) and degrees (rot_deg). side is "top" or "bottom".
- Keep decouplers ~3 mm from the target IC VCC pins on the SAME side.
- If asked "power south edge", anchor PSU cluster near that edge.
- If asked "antenna top-right", anchor antenna/RF there with a small spacing.
- If no moves are needed, return "plan": [].
"""


# ----------------------------------------------------------------------------
# Models
class OpenBoardReq(BaseModel):
    path: str


class FootprintPad(BaseModel):
    pad: str
    net: str
    x_mm: float
    y_mm: float


class FPInfo(BaseModel):
    ref: str
    value: Optional[str] = None
    footprint: Optional[str] = None
    x_mm: float
    y_mm: float
    rot_deg: float = 0.0
    side: str = "top"  # "top" or "bottom"
    pads: List[FootprintPad] = Field(default_factory=list)


class BoardSnapshot(BaseModel):
    width_mm: float
    height_mm: float
    outline_bbox: Tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
    footprints: List[FPInfo]


class PlanRuleToggles(BaseModel):
    antenna_region: str = "top-right"  # top-left/top-right/bottom-left/bottom-right/auto
    psu_edge: str = "south"            # north/south/east/west/auto
    decap_radius_mm: float = 3.0       # place decaps within this of target pins
    channel_track_mm: float = 0.25     # nominal track width for channel spacing planning
    channel_clear_mm: float = 0.25     # clearance for channel spacing


class PlanRequest(BaseModel):
    prompt: str = Field("", description="free text instructions from chat")
    rules: PlanRuleToggles = Field(default_factory=PlanRuleToggles)
    snapshot: Optional[BoardSnapshot] = None  # if None and pcbnew available, read live board


class PlanItem(BaseModel):
    ref: str
    x_mm: float
    y_mm: float
    rot_deg: float
    side: str


class PlanResponse(BaseModel):
    ok: bool
    items: List[PlanItem]
    notes: List[str] = []


# ----------------------------------------------------------------------------
# LLM (Gemini) — REST helper (kept optional; server runs fine without key)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
)

def _call_gemini_rest(user_text: str, snapshot: dict) -> Optional[dict]:
    if not GEMINI_API_KEY:
        return None
    try:
        body = {
            "system_instruction": {"role": "system", "parts": [{"text": SYSTEM_INSTRUCTIONS}]},
            "contents": [{
                "role": "user",
                "parts": [{
                    "text": f"USER INSTRUCTION:\n{user_text}\n\nBOARD SNAPSHOT (JSON):\n{json.dumps(snapshot)}"
                }]
            }],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048}
        }
        data = json.dumps(body).encode("utf-8")
        req = __import__("urllib.request").request.Request(
            GEMINI_ENDPOINT + f"?key={GEMINI_API_KEY}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with __import__("urllib.request").request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read().decode("utf-8"))

        # Extract raw text
        txt_parts: List[str] = []
        for cand in resp.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if "text" in part:
                    txt_parts.append(part["text"])
        txt = "\n".join(txt_parts).strip()

        # Extract JSON object from text (first {...} block)
        m = re.search(r"\{[\s\S]*\}", txt)
        obj = json.loads(m.group(0)) if m else {"text": txt, "plan": []}
        if "text" not in obj:
            obj["text"] = txt[:500]

        # Sanitize plan
        plan = []
        for it in obj.get("plan", []):
            plan.append({
                "ref": str(it.get("ref", "")),
                "x_mm": float(it.get("x_mm", 0.0)),
                "y_mm": float(it.get("y_mm", 0.0)),
                "rot_deg": float(it.get("rot_deg", 0.0)),
                "side": it.get("side", "top"),
            })
        obj["plan"] = plan
        return obj
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Parsing helpers for prompts / names
_name_tokens = re.compile(r"[A-Za-z0-9_+-]+")

def _which_region_from_text(p: str) -> Optional[str]:
    p = p.lower()
    # accept "edge" phrasing too
    if ("south" in p or "bottom" in p): return "south"
    if ("north" in p or "top"    in p): return "north"
    if ("east"  in p or "right"  in p): return "east"
    if ("west"  in p or "left"   in p): return "west"
    if "top-right" in p or "north-east" in p or re.search(r"\bne\b", p): return "top-right"
    if "top-left"  in p or "north-west" in p or re.search(r"\bnw\b", p): return "top-left"
    if "bottom-right" in p or "south-east" in p or re.search(r"\bse\b", p): return "bottom-right"
    if "bottom-left"  in p or "south-west" in p or re.search(r"\bsw\b", p): return "bottom-left"
    return None

def _has_slight_nudge(p: str) -> Optional[Tuple[float, float]]:
    """
    Detect "slightly left/right/up/down" and return (dx, dy) in mm.
    Only triggers if the word 'slightly' (or 'a bit') appears.
    Defaults to 2.0 mm magnitude.
    """
    s = p.lower()
    if not (("slightly" in s) or ("a bit" in s) or ("a little" in s)):
        return None
    mag = 2.0
    dx = dy = 0.0
    if "left" in s:  dx -= mag
    if "right" in s: dx += mag
    if "up" in s or "north" in s or "top" in s:    dy -= mag  # consistent with this file's semantics
    if "down" in s or "south" in s or "bottom" in s: dy += mag
    if dx == 0.0 and dy == 0.0:
        return None
    return (dx, dy)

def _matches_part(fp: FPInfo, token: str) -> bool:
    t = token.lower()
    if not t: return False
    if fp.ref.lower() == t: return True
    if fp.ref.lower().startswith(t): return True  # e.g. "u3" or "u"
    if (fp.value or "").lower().find(t) >= 0: return True
    if (fp.footprint or "").lower().find(t) >= 0: return True
    return False


# ----------------------------------------------------------------------------
# Safe outline (prevents degenerate clamping)
def _safe_outline_from(b: BoardSnapshot) -> Tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = b.outline_bbox
    if (xmax - xmin) >= 4.0 and (ymax - ymin) >= 4.0:
        return xmin, ymin, xmax, ymax
    # Fallback: compute from footprints
    if not b.footprints:
        # final fallback: a small default board
        return (0.0, 0.0, 100.0, 80.0)
    xs = [fp.x_mm for fp in b.footprints]
    ys = [fp.y_mm for fp in b.footprints]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    pad = 5.0  # mm margin around parts
    return (minx - pad, miny - pad, maxx + pad, maxy + pad)


# ----------------------------------------------------------------------------
# Board I/O (optional if pcbnew present on the server)
_CURRENT_BOARD: Optional[Path] = None

def _from_nm(nm: int) -> float:
    return float(nm) / 1e6

def _to_snapshot_from_pcbnew() -> BoardSnapshot:
    if pcbnew is None:
        raise HTTPException(400, "pcbnew not available on server; send snapshot from plugin")
    brd = pcbnew.GetBoard()
    if brd is None:
        raise HTTPException(404, "No board open on server")
    bbox = brd.GetBoardEdgesBoundingBox()
    xmin, ymin = _from_nm(bbox.GetX()), _from_nm(bbox.GetY())
    width, height = _from_nm(bbox.GetWidth()), _from_nm(bbox.GetHeight())
    xmax, ymax = xmin + width, ymin + height

    fps: List[FPInfo] = []
    for fp in brd.GetFootprints():
        pos = fp.GetPosition()
        side = "bottom" if fp.IsFlipped() else "top"
        pads: List[FootprintPad] = []
        for pad in fp.Pads():
            netname = pad.GetNet().GetNetname() if pad.GetNet() else ""
            p = pad.GetPosition()
            pads.append(FootprintPad(
                pad=str(pad.GetName()),
                net=str(netname),
                x_mm=_from_nm(p.x),
                y_mm=_from_nm(p.y),
            ))
        # footprint name can be a KiCad type — coerce to str
        fp_name = None
        try:
            fp_name = str(fp.GetFPID().GetLibItemName())
        except Exception:
            try:
                fp_name = str(fp.GetFPID())
            except Exception:
                fp_name = None

        rot_deg = 0.0
        try:
            rot_deg = float(fp.GetOrientationDegrees())
        except Exception:
            try:
                rot_deg = float(fp.GetOrientation() / 10.0)
            except Exception:
                rot_deg = 0.0

        fps.append(FPInfo(
            ref=str(fp.GetReference()),
            value=(None if fp.GetValue() in (None, "") else str(fp.GetValue())),
            footprint=fp_name,
            x_mm=_from_nm(pos.x),
            y_mm=_from_nm(pos.y),
            rot_deg=rot_deg,
            side=side,
            pads=pads,
        ))

    return BoardSnapshot(
        width_mm=width,
        height_mm=height,
        outline_bbox=(xmin, ymin, xmax, ymax),
        footprints=fps
    )

@app.post("/board/open")
def board_open(req: OpenBoardReq):
    global _CURRENT_BOARD
    path = Path(req.path).expanduser().resolve()
    if not path.exists():
        raise HTTPException(404, f"Board not found: {path}")
    _CURRENT_BOARD = path
    return {"ok": True, "path": str(path)}

@app.get("/board/snapshot", response_model=BoardSnapshot)
def board_snapshot():
    return _to_snapshot_from_pcbnew()


# ----------------------------------------------------------------------------
# Heuristics / utilities
VCC_NAMES = {"VCC", "+3V3", "+5V", "+1V8", "+2V5", "+12V", "VBAT", "VIN", "VDD", "+3.3V", "+5V0"}
GND_NAMES = {"GND", "PGND", "AGND"}
_decap_val_pat = re.compile(r"\b(\d+(?:\.\d+)?\s*(nF|uF|µF|pF))\b", re.I)

# placer.py
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math, random

Vec = Tuple[float, float]
BBox = Tuple[float, float, float, float]

@dataclass
class Part:
    ref: str
    w: float
    h: float
    pins: Dict[str, Vec]          # pad -> (x_local, y_local) in mm
    x: float
    y: float
    rot: float                    # deg
    side: str                     # "top"/"bottom"
    fixed: bool = False
    cls: Optional[str] = None     # from policy

@dataclass
class Net:
    name: str
    pins: List[Tuple[str, str]]   # list of (ref, pad)
    weight: float

@dataclass
class Board:
    xmin: float; ymin: float; xmax: float; ymax: float
    keepouts: List[BBox]

GRID = 0.5

def clamp(v, a, b): return max(a, min(b, v))

def rotate(x, y, deg):
    a = math.radians(deg % 360.0); ca, sa = math.cos(a), math.sin(a)
    return (x*ca - y*sa, x*sa + y*ca)

def part_bbox_world(p: Part) -> BBox:
    # conservative axis-aligned bbox of rotated rectangle
    hw, hh = p.w/2, p.h/2
    corners = [rotate(dx,dy,p.rot) for dx in (+hw,-hw,+hw,-hw) for dy in (+hh,+hh,-hh,-hh)]
    xs = [p.x + cx for cx,cy in corners]; ys = [p.y + cy for cx,cy in corners]
    return (min(xs), min(ys), max(xs), max(ys))

def overlap(a:BBox, b:BBox, halo:float=0.2)->bool:
    return not (a[2]+-halo <= b[0] or b[2]+-halo <= a[0] or a[3]+-halo <= b[1] or b[3]+-halo <= a[1])

def legal_position(parts: Dict[str,Part], ref: str, board: Board, pos: Vec, rot: float) -> Optional[Tuple[float,float,float]]:
    p = parts[ref]; old = (p.x, p.y, p.rot)
    p.x, p.y, p.rot = round(pos[0]/GRID)*GRID, round(pos[1]/GRID)*GRID, rot
    bb = part_bbox_world(p)
    if bb[0] < board.xmin or bb[1] < board.ymin or bb[2] > board.xmax or bb[3] > board.ymax:
        p.x, p.y, p.rot = old; return None
    for ko in board.keepouts:
        if overlap(bb, ko, halo=0.0): p.x, p.y, p.rot = old; return None
    for q in parts.values():
        if q.ref == ref: continue
        if overlap(bb, part_bbox_world(q), halo=0.3): p.x, p.y, p.rot = old; return None
    return (p.x, p.y, p.rot)

def net_hpwl(parts: Dict[str,Part], net: Net) -> float:
    xs, ys = [], []
    for ref, pad in net.pins:
        p = parts.get(ref); 
        if not p: continue
        px, py = p.x, p.y
        # approximate pin location by part center (fast) — optional: add rotated pin offsets
        xs.append(px); ys.append(py)
    if not xs: return 0.0
    return (max(xs)-min(xs)) + (max(ys)-min(ys))

def cost(parts: Dict[str,Part], nets: List[Net], board: Board) -> float:
    c = 0.0
    for n in nets:
        hp = net_hpwl(parts, n)
        c += n.weight * hp
        if len(n.pins) == 2:   # tighten 2-pin nets
            (r1,_),(r2,_) = n.pins
            p1, p2 = parts[r1], parts[r2]
            d = math.hypot(p1.x-p2.x, p1.y-p2.y)
            c += 0.4 * n.weight * d
    # keepout soft penalty
    for p in parts.values():
        bb = part_bbox_world(p)
        for ko in board.keepouts:
            if overlap(bb, ko, halo=0.0):
                # area overlap proxy
                c += 50.0
    return c

def anneal(parts: Dict[str,Part], nets: List[Net], board: Board, iters=8000):
    # initial temp from random probes
    base = cost(parts, nets, board)
    probes = []
    movables = [p for p in parts.values() if not p.fixed]
    for _ in range(min(50, len(movables))):
        p = random.choice(movables)
        dx, dy = (random.uniform(-3,3), random.uniform(-3,3))
        old = (p.x,p.y,p.rot); trial = legal_position(parts,p.ref,board,(p.x+dx,p.y+dy),p.rot) 
        if trial: probes.append(abs(cost(parts,nets,board)-base))
        p.x,p.y,p.rot = old
    T = (sorted(probes)[int(0.95*len(probes))] if probes else 10.0) or 10.0

    bestC, bestPose = base, {r:(p.x,p.y,p.rot) for r,p in parts.items()}
    for k in range(iters):
        p = random.choice(movables)
        move = random.random()
        if move < 0.6:
            dx, dy = (random.uniform(-1.5,1.5), random.uniform(-1.5,1.5))
            trial = legal_position(parts, p.ref, board, (p.x+dx, p.y+dy), p.rot)
        elif move < 0.85:
            rot = (p.rot + random.choice([90, -90])) % 360
            trial = legal_position(parts, p.ref, board, (p.x, p.y), rot)
        else:
            # small swap/nudge within same class/cluster (if available)
            same = [q for q in movables if q.cls == p.cls and q.ref != p.ref]
            if not same: continue
            q = random.choice(same)
            p.x, p.y, q.x, q.y = q.x, q.y, p.x, p.y
            if any(legal_position(parts, x.ref, board, (x.x, x.y), x.rot) is None for x in (p,q)):
                p.x, p.y, q.x, q.y = q.x, q.y, p.x, p.y; continue
            trial = (p.x,p.y,p.rot)

        if not trial: continue
        newC = cost(parts, nets, board)
        dC = newC - base
        if dC <= 0 or math.exp(-dC/max(T,1e-6)) > random.random():
            base = newC
            if newC < bestC:
                bestC, bestPose = newC, {r:(pt.x,pt.y,pt.rot) for r,pt in parts.items()}
        else:
            # revert
            p.x, p.y, p.rot = trial[0], trial[1], trial[2]  # trial already applied; revert using bestPose snapshot for this ref
            px,py,pr = bestPose[p.ref]
            legal_position(parts, p.ref, board, (px,py), pr)

        if (k+1) % 100 == 0: T *= 0.95

    # commit best
    for r,(x,y,rot) in bestPose.items():
        parts[r].x, parts[r].y, parts[r].rot = x,y,rot
    return bestC



def _is_antenna_part(fp: FPInfo) -> bool:
    ref_up = (fp.ref or "").upper()
    val = (fp.value or "").lower()
    fpname = (fp.footprint or "").lower()
    if ref_up.startswith("AE") or ref_up.startswith("ANT"):
        return True
    if any(k in val for k in ["antenna", "ceramic antenna", "chip antenna"]):
        return True
    if any(k in fpname for k in ["antenna", "ipex", "u.fl", "sma_edge", "rf_connector"]):
        return True
    return False

def _is_psu_controller(fp: FPInfo) -> bool:
    val = (fp.value or "").lower()
    # Most PSU controllers are U*, but we narrow by keywords
    if fp.ref.startswith("U") and any(k in val for k in ["buck", "boost", "reg", "dcdc", "ldo", "switching regulator"]):
        return True
    return False

def _is_power_inductor(fp: FPInfo) -> bool:
    return fp.ref.startswith("L")

def _is_rectifier(fp: FPInfo) -> bool:
    return fp.ref.startswith("D")

def _in_power_island(fp: FPInfo) -> bool:
    nets = {p.net.upper() for p in fp.pads if p.net}
    return any(n in nets for n in VCC_NAMES) or any(n.startswith("+") for n in nets)

def _is_decap(fp: FPInfo) -> bool:
    if not fp.ref.startswith("C"):
        return False
    if (fp.value or "") and _decap_val_pat.search(fp.value or ""):
        return True
    # also treat common shorthand as decap
    return (fp.value or "").lower() in {"100n", "0.1u", "0.1µ"}

def _net_between(a: FPInfo, b: FPInfo) -> int:
    nets_a = {p.net for p in a.pads if p.net}
    nets_b = {p.net for p in b.pads if p.net}
    shared = nets_a.intersection(nets_b)
    shared_no_gnd = {n for n in shared if n.upper() not in GND_NAMES}
    return max(0, len(shared_no_gnd))

def _closest_pin_xy(target_net: str, fp: FPInfo) -> Tuple[float, float]:
    for p in fp.pads:
        if p.net == target_net:
            return (p.x_mm, p.y_mm)
    return (fp.x_mm, fp.y_mm)

def _edge_anchor(b: BoardSnapshot, where: str) -> Tuple[float, float]:
    xmin, ymin, xmax, ymax = _safe_outline_from(b)
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    pad = 3.0  # mm inset
    w = (where or "").lower()
    if w in ("north", "top"):
        return (cx, ymin + pad)
    if w in ("south", "bottom"):
        return (cx, ymax - pad)
    if w == "east":
        return (xmax - pad, cy)
    if w == "west":
        return (xmin + pad, cy)
    if w in ("top-right", "north-east", "ne"):
        return (xmax - pad, ymin + pad)
    if w in ("top-left", "north-west", "nw"):
        return (xmin + pad, ymin + pad)
    if w in ("bottom-right", "south-east", "se"):
        return (xmax - pad, ymax - pad)
    if w in ("bottom-left", "south-west", "sw"):
        return (xmin + pad, ymax - pad)
    return (cx, cy)


# ----------------------------------------------------------------------------
# Force layout (unchanged) — but we’ll skip it when we have targeted moves
def _channel_clearance_mm(n_traces: int, w: float, c: float) -> float:
    return n_traces * (w + c) + c

def _weighted_centroid_layout(b: BoardSnapshot, fps: List[FPInfo]) -> Dict[str, Tuple[float, float]]:
    xmin, ymin, xmax, ymax = _safe_outline_from(b)

    pts = {fp.ref: (fp.x_mm, fp.y_mm) for fp in fps}
    for _ in range(12):
        new_pts: Dict[str, Tuple[float, float]] = {}
        for a in fps:
            ax, ay = pts[a.ref]
            nx = ny = wsum = 0.0
            for bfp in fps:
                if bfp.ref == a.ref:
                    continue
                w = _net_between(a, bfp)
                if w <= 0:
                    continue
                bx, by = pts[bfp.ref]
                nx += bx * w
                ny += by * w
                wsum += w
            if wsum > 0:
                tx, ty = nx / wsum, ny / wsum
                mx = ax + 0.3 * (tx - ax)
                my = ay + 0.3 * (ty - ay)
            else:
                mx, my = ax, ay
            inset = 1.0 if (xmax - xmin) >= 2.0 and (ymax - ymin) >= 2.0 else 0.0
            mx = max(xmin + inset, min(xmax - inset, mx))
            my = max(ymin + inset, min(ymax - inset, my))
            new_pts[a.ref] = (mx, my)
        pts = new_pts
    return pts


# ----------------------------------------------------------------------------
# Edge distribution + named-part selection/anchoring
def _distribute_along_region(
    b: BoardSnapshot,
    region: str,
    n: int,
    edge_gap_mm: float = 3.0,
    end_margin_mm: float = 3.0,
) -> List[Tuple[float, float]]:
    """
    Return n (x,y) points along the requested region:
      - west/east: points spread vertically (y) with fixed x near edge
      - north/south: points spread horizontally (x) with fixed y near edge
      - corners: stack points away from the corner along the longer axis
    """
    xmin, ymin, xmax, ymax = _safe_outline_from(b)
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    region = (region or "").lower().strip()

    # Clamp in-board usable span
    x_left  = xmin + edge_gap_mm
    x_right = xmax - edge_gap_mm
    y_top   = ymin + edge_gap_mm
    y_bot   = ymax - edge_gap_mm

    def linspace(a: float, b: float, k: int) -> List[float]:
        if k <= 1:
            return [ (a + b) / 2.0 ]
        step = (b - a) / float(k - 1)
        return [ a + i*step for i in range(k) ]

    # Edges
    if region in ("west", "left"):
        ys = linspace(y_top + end_margin_mm, y_bot - end_margin_mm, n)
        x  = x_left
        return [(x, y) for y in ys]

    if region in ("east", "right"):
        ys = linspace(y_top + end_margin_mm, y_bot - end_margin_mm, n)
        x  = x_right
        return [(x, y) for y in ys]

    if region in ("north", "top"):
        xs = linspace(x_left + end_margin_mm, x_right - end_margin_mm, n)
        y  = y_top
        return [(x, y) for x in xs]

    if region in ("south", "bottom"):
        xs = linspace(x_left + end_margin_mm, x_right - end_margin_mm, n)
        y  = y_bot
        return [(x, y) for x in xs]

    # Corners: “stack” away from corner along the longer board dimension
    w = (xmax - xmin); h = (ymax - ymin)
    stack_along_x = (w >= h)  # if board is wider, step along x; else along y
    step = 4.0  # mm between parts in the stack

    def stack_from(x0: float, y0: float, dx: float, dy: float) -> List[Tuple[float, float]]:
        return [(x0 + dx*i, y0 + dy*i) for i in range(n)]

    if region in ("top-left", "north-west", "nw"):
        x0, y0 = x_left, y_top
        return stack_from(x0, y0, step if stack_along_x else 0.0, 0.0 if stack_along_x else step)

    if region in ("top-right", "north-east", "ne"):
        x0, y0 = x_right, y_top
        return stack_from(x0, y0, -step if stack_along_x else 0.0, 0.0 if stack_along_x else step)

    if region in ("bottom-left", "south-west", "sw"):
        x0, y0 = x_left, y_bot
        return stack_from(x0, y0, step if stack_along_x else 0.0, 0.0 if stack_along_x else -step)

    if region in ("bottom-right", "south-east", "se"):
        x0, y0 = x_right, y_bot
        return stack_from(x0, y0, -step if stack_along_x else 0.0, 0.0 if stack_along_x else -step)

    # Default: center line distribution on the longer axis
    if w >= h:
        xs = linspace(x_left + end_margin_mm, x_right - end_margin_mm, n)
        return [(x, cy) for x in xs]
    else:
        ys = linspace(y_top + end_margin_mm, y_bot - end_margin_mm, n)
        return [(cx, y) for y in ys]


def _select_named_parts(b: BoardSnapshot, prompt: str) -> List[FPInfo]:
    """
    Heuristics:
      1) If 'stm' (or stm32, etc.) in prompt, prefer parts with value/footprint containing 'stm'
      2) Otherwise, try explicit refs mentioned in prompt (e.g., U3, U12)
      3) As a fallback when 'stm' mentioned but values absent (e.g., Chat snapshot), match U*
    """
    p = (prompt or "").lower()
    has_stm = ("stm" in p)
    tokens = [t for t in _name_tokens.findall(prompt)]
    refs_asked = {t.upper() for t in tokens if re.fullmatch(r"[A-Za-z]+[0-9]+", t)}

    # 1) Prefer STM by value/footprint keywords
    def is_stm(fp: FPInfo) -> bool:
        v = (fp.value or "").lower()
        f = (fp.footprint or "").lower()
        return ("stm" in v) or ("stm" in f) or ("stm32" in v) or ("stm32" in f)

    if has_stm:
        picks = [fp for fp in b.footprints if is_stm(fp)]
        if picks:
            return picks

    # 2) Explicit refs
    if refs_asked:
        picks = [fp for fp in b.footprints if fp.ref.upper() in refs_asked]
        if picks:
            return picks

    # 3) Fallback: if user said STM but we didn’t find values, guess U*
    if has_stm:
        picks = [fp for fp in b.footprints if fp.ref.upper().startswith("U")]
        if picks:
            return picks

    return []


def _anchor_or_nudge_named_parts(b: BoardSnapshot, prompt: str, plan: Dict[str, PlanItem], notes: List[str]) -> Set[str]:
    """
    Anchor to edge/corner if region words present.
    If 'slightly'/'a bit' present with a direction, apply only a small delta.
    Returns the set of refs that were explicitly targeted (locked).
    """
    targets = _select_named_parts(b, prompt)
    if not targets:
        return set()

    # slight nudge?
    nudge = _has_slight_nudge(prompt)
    if nudge is not None:
        dx, dy = nudge
        for fp in targets:
            plan[fp.ref] = PlanItem(ref=fp.ref, x_mm=fp.x_mm + dx, y_mm=fp.y_mm + dy, rot_deg=fp.rot_deg, side=fp.side)
        names = ", ".join(t.ref for t in targets[:6]) + ("…" if len(targets) > 6 else "")
        notes.append(f"Nudged {len(targets)} part(s) ({names}) by ({dx:.2f} mm, {dy:.2f} mm).")
        return {fp.ref for fp in targets}

    # else, hard anchor to specific region (full move “all the way”)
    region = _which_region_from_text(prompt)
    if not region:
        return set()

    positions = _distribute_along_region(b, region, len(targets), edge_gap_mm=3.0, end_margin_mm=3.0)
    for fp, (x, y) in zip(targets, positions):
        plan[fp.ref] = PlanItem(ref=fp.ref, x_mm=x, y_mm=y, rot_deg=fp.rot_deg, side=fp.side)

    names = ", ".join(t.ref for t in targets[:6]) + ("…" if len(targets) > 6 else "")
    notes.append(f"Anchored {len(targets)} part(s) ({names}) to {region} (absolute edge coords).")
    return {fp.ref for fp in targets}


def _anchor_antenna(b: BoardSnapshot, rules: PlanRuleToggles, plan: Dict[str, PlanItem], notes: List[str]):
    ants = [fp for fp in b.footprints if _is_antenna_part(fp)]
    if not ants:
        return
    ax, ay = _edge_anchor(b, rules.antenna_region)
    step = 2.0
    for i, fp in enumerate(ants):
        plan[fp.ref] = PlanItem(ref=fp.ref, x_mm=ax - i * step, y_mm=ay, rot_deg=0.0, side="top")
    notes.append(f"Placed {len(ants)} antenna parts at {rules.antenna_region}")

def _cluster_psu_and_anchor(b: BoardSnapshot, rules: PlanRuleToggles, plan: Dict[str, PlanItem], notes: List[str]):
    parts = [fp for fp in b.footprints if (_is_psu_controller(fp) or _is_power_inductor(fp) or _is_rectifier(fp)) and _in_power_island(fp)]
    if len(parts) < 3:
        return
    ax, ay = _edge_anchor(b, rules.psu_edge)
    spacing = 3.0
    start_x = ax - (len(parts) - 1) * spacing / 2.0
    for i, fp in enumerate(parts):
        plan[fp.ref] = PlanItem(ref=fp.ref, x_mm=start_x + i * spacing, y_mm=ay, rot_deg=0.0, side="top")
    notes.append(f"Anchored PSU cluster ({len(parts)} parts) at {rules.psu_edge}")


def _apply_channel_spacing(b: BoardSnapshot, fps: List[FPInfo], plan: Dict[str, PlanItem], rules: PlanRuleToggles, notes: List[str]):
    ics = [f for f in fps if f.ref.startswith("U")]
    for i, a in enumerate(ics):
        for bfp in ics[i + 1:]:
            n = _net_between(a, bfp)
            if n <= 1:
                continue
            ax = plan[a.ref].x_mm if a.ref in plan else a.x_mm
            ay = plan[a.ref].y_mm if a.ref in plan else a.y_mm
            bx = plan[bfp.ref].x_mm if bfp.ref in plan else bfp.x_mm
            by = plan[bfp.ref].y_mm if bfp.ref in plan else bfp.y_mm
            req = _channel_clearance_mm(n, rules.channel_track_mm, rules.channel_clear_mm)
            dist = math.hypot(ax - bx, ay - by)
            if dist < req:
                midx, midy = (ax + bx) / 2.0, (ay + by) / 2.0
                dx, dy = (ax - bx), (ay - by)
                L = math.hypot(dx, dy) or 1.0
                ux, uy = dx / L, dy / L
                a_new = (midx + ux * req / 2.0, midy + uy * req / 2.0)
                b_new = (midx - ux * req / 2.0, midy - uy * req / 2.0)
                plan[a.ref] = PlanItem(ref=a.ref, x_mm=a_new[0], y_mm=a_new[1], rot_deg=0.0, side=a.side)
                plan[bfp.ref] = PlanItem(ref=bfp.ref, x_mm=b_new[0], y_mm=b_new[1], rot_deg=0.0, side=bfp.side)
                notes.append(f"Spaced {a.ref}↔{bfp.ref} for {n} traces (~{req:.1f} mm corridor)")


# ----------------------------------------------------------------------------
# Prompt → rules helper
def _infer_rules_from_prompt(p: str, defaults: PlanRuleToggles) -> PlanRuleToggles:
    p = (p or "").lower()
    # copy defaults
    r = PlanRuleToggles(**defaults.model_dump())
    if "south" in p and ("psu" in p or "power" in p):
        r.psu_edge = "south"
    if "north" in p and ("psu" in p or "power" in p):
        r.psu_edge = "north"
    if "east" in p and ("psu" in p or "power" in p):
        r.psu_edge = "east"
    if "west" in p and ("psu" in p or "power" in p):
        r.psu_edge = "west"

    if ("top-right" in p) or ("north-east" in p) or re.search(r"\bne\b", p):
        r.antenna_region = "top-right"
    if ("top-left" in p) or ("north-west" in p) or re.search(r"\bnw\b", p):
        r.antenna_region = "top-left"
    if "bottom-right" in p or "south-east" in p or re.search(r"\bse\b", p):
        r.antenna_region = "bottom-right"
    if "bottom-left" in p or "south-west" in p or re.search(r"\bsw\b", p):
        r.antenna_region = "bottom-left"

    m = re.search(r"(?:decap|decoupl)[^\d]*?(\d+(?:\.\d+)?)\s*mm", p)
    if m:
        try:
            r.decap_radius_mm = float(m.group(1))
        except Exception:
            pass
    return r


# ----------------------------------------------------------------------------
# Snapshot normalizer (fixes missing width/height and coercion)
def _normalize_snapshot_for_model(snapshot: Any) -> BoardSnapshot:
    """
    Accepts dict or BoardSnapshot-like input and returns a valid BoardSnapshot.
    - Adds width_mm/height_mm if missing using outline_bbox.
    - Coerces strings/floats so Pydantic won’t choke on KiCad/wx types.
    """
    if isinstance(snapshot, BoardSnapshot):
        return snapshot
    if not isinstance(snapshot, dict):
        raise HTTPException(400, "snapshot must be an object")

    snap = dict(snapshot)  # shallow copy
    bbox = snap.get("outline_bbox")
    if bbox and ("width_mm" not in snap or "height_mm" not in snap):
        xmin, ymin, xmax, ymax = [float(v) for v in bbox]
        snap["width_mm"] = float(xmax - xmin)
        snap["height_mm"] = float(ymax - ymin)

    fixed_fps: List[Dict[str, Any]] = []
    for fp in snap.get("footprints", []):
        fixed_fps.append({
            "ref": str(fp.get("ref", "")),
            "value": (None if fp.get("value") in (None, "") else str(fp.get("value"))),
            "footprint": (None if fp.get("footprint") in (None, "") else str(fp.get("footprint"))),
            "x_mm": float(fp.get("x_mm", 0.0)),
            "y_mm": float(fp.get("y_mm", 0.0)),
            "rot_deg": float(fp.get("rot_deg", 0.0)),
            "side": fp.get("side", "top"),
            "pads": [
                {
                    "pad": str(p.get("pad", "")),
                    "net": str(p.get("net", "")),
                    "x_mm": float(p.get("x_mm", 0.0)),
                    "y_mm": float(p.get("y_mm", 0.0)),
                } for p in fp.get("pads", [])
            ],
        })
    snap["footprints"] = fixed_fps

    try:
        return BoardSnapshot(**snap)
    except Exception as e:
        raise HTTPException(400, f"Invalid snapshot: {e}")


# ----------------------------------------------------------------------------
# Chat + fallback planner
@app.post("/chat/complete")
def chat_complete(body: Dict[str, Any]):
    prompt = str(body.get("prompt", ""))
    raw_snapshot = body.get("snapshot")

    if raw_snapshot is None:
        bs = _to_snapshot_from_pcbnew()
        snapshot_dict = bs.model_dump()
    else:
        bs = _normalize_snapshot_for_model(raw_snapshot)
        snapshot_dict = bs.model_dump()

    # Try Gemini first
    g = _call_gemini_rest(prompt, snapshot_dict)
    if g:
        return g

    # Fallback: NL → rules → internal planner
    rules = _infer_rules_from_prompt(prompt, PlanRuleToggles())
    plan_resp = plan_place(PlanRequest(prompt=prompt, rules=rules, snapshot=bs))
    return {"text": "Planned with rule-based fallback.", "plan": [it.model_dump() for it in plan_resp.items]}


# ----------------------------------------------------------------------------
# Planner
@app.post("/plan/place", response_model=PlanResponse)
def plan_place(req: PlanRequest):
    # snapshot
    b = req.snapshot if req.snapshot is not None else _to_snapshot_from_pcbnew()

    plan: Dict[str, PlanItem] = {}
    notes: List[str] = []

    # 1) Anchors that don’t cause drift
    _anchor_antenna(b, req.rules, plan, notes)
    _cluster_psu_and_anchor(b, req.rules, plan, notes)

    # 2) Named STM/U* handling:
    #    - If the prompt contains named targets and a region → place at exact edge coords in one shot.
    #    - If the prompt says "slightly ..." → do a small delta move only.
    #    In both cases we lock these targets and do NOT move anything else in this run.
    locked_refs = _anchor_or_nudge_named_parts(b, req.prompt, plan, notes)

    # 3) If we did NOT target specific parts, we can run the global connectivity layout & spacing.
    if not locked_refs:
        centroids = _weighted_centroid_layout(b, b.footprints)
        for fp in b.footprints:
            if fp.ref not in plan:
                x, y = centroids.get(fp.ref, (fp.x_mm, fp.y_mm))
                plan[fp.ref] = PlanItem(ref=fp.ref, x_mm=x, y_mm=y, rot_deg=fp.rot_deg, side=fp.side)

        _apply_channel_spacing(b, b.footprints, plan, req.rules, notes)

    # 4) Decouplers near VCC pins (non-invasive to locked targets)
    def _place_decaps_local():
        ics = [fp for fp in b.footprints if fp.ref.startswith("U")]
        caps = [fp for fp in b.footprints if _is_decap(fp)]
        vcc_names = {n.upper() for n in VCC_NAMES}
        net_to_ics: Dict[str, List[FPInfo]] = {}
        for u in ics:
            nets = {p.net for p in u.pads if p.net}
            for n in nets:
                if n.upper() in vcc_names:
                    net_to_ics.setdefault(n, []).append(u)
        for c in caps:
            nets = {p.net for p in c.pads if p.net}
            vcc = None; gnd_ok = False
            for n in nets:
                if n.upper() in vcc_names: vcc = n
                if n.upper() in GND_NAMES: gnd_ok = True
            if vcc and gnd_ok and vcc in net_to_ics:
                # pick closest IC by current/plan pose
                def cur_xy(u: FPInfo) -> Tuple[float,float]:
                    x = plan[u.ref].x_mm if u.ref in plan else u.x_mm
                    y = plan[u.ref].y_mm if u.ref in plan else u.y_mm
                    return (x, y)
                best_u = min(net_to_ics[vcc], key=lambda u: math.hypot(cur_xy(u)[0] - c.x_mm, cur_xy(u)[1] - c.y_mm))
                ux, uy = cur_xy(best_u)
                dx, dy = c.x_mm - ux, c.y_mm - uy
                L = math.hypot(dx, dy) or 1.0
                nx, ny = dx / L, dy / L
                tx, ty = ux + nx * req.rules.decap_radius_mm, uy + ny * req.rules.decap_radius_mm
                plan[c.ref] = PlanItem(ref=c.ref, x_mm=tx, y_mm=ty, rot_deg=0.0, side=best_u.side)
                notes.append(f"Decap {c.ref} -> near {best_u.ref} {vcc}")
    _place_decaps_local()

    # 5) Notes
    p = (req.prompt or "").lower()
    if "power" in p and "edge" in p:
        notes.append("Prompt asked power at edge — handled by PSU anchoring.")
    if ("slightly" in p) or ("a bit" in p) or ("a little" in p):
        notes.append("Performed small nudge only; skipped global moves.")
    elif not locked_refs:
        notes.append("Applied weighted-centroid & spacing (no targeted refs).")
    else:
        notes.append("Moved only explicitly requested parts; skipped global moves.")
    notes.append(f"Plan contains {len(plan)} items.")

    return PlanResponse(ok=True, items=list(plan.values()), notes=notes)


# ----------------------------------------------------------------------------
# Root
@app.get("/")
def root():
    return {"ok": True, "try": ["POST /plan/place", "POST /chat/complete"], "version": app.version}
