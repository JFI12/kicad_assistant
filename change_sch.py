# pip install sexpdata
import sys, math, collections
from uuid import uuid4
from typing import Tuple, Dict, Any, Optional, List
from sexpdata import loads, dumps, Symbol

S = Symbol

# --- routing tunables ---
OBSTACLE_CLEARANCE_MM = 1.0  # margin around symbol hitboxes (for obstacle avoidance)
GRID_PAD_MM = 1.0            # halo around obstacles in the grid
ESCAPE_MARGIN_MM = 1.5       # how far outside a symbol edge to place escape points
MAX_BFS_NODES = 10000

# ---------- I/O + pretty ----------
def load_sch(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return loads(f.read())

def pretty_dumps(expr, indent=0) -> str:
    out = ""
    spacing = "  " * indent
    if isinstance(expr, list):
        out += spacing + "("
        for i, item in enumerate(expr):
            if isinstance(item, list):
                out += "\n" + pretty_dumps(item, indent + 1)
            else:
                out += ("" if i == 0 else " ") + dumps(item)
        out += ")"
    else:
        out += spacing + dumps(expr)
    return out

def save_sch(sexp, path: str):
    text = pretty_dumps(sexp) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def uid() -> str:
    return str(uuid4())

# ---------- helpers ----------
def Q(v: float) -> float:
    """Quantize to KiCad's 0.01 mm grid to avoid 'almost' coordinates."""
    return round(float(v), 2)

def node_tag(n): return n[0] if isinstance(n, list) and n else None

def find_section(doc, tag: str):
    for it in doc[1:]:
        if isinstance(it, list) and node_tag(it) == S(tag):
            return it
    return None

def find_symbols(doc):
    for it in doc[1:]:
        if isinstance(it, list) and node_tag(it) == S("symbol"):
            yield it

def get_prop(symbol_node, key: str) -> Optional[str]:
    for it in symbol_node:
        if isinstance(it, list) and node_tag(it) == S("property"):
            if len(it) >= 3 and it[1] == key:
                return it[2]
    return None

def parse_at(node) -> Tuple[float, float, float]:
    for it in node:
        if isinstance(it, list) and node_tag(it) == S("at"):
            if len(it) == 3:
                return Q(it[1]), Q(it[2]), 0.0
            elif len(it) >= 4:
                return Q(it[1]), Q(it[2]), float(it[3])
    return 0.0, 0.0, 0.0

def parse_unit_convert(node) -> Tuple[int, int]:
    u = 1; c = 0
    for it in node:
        if isinstance(it, list) and node_tag(it) == S("unit") and len(it) >= 2:
            try: u = int(it[1])
            except Exception: pass
        if isinstance(it, list) and node_tag(it) == S("convert") and len(it) >= 2:
            try: c = int(it[1])
            except Exception: pass
    return u, c

def parse_mirror(node) -> Tuple[bool, bool]:
    mx = my = False
    for it in node:
        if isinstance(it, list) and node_tag(it) == S("mirror"):
            # KiCad may put 'x' and/or 'y' as separate tokens
            for tok in it[1:]:
                if isinstance(tok, str):
                    if tok == "x": mx = True
                    if tok == "y": my = True
    return mx, my

# ---------- build library pin maps (unit/alt aware) ----------
def build_lib_pin_maps(lib_symbols_node):
    """
    { lib_id: {
        'pins': [ {'number','name','x','y','angle','unit','convert'}, ... ],
        'bbox_local': (xmin,xmax,ymin,ymax)
      } }
    """
    res = {}
    if not lib_symbols_node:
        return res

    def walk(node, u_ctx: int, c_ctx: int, pins_out: List[dict], xs: List[float], ys: List[float]):
        if not isinstance(node, list): return
        tag = node_tag(node)
        if tag == S("symbol"):
            u_local, c_local = u_ctx, c_ctx
            for it in node[1:]:
                if isinstance(it, list) and node_tag(it) == S("unit") and len(it) >= 2:
                    try: u_local = int(it[1])
                    except Exception: pass
                if isinstance(it, list) and node_tag(it) == S("convert") and len(it) >= 2:
                    try: c_local = int(it[1])
                    except Exception: pass
            for it in node[1:]:
                walk(it, u_local, c_local, pins_out, xs, ys)
            return
        if tag == S("pin"):
            dx = dy = ang = 0.0
            name = number = None
            for sub in node:
                if isinstance(sub, list):
                    tg = node_tag(sub)
                    if tg == S("at"):
                        if len(sub) >= 4:
                            dx, dy, ang = Q(sub[1]), Q(sub[2]), float(sub[3])
                        elif len(sub) == 3:
                            dx, dy, ang = Q(sub[1]), Q(sub[2]), 0.0
                    elif tg == S("name") and len(sub) >= 2:
                        name = sub[1]
                    elif tg == S("number") and len(sub) >= 2:
                        number = sub[1]
            pins_out.append({
                "number": (str(number) if number else None),
                "name":   (str(name)   if name   else None),
                "x": dx, "y": dy, "angle": ang,
                "unit": u_ctx, "convert": c_ctx
            })
            xs.append(dx); ys.append(dy)
            return
        for it in node[1:]:
            walk(it, u_ctx, c_ctx, pins_out, xs, ys)

    for sym in lib_symbols_node[1:]:
        if not (isinstance(sym, list) and node_tag(sym) == S("symbol")): continue
        if len(sym) < 2 or not isinstance(sym[1], str): continue
        lib_id = sym[1]

        pins: List[dict] = []
        xs: List[float] = []; ys: List[float] = []
        walk(sym, 1, 0, pins, xs, ys)
        bbox_local = (min(xs), max(xs), min(ys), max(ys)) if xs and ys else (0.0,0.0,0.0,0.0)
        res[lib_id] = {"pins": pins, "bbox_local": bbox_local}
    return res

# ---------- exact transforms & connection points ----------
def transform_local_to_world(dx: float, dy: float, inst_at, mirror_x: bool, mirror_y: bool):
    """
    KiCad 'mirror' semantics:
      - 'x' => mirror about X-axis (flip Y)
      - 'y' => mirror about Y-axis (flip X)
    """
    # Apply mirror in local coords using correct axes:
    if mirror_y:  # mirror about Y-axis flips X
        dx = -dx
    if mirror_x:  # mirror about X-axis flips Y
        dy = -dy

    x0, y0, ang = inst_at
    a = int(round(ang)) % 360
    if a == 0:
        X, Y = dx, dy
    elif a == 90:
        X, Y = -dy, dx
    elif a == 180:
        X, Y = -dx, -dy
    elif a == 270:
        X, Y =  dy, -dx
    else:
        r = math.radians(ang)
        X = dx*math.cos(r) - dy*math.sin(r)
        Y = dx*math.sin(r) + dy*math.cos(r)
    # Quantize to KiCad grid
    return (round(x0 + X, 2), round(y0 + Y, 2))

def pin_world_conn(pin: dict, inst_at, mx: bool, my: bool) -> tuple[float, float]:
    """Return the EXACT electrical connection point of a pin in world coords."""
    return transform_local_to_world(float(pin["x"]), float(pin["y"]), inst_at, mx, my)

# ---------- drawing ----------
def add_wire(doc, x1, y1, x2, y2):
    x1, y1, x2, y2 = Q(x1), Q(y1), Q(x2), Q(y2)
    wire = [
        S('wire'),
        [S('pts'), [S('xy'), x1, y1], [S('xy'), x2, y2]],
        [S('uuid'), uid()]
    ]
    doc.append(wire)

def add_junction(doc, x, y):
    x, y = Q(x), Q(y)
    j = [ S('junction'), [S('at'), x, y], [S('uuid'), uid()] ]
    doc.append(j)

# ---------- routing helpers ----------
def symbol_bbox_world(bbox_local, inst_at, mx, my):
    xmin, xmax, ymin, ymax = bbox_local
    c = [
        transform_local_to_world(xmin, ymin, inst_at, mx, my),
        transform_local_to_world(xmin, ymax, inst_at, mx, my),
        transform_local_to_world(xmax, ymin, inst_at, mx, my),
        transform_local_to_world(xmax, ymax, inst_at, mx, my),
    ]
    xs = [p[0] for p in c]; ys = [p[1] for p in c]
    return (min(xs), max(xs), min(ys), max(ys))

def expand_rect(rect, m):
    x0,x1,y0,y1 = rect
    return (Q(x0-m), Q(x1+m), Q(y0-m), Q(y1+m))

def seg_intersects_rect(x1,y1,x2,y2, rect) -> bool:
    rx0, rx1, ry0, ry1 = rect
    if max(x1,x2) < rx0 or min(x1,x2) > rx1 or max(y1,y2) < ry0 or min(y1,y2) > ry1:
        return False
    if (rx0 <= x1 <= rx1 and ry0 <= y1 <= ry1) or (rx0 <= x2 <= rx1 and ry0 <= y2 <= ry1):
        return True
    if x1 == x2:
        return (rx0 <= x1 <= rx1) and not (max(y1,y2) < ry0 or min(y1,y2) > ry1)
    if y1 == y2:
        return (ry0 <= y1 <= ry1) and not (max(x1,x2) < rx0 or min(x1,x2) > rx1)
    return False

def path_clear(segments, obstacles):
    for (a,b) in segments:
        x1,y1 = a; x2,y2 = b
        for rect in obstacles:
            if seg_intersects_rect(x1,y1,x2,y2, rect):
                return False
    return True

def manhattan_candidate_paths(p1, p2):
    x1,y1 = p1; x2,y2 = p2
    return [
        [(p1,(x2,y1)), ((x2,y1),p2)],
        [(p1,(x1,y2)), ((x1,y2),p2)],
    ]

def bfs_route(p1, p2, obstacles, xs, ys):
    xs = sorted(set(Q(x) for x in xs + [p1[0], p2[0]]))
    ys = sorted(set(Q(y) for y in ys + [p1[1], p2[1]]))
    xi = {x:i for i, x in enumerate(xs)}
    yi = {y:i for i, y in enumerate(ys)}
    start = (xi[p1[0]], yi[p1[1]])
    goal  = (xi[p2[0]], yi[p2[1]])

    def passable(a, b):
        (i1,j1),(i2,j2) = a,b
        if i1==i2 and j1!=j2:
            x = xs[i1]; y_lo, y_hi = sorted((ys[j1], ys[j2]))
            seg = ((x,y_lo),(x,y_hi))
        elif j1==j2 and i1!=i2:
            y = ys[j1]; x_lo, x_hi = sorted((xs[i1], xs[i2]))
            seg = ((x_lo,y),(x_hi,y))
        else:
            return False
        return path_clear([seg], obstacles)

    Qd = collections.deque([start])
    prev = {start: None}
    count = 0
    while Qd and count < MAX_BFS_NODES:
        u = Qd.popleft(); count += 1
        if u == goal: break
        i,j = u
        nbrs = []
        if i-1 >= 0: nbrs.append((i-1,j))
        if i+1 < len(xs): nbrs.append((i+1,j))
        if j-1 >= 0: nbrs.append((i,j-1))
        if j+1 < len(ys): nbrs.append((i,j+1))
        for v in nbrs:
            if v in prev: continue
            if passable(u,v):
                prev[v] = u
                Qd.append(v)

    if goal not in prev:
        return None

    # reconstruct & compress
    path_nodes = []
    cur = goal
    while cur is not None:
        path_nodes.append(cur)
        cur = prev[cur]
    path_nodes.reverse()
    pts = [(xs[i], ys[j]) for (i,j) in path_nodes]
    if len(pts) < 2: return []
    segs = []
    s = pts[0]
    prev_dir = (pts[1][0]-pts[0][0], pts[1][1]-pts[0][1])
    for k in range(1,len(pts)):
        d = (pts[k][0]-pts[k-1][0], pts[k][1]-pts[k-1][1])
        if d == prev_dir: continue
        segs.append((s, pts[k-1])); s = pts[k-1]; prev_dir = d
    segs.append((s, pts[-1]))
    return segs

def escape_point(pin_xy, bbox_world, margin=ESCAPE_MARGIN_MM):
    x, y = pin_xy
    x0, x1, y0, y1 = bbox_world
    dl = abs(x - x0)
    dr = abs(x1 - x)
    db = abs(y - y0)
    dt = abs(y1 - y)
    m = min(dl, dr, db, dt)
    if m == dl:   return (Q(x0 - margin), y)
    if m == dr:   return (Q(x1 + margin), y)
    if m == db:   return (x, Q(y0 - margin))
    return (x, Q(y1 + margin))

# ---------- menus (return ACTUAL pin entries) ----------
def list_modules(doc, pin_maps, preview_count=10):
    rows = []
    for sym in find_symbols(doc):
        ref = get_prop(sym, "Reference") or "<?>"
        val = get_prop(sym, "Value") or ""
        lib_id = None
        for it in sym:
            if isinstance(it, list) and node_tag(it) == S("lib_id") and len(it) >= 2:
                lib_id = it[1]; break
        preview = ""
        if lib_id and lib_id in pin_maps:
            names = sorted({p["name"] for p in pin_maps[lib_id]["pins"] if p["name"]})
            if names:
                preview = ", ".join(list(names)[:preview_count]) + ("…" if len(names)>preview_count else "")
        rows.append((ref, val, lib_id or "", preview))
    rows.sort(key=lambda r: r[0])
    return rows

def choose_module(doc, pin_maps, prompt):
    rows = list_modules(doc, pin_maps)
    if not rows:
        raise RuntimeError("No (symbol ...) instances found.")
    print(prompt)
    print("\nIdx  Reference  Value                 lib_id                              Pins (preview)")
    print("---- ---------  --------------------  ----------------------------------  --------------------------")
    for i, (ref, val, lib_id, preview) in enumerate(rows, 1):
        print(f"{i:>3}. {ref:<9} {val:<20} {lib_id:<34}  {preview}")
    choice = input("\nType number, Reference (e.g. U3), or exact Value (if unique): ").strip()

    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(rows): ref = rows[idx-1][0]
        else: raise RuntimeError("Index out of range.")
    else:
        choice_up = choice.upper()
        ref = None
        for r in rows:
            if r[0].upper() == choice_up:
                ref = r[0]; break
        if ref is None:
            matches = [r for r in rows if r[1] == choice]
            if len(matches) == 1: ref = matches[0][0]
            elif len(matches) > 1: raise RuntimeError(f"Value '{choice}' matches multiple modules. Pick by index or Reference.")
            else: raise RuntimeError(f"No module found for '{choice}'.")

    for sym in find_symbols(doc):
        if get_prop(sym, "Reference") == ref:
            lib_id = ""
            for it in sym:
                if isinstance(it, list) and node_tag(it) == S("lib_id") and len(it) >= 2:
                    lib_id = it[1]
            at = parse_at(sym)
            mx, my = parse_mirror(sym)
            unit, convert = parse_unit_convert(sym)
            return ref, lib_id, at, mx, my, unit, convert
    raise RuntimeError(f"Internal: selected Reference '{ref}' not found.")

def list_pins_for_instance(pin_maps, lib_id, unit, convert):
    m = pin_maps.get(lib_id)
    if not m: raise RuntimeError(f"Embedded library for '{lib_id}' not found in (lib_symbols).")
    pins = [p for p in m["pins"] if p["unit"]==unit and p["convert"]==convert] or m["pins"]
    def sort_key(p):
        n = p["number"]
        if n is None: return (2, "", p["name"] or "")
        try: return (0, int(n), p["name"] or "")
        except ValueError: return (1, n, p["name"] or "")
    return sorted(pins, key=sort_key)

def choose_pin(pin_maps, lib_id, unit, convert, prompt):
    entries = list_pins_for_instance(pin_maps, lib_id, unit, convert)
    print(prompt)
    print("\nIdx  Pin (number : name)")
    print("---- --------------------")
    for i, p in enumerate(entries, 1):
        num = p["number"] or ""; nam = p["name"] or ""
        disp = f"{num}: {nam}" if num and nam else (num or nam or "?")
        print(f"{i:>3}. {disp}")
    choice = input("\nType index, pin NUMBER (e.g. 1), or pin NAME (e.g. DVDD): ").strip()
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(entries): return entries[idx-1]
        raise RuntimeError("Index out of range.")
    q = choice.upper()
    matches = [p for p in entries if (p["number"] and p["number"].upper()==q) or (p["name"] and p["name"].upper()==q)]
    if len(matches)==1: return matches[0]
    if len(matches)>1:
        print("\nAmbiguous pin name; choose by index:")
        for i,p in enumerate(matches,1):
            num=p['number'] or '?'
            print(f"{i}. number={num} at ({p['x']:.2f},{p['y']:.2f})")
        while True:
            ans=input("Index: ").strip()
            if ans.isdigit() and 1<=int(ans)<=len(matches): return matches[int(ans)-1]
    raise RuntimeError(f"Pin '{choice}' not found in this unit/alt.")

# ---------- main connect ----------
def connect_pins_interactive(sch_path: str, avoid_obstacles=True):
    doc = load_sch(sch_path)
    if node_tag(doc) != S('kicad_sch'):
        raise RuntimeError("Not a KiCad schematic root.")

    lib_syms = find_section(doc, "lib_symbols")
    pin_maps = build_lib_pin_maps(lib_syms)
    if not pin_maps:
        raise RuntimeError("No (lib_symbols ...) section found. In KiCad, run: Tools → Update Symbols so pins are embedded.")

    # pick source
    from_ref, from_lib, from_at, fmx, fmy, funit, fconv = choose_module(doc, pin_maps, "\nAvailable modules:")
    A = choose_pin(pin_maps, from_lib, funit, fconv, f"\nPins for {from_ref} ({from_lib}) unit {funit} alt {fconv}:")
    Ax, Ay = pin_world_conn(A, from_at, fmx, fmy)
    print(f"Resolved {from_ref}: {A['number'] or ''} {A['name'] or ''} @ ({Ax:.2f}, {Ay:.2f})")

    # pick destination
    to_ref, to_lib, to_at, tmx, tmy, tunit, tconv = choose_module(doc, pin_maps, "\nConnect to which module?")
    B = choose_pin(pin_maps, to_lib, tunit, tconv, f"\nPins for {to_ref} ({to_lib}) unit {tunit} alt {tconv}:")
    Bx, By = pin_world_conn(B, to_at, tmx, tmy)
    print(f"Resolved {to_ref}: {B['number'] or ''} {B['name'] or ''} @ ({Bx:.2f}, {By:.2f})")

    # junctions at pin tips (exact coords)
    add_junction(doc, Ax, Ay)
    add_junction(doc, Bx, By)

    # --- escape points outside edges (avoid running along pin columns) ---
    from_bb_world = symbol_bbox_world(pin_maps[from_lib]["bbox_local"], from_at, fmx, fmy)
    to_bb_world   = symbol_bbox_world(pin_maps[to_lib]["bbox_local"],   to_at,   tmx, tmy)
    A_esc = escape_point((Ax, Ay), from_bb_world, ESCAPE_MARGIN_MM)
    B_esc = escape_point((Bx, By), to_bb_world,   ESCAPE_MARGIN_MM)

    # short stubs from pin tips to escape points
    add_wire(doc, Ax, Ay, A_esc[0], A_esc[1])
    add_wire(doc, B_esc[0], B_esc[1], Bx, By)

    # route BETWEEN escape points
    p1 = A_esc
    p2 = B_esc

    # obstacles: include all symbols
    obstacles = []
    xs_grid = [p1[0], p2[0]]
    ys_grid = [p1[1], p2[1]]
    if avoid_obstacles:
        for sym in find_symbols(doc):
            lib_id = None
            for it in sym:
                if isinstance(it, list) and node_tag(it) == S("lib_id") and len(it) >= 2:
                    lib_id = it[1]; break
            if not lib_id or lib_id not in pin_maps:
                continue
            at = parse_at(sym); mx,my = parse_mirror(sym)
            rect = expand_rect(symbol_bbox_world(pin_maps[lib_id]["bbox_local"], at, mx, my),
                               OBSTACLE_CLEARANCE_MM)
            obstacles.append(rect)
            xs_grid += [rect[0]-GRID_PAD_MM, rect[1]+GRID_PAD_MM]
            ys_grid += [rect[2]-GRID_PAD_MM, rect[3]+GRID_PAD_MM]

    # try simple L first
    for segs in manhattan_candidate_paths(p1, p2):
        if path_clear(segs, obstacles):
            for (a,b) in segs:
                add_wire(doc, a[0], a[1], b[0], b[1])
            save_sch(doc, sch_path)
            print(f"\nConnected {from_ref}:{A['number'] or A['name']} -> {to_ref}:{B['number'] or B['name']} (L-route via escapes)")
            print("Saved schematic. Open in KiCad and run ERC.")
            return

    # BFS detour
    segs = bfs_route(p1, p2, obstacles, xs_grid, ys_grid)
    if not segs:
        # last resort: straight between escapes
        add_wire(doc, p1[0], p1[1], p2[0], p2[1])
        save_sch(doc, sch_path)
        print("\nWARNING: No obstacle-free orthogonal path; drew straight between escapes.")
        return

    for (a,b) in segs:
        add_wire(doc, a[0], a[1], b[0], b[1])
    save_sch(doc, sch_path)
    print(f"\nConnected {from_ref}:{A['number'] or A['name']} -> {to_ref}:{B['number'] or B['name']} (detoured via escapes)")
    print("Saved schematic. Open in KiCad and run ERC.")

# ---------- CLI ----------
if __name__ == "__main__":
    sch = sys.argv[1] if len(sys.argv) >= 2 else input("Path to .kicad_sch: ").strip()
    connect_pins_interactive(sch_path=sch, avoid_obstacles=True)
