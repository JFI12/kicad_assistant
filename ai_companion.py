# ai_companion.py — KiCad action plugin + editor-friendly shims (KiCad 9-ready)
from __future__ import annotations

import json, os, threading, urllib.request, urllib.error, socket, subprocess, time, sys, urllib.parse
from typing import TYPE_CHECKING, Optional

# ---------- KiCad/wx imports with shims so VS Code is happy ----------
if TYPE_CHECKING:
    import pcbnew
    import wx

try:
    import pcbnew  # type: ignore
except Exception:
    pcbnew = None  # type: ignore

try:
    import wx  # type: ignore
except Exception:
    wx = None  # type: ignore


class _ActionPluginShim:
    def defaults(self): pass
    def Run(self): pass
    def register(self): pass


if pcbnew is None:
    class _pcbnew_shim:  # type: ignore
        ActionPlugin = _ActionPluginShim
    pcbnew = _pcbnew_shim()  # type: ignore

# ---------- Config ----------
API = os.environ.get("KICAD_AI_API", "http://127.0.0.1:8000")
SERVER_EXE = os.path.join(os.path.dirname(__file__), "kicad_ai_server.exe")

# ---------- Small HTTP helpers ----------
def _server_is_up(host="127.0.0.1", port=8000, timeout=0.4):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def _ensure_server_async(panel):
    def run():
        if _server_is_up():
            if wx: wx.CallAfter(panel.info, f"Backend OK at {API}")
            return
        if os.path.exists(SERVER_EXE):
            try:
                creationflags = 0
                if sys.platform.startswith("win"):
                    creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                subprocess.Popen(
                    [SERVER_EXE],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                    creationflags=creationflags
                )
                for _ in range(40):  # ~4s
                    if _server_is_up():
                        if wx: wx.CallAfter(panel.info, "Started backend.")
                        return
                    time.sleep(0.1)
                if wx: wx.CallAfter(panel.info, "Backend failed to start (timeout). Start it manually.")
            except Exception as e:
                if wx: wx.CallAfter(panel.info, f"Could not start backend: {e}")
        else:
            if wx:
                wx.CallAfter(panel.info, f"No backend exe at {SERVER_EXE}. Start the server manually:")
                wx.CallAfter(panel.info, "  python -m uvicorn companion.app:APP --reload --host 127.0.0.1 --port 8000")
    threading.Thread(target=run, daemon=True).start()

def _post_json(path, payload):
    req = urllib.request.Request(
        API + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

def _get_json(path):
    with urllib.request.urlopen(API + path, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

# ---------- Windows: target Eeschema File→Reload via WM_COMMAND -----------
_last_reload_ts = 0.0

def _win__send_keys(vks: list[int]):
    """Send a sequence of virtual-key presses (down/up)."""
    import ctypes, time as _t
    import ctypes.wintypes as wt
    user32 = ctypes.windll.user32

    INPUT_KEYBOARD = 1
    KEYEVENTF_KEYUP = 0x0002

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD),
                    ("dwFlags", wt.DWORD), ("time", wt.DWORD),
                    ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", wt.DWORD), ("ki", KEYBDINPUT)]

    def send(vk, up=False):
        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.ki = KEYBDINPUT(wVk=vk, wScan=0, dwFlags=(KEYEVENTF_KEYUP if up else 0), time=0, dwExtraInfo=None)
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    for k in vks:
        send(k, up=False); _t.sleep(0.015)
    for k in reversed(vks):
        send(k, up=True);  _t.sleep(0.015)

def _win__find_eeschema_hwnd(filename_hint: Optional[str] = None) -> Optional[int]:
    """Find an Eeschema main window. Prefer one whose title contains filename_hint."""
    import ctypes
    import ctypes.wintypes as wt
    user32 = ctypes.windll.user32

    EnumWindows      = user32.EnumWindows
    EnumWindowsProc  = ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
    GetWindowTextW   = user32.GetWindowTextW
    GetWindowTextLengthW = user32.GetWindowTextLengthW
    IsWindowVisible  = user32.IsWindowVisible

    best = wt.HWND(0)
    fallback = wt.HWND(0)

    def callback(hwnd, _):
        nonlocal best, fallback
        if not IsWindowVisible(hwnd):
            return True
        length = GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buff = ctypes.create_unicode_buffer(length + 1)
        GetWindowTextW(hwnd, buff, length + 1)
        title = buff.value or ""
        if ("Eeschema" in title) or ("Schematic" in title):
            if filename_hint and os.path.basename(filename_hint) in title:
                best = hwnd
                return False
            if not fallback:
                fallback = hwnd
        return True

    EnumWindows(EnumWindowsProc(callback), 0)
    return (best or fallback).value or None

def _win_reload_eeschema(filename_hint: Optional[str] = None) -> bool:
    """Try File→Revert, then File→Reload, then Ctrl+R. Debounced to avoid spam."""
    if not sys.platform.startswith("win"):
        return False

    global _last_reload_ts
    now = time.time()
    if now - _last_reload_ts < 0.7:
        return True
    _last_reload_ts = now

    try:
        import ctypes
        import ctypes.wintypes as wt
        user32 = ctypes.windll.user32

        hwnd = _win__find_eeschema_hwnd(filename_hint)
        if not hwnd:
            return False
        user32.SetForegroundWindow(wt.HWND(hwnd))
        time.sleep(0.05)

        GetMenu         = user32.GetMenu
        GetSubMenu      = user32.GetSubMenu
        GetMenuItemCount= user32.GetMenuItemCount
        GetMenuStringW  = user32.GetMenuStringW
        GetMenuItemID   = user32.GetMenuItemID
        PostMessageW    = user32.PostMessageW

        MF_BYPOSITION = 0x00000400
        WM_COMMAND    = 0x0111

        hMenu = GetMenu(wt.HWND(hwnd))
        if not hMenu:
            raise RuntimeError("no menu")

        fileMenu = GetSubMenu(hMenu, 0)
        if not fileMenu:
            raise RuntimeError("no file submenu")

        count = GetMenuItemCount(fileMenu)
        buf   = ctypes.create_unicode_buffer(256)

        def find_cmd_id(target_words):
            for i in range(max(0, count)):
                GetMenuStringW(fileMenu, i, buf, len(buf), MF_BYPOSITION)
                text = (buf.value or "").lower()
                if any(w in text for w in target_words):
                    cmd_id = GetMenuItemID(fileMenu, i)
                    if cmd_id != -1:
                        return cmd_id
            return None

        # Try “Revert” first (some builds use this), then “Reload”
        cmd = find_cmd_id(["revert"])
        if cmd is None:
            cmd = find_cmd_id(["reload"])
        if cmd is not None:
            PostMessageW(wt.HWND(hwnd), WM_COMMAND, cmd, 0)
            return True
    except Exception:
        pass

    # Fallback: Ctrl+R
    try:
        _win__send_keys([0x11, 0x52])  # CTRL + R
        time.sleep(0.1)
        return True
    except Exception:
        return False

# ---------- UI (only when wx is available) ----------
if wx is not None:

    class AIPanel(wx.Panel):
        def __init__(self, parent):
            super().__init__(parent)

            # --- Top: choose schematic
            self.txt_path = wx.TextCtrl(self, value="", size=(520, -1))
            self.btn_browse = wx.Button(self, label="Browse .kicad_sch…")
            self.btn_set = wx.Button(self, label="Set Current Schematic")

            # --- Middle left: place controls
            self.txt_lib = wx.TextCtrl(self, value="Device:R", size=(220, -1))
            self.txt_x   = wx.TextCtrl(self, value="50", size=(60, -1))
            self.txt_y   = wx.TextCtrl(self, value="50", size=(60, -1))
            self.txt_ref = wx.TextCtrl(self, value="R",  size=(40, -1))
            self.btn_place = wx.Button(self, label="Place Symbol")
            self.btn_list  = wx.Button(self, label="List Symbols")
            self.btn_reload= wx.Button(self, label="Reload Eeschema")

            # --- Middle right: library search
            self.txt_query = wx.TextCtrl(self, value="", size=(240, -1))
            self.btn_scan  = wx.Button(self, label="Scan Libs")
            self.btn_search= wx.Button(self, label="Search")
            self.lst_results = wx.ListCtrl(self, style=wx.LC_REPORT|wx.BORDER_SUNKEN, size=(-1, 180))
            self.lst_results.InsertColumn(0, "lib_id", width=240)
            self.lst_results.InsertColumn(1, "name",   width=120)
            self.lst_results.InsertColumn(2, "desc",   width=260)

            self.log = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY)

            # Layout
            top = wx.FlexGridSizer(2, 3, 6, 6)
            top.AddMany([
                (wx.StaticText(self, label="Schematic (.kicad_sch):"), 0, wx.ALIGN_CENTER_VERTICAL),
                (self.txt_path, 1, wx.EXPAND),
                (self.btn_browse, 0),
                (self.btn_set, 0), (wx.Panel(self), 0), (wx.Panel(self), 0),
            ])
            top.AddGrowableCol(1, 1)

            place_row = wx.BoxSizer(wx.HORIZONTAL)
            place_row.Add(wx.StaticText(self, label="lib_id:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
            place_row.Add(self.txt_lib, 0, wx.RIGHT, 12)
            place_row.Add(wx.StaticText(self, label="X:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
            place_row.Add(self.txt_x, 0, wx.RIGHT, 10)
            place_row.Add(wx.StaticText(self, label="Y:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
            place_row.Add(self.txt_y, 0, wx.RIGHT, 10)
            place_row.Add(wx.StaticText(self, label="Ref prefix:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
            place_row.Add(self.txt_ref, 0, wx.RIGHT, 10)
            place_row.Add(self.btn_place, 0, wx.RIGHT, 6)
            place_row.Add(self.btn_list, 0, wx.RIGHT, 6)
            place_row.Add(self.btn_reload, 0)

            srch_row = wx.BoxSizer(wx.HORIZONTAL)
            srch_row.Add(wx.StaticText(self, label="Search:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
            srch_row.Add(self.txt_query, 0, wx.RIGHT, 8)
            srch_row.Add(self.btn_search, 0, wx.RIGHT, 8)
            srch_row.Add(self.btn_scan, 0)

            mid = wx.BoxSizer(wx.HORIZONTAL)
            left = wx.BoxSizer(wx.VERTICAL)
            left.Add(place_row, 0, wx.BOTTOM, 8)
            right = wx.BoxSizer(wx.VERTICAL)
            right.Add(srch_row, 0, wx.BOTTOM, 4)
            right.Add(self.lst_results, 0, wx.EXPAND)
            mid.Add(left, 0, wx.RIGHT, 16)
            mid.Add(right, 1, wx.EXPAND)

            s = wx.BoxSizer(wx.VERTICAL)
            s.Add(top, 0, wx.ALL | wx.EXPAND, 8)
            s.Add(wx.StaticLine(self), 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
            s.Add(mid, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 8)
            s.Add(self.log, 1, wx.EXPAND | wx.ALL, 8)
            self.SetSizer(s)

            # Bindings
            self.btn_browse.Bind(wx.EVT_BUTTON, self.on_browse)
            self.btn_set.Bind(wx.EVT_BUTTON, self.on_set_current)
            self.btn_place.Bind(wx.EVT_BUTTON, self.on_place)
            self.btn_list.Bind(wx.EVT_BUTTON, self.on_list)
            self.btn_reload.Bind(wx.EVT_BUTTON, self.on_reload)
            self.btn_scan.Bind(wx.EVT_BUTTON, self.on_scan)
            self.btn_search.Bind(wx.EVT_BUTTON, self.on_search)
            self.lst_results.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self.on_pick_result)

            self.info(f"API: {API}")
            _ensure_server_async(self)

        # helpers
        def info(self, msg): self.log.AppendText(str(msg) + "\n")

        def _current_path(self) -> Optional[str]:
            p = self.txt_path.GetValue().strip()
            return p if p else None

        def _fill_results(self, items):
            self.lst_results.DeleteAllItems()
            for it in items[:250]:
                idx = self.lst_results.InsertItem(self.lst_results.GetItemCount(), it.get("lib_id",""))
                self.lst_results.SetItem(idx, 1, it.get("name",""))
                self.lst_results.SetItem(idx, 2, it.get("desc",""))

        # events
        def on_browse(self, _evt):
            with wx.FileDialog(self, "Choose schematic", wildcard="KiCad schematic (*.kicad_sch)|*.kicad_sch",
                               style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
                if dlg.ShowModal() == wx.ID_OK:
                    self.txt_path.SetValue(dlg.GetPath())

        def on_set_current(self, _evt):
            path = self._current_path()
            if not path:
                self.info("Pick a .kicad_sch path first.")
                return
            if not os.path.exists(path):
                self.info(f"Not found: {path}")
                return

            def work():
                try:
                    r = _post_json("/project/open", {"path": path})
                    wx.CallAfter(self.info, f"Current schematic set: {r.get('path')}")
                    wx.CallAfter(self.info, "Eeschema: File → Reload (or Revert) to see changes.")
                except urllib.error.HTTPError as e:
                    wx.CallAfter(self.info, f"project/open error: {e.read().decode('utf-8', 'ignore')}")
                except Exception as e:
                    wx.CallAfter(self.info, f"project/open error: {e}")

            threading.Thread(target=work, daemon=True).start()

        def on_place(self, _evt):
            try:
                lib  = self.txt_lib.GetValue().strip() or "Device:R"
                x    = float(self.txt_x.GetValue())
                y    = float(self.txt_y.GetValue())
                refp = self.txt_ref.GetValue().strip() or "R"
            except Exception:
                self.info("Invalid numbers for X/Y.")
                return

            filename_hint = self._current_path()

            def work():
                try:
                    r = _post_json("/schematic/place_symbol",
                                   {"lib_id": lib, "x": x, "y": y, "rotation": 0.0, "ref_prefix": refp})
                    ref = r.get("ref"); uid = r.get("uuid")
                    wx.CallAfter(self.info, f"Placed {lib} at ({r.get('x')},{r.get('y')}) → uuid={uid}, ref={ref}, fp={r.get('footprint') or ''}")

                    # Give mtime watcher a beat, then ask Eeschema to reload/revert
                    time.sleep(0.15)
                    if _win_reload_eeschema(filename_hint):
                        wx.CallAfter(self.info, "Reload/Revert command sent to Eeschema.")
                    else:
                        wx.CallAfter(self.info, "Reload not detected; try File→Revert manually.")
                except urllib.error.HTTPError as e:
                    wx.CallAfter(self.info, f"place_symbol error: {e.read().decode('utf-8','ignore')}")
                except Exception as e:
                    wx.CallAfter(self.info, f"place_symbol error: {e}")

            threading.Thread(target=work, daemon=True).start()

        def on_list(self, _evt):
            def work():
                try:
                    items = _get_json("/symbols/list")
                    wx.CallAfter(self.info, f"{len(items)} symbols:")
                    for s in items[-12:]:
                        wx.CallAfter(self.info, f" - {s.get('ref')}  {s.get('lib_id')}  at={s.get('at')}  uuid={s.get('uuid')}")
                except Exception as e:
                    wx.CallAfter(self.info, f"symbols/list error: {e}")
            threading.Thread(target=work, daemon=True).start()

        def on_reload(self, _evt):
            if _win_reload_eeschema(self._current_path()):
                self.info("Reload/Revert command sent to Eeschema.")
            else:
                self.info("Could not find Eeschema window. Use File→Revert or File→Reload.")

        def on_scan(self, _evt):
            def work():
                try:
                    r = _post_json("/libs/scan", {})
                    wx.CallAfter(self.info, f"Indexed {r.get('count')} symbols.")
                except Exception as e:
                    wx.CallAfter(self.info, f"libs/scan error: {e}")
            threading.Thread(target=work, daemon=True).start()

        def on_search(self, _evt):
            q = self.txt_query.GetValue().strip()
            def work():
                try:
                    qs = "?q=" + urllib.parse.quote(q) if q else ""
                    items = _get_json("/libs/search" + qs)
                    wx.CallAfter(self._fill_results, items)
                    wx.CallAfter(self.info, f"Search '{q}' → {len(items)} hits (showing up to 250).")
                except Exception as e:
                    wx.CallAfter(self.info, f"libs/search error: {e}")
            threading.Thread(target=work, daemon=True).start()

        def on_pick_result(self, evt):
            idx = evt.GetIndex()
            lib_id = self.lst_results.GetItemText(idx, 0)
            if lib_id:
                self.txt_lib.SetValue(lib_id)
                self.info(f"Selected: {lib_id}")

else:
    class AIPanel:  # stub outside KiCad
        def __init__(self, *a, **k): pass

# ---------- Action plugin ----------
class AICompanion(pcbnew.ActionPlugin):  # type: ignore[attr-defined]
    def defaults(self):
        self.name = "AI Companion (Schematic)"
        self.category = "AI"
        self.description = "Edit schematic via FastAPI (GUI-only)"

    def Run(self):
        if wx is None:
            print("This plugin must be run inside KiCad (wx not available).")
            return
        frame = wx.Frame(None, title=self.name, size=(1000, 700))
        AIPanel(frame)
        frame.Show(True)

AICompanion().register()

# Optional standalone runner for quick testing outside KiCad
if __name__ == "__main__":
    if wx is None:
        print("wx not available — run inside KiCad to open the panel.")
        sys.exit(0)
    app = wx.App(False)
    frame = wx.Frame(None, title="AI Companion (Standalone)", size=(1000, 700))
    AIPanel(frame)
    frame.Show(True)
    app.MainLoop()
