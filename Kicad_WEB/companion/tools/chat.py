# companion/tools/chat.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import os
import httpx
from dotenv import load_dotenv
import re

router = APIRouter(tags=["chat"])

class ChatInput(BaseModel):
    message: str

# Load .env (if present)
load_dotenv()

# --- Gemini config ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-1.5-flash:generateContent"
)

# --- Simple MPN detection and family → file guess ---
MPN_RE = re.compile(r"\b([A-Z0-9]{6,})\b", re.IGNORECASE)

def guess_kicad_symbol_filename(mpn: str) -> Optional[str]:
    """
    Minimal mapping MPN -> likely KiCad symbols file.
    Extend as needed:
      - STM32H7* -> MCU_ST_STM32H7.kicad_sym
      - STM32F4* -> MCU_ST_STM32F4.kicad_sym
      - etc.
    """
    up = mpn.upper()
    if up.startswith("STM32C0"):
        return "MCU_ST_STM32C0.kicad_sym"
    return None

# Helper: call internal tool endpoints
async def call_tool(method: str, path: str, json: Dict[str, Any] | None = None):
    url = os.environ.get("BACKEND_SELF_URL", "http://localhost:8000") + path
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await (client.post(url, json=json) if method.upper() == "POST" else client.get(url))
        r.raise_for_status()
        return r.json()

# Gemini tool declarations (function calling)
TOOLS_DECL = {
    "functionDeclarations": [
        {
            "name": "search_parts",
            "description": "Search for parts/boards by text query. Use for discovering whether a PCB/MPN exists online and to get candidate MPNs with vendor links.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": {"type": "STRING"},
                    "filters": {"type": "OBJECT"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "fetch_url",
            "description": (
                "Fetch content from a URL. For KiCad official libraries, always use RAW links: "
                "kicad-symbols (schematic .kicad_sym), kicad-footprints (.kicad_mod), "
                "kicad-packages3D (3D models). Convert blob URLs to raw before fetching."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {"url": {"type": "STRING"}},
                "required": ["url"]
            }
        },
        {
            "name": "save_project_file",
            "description": "Save text content into the project folder (e.g., ProjectSymbols/*.kicad_sym or footprints.pretty/*.kicad_mod).",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "path": {"type": "STRING"},
                    "content": {"type": "STRING"}
                },
                "required": ["path", "content"]
            }
        },
        {
            "name": "get_kicad_assets",
            "description": "Get distributor-provided symbol & footprint for an MPN. If Digi-Key login is required, return login_required.",
            "parameters": {
                "type": "OBJECT",
                "properties": {"mpn": {"type": "STRING"}},
                "required": ["mpn"]
            }
        },
        {
            "name": "add_symbol_to_project",
            "description": "Merge a .kicad_sym entry/file into the project's symbol library.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "symbol_kicad_sym": {"type": "STRING"},
                    "library_name": {"type": "STRING"}
                },
                "required": ["symbol_kicad_sym", "library_name"]
            }
        },
        {
            "name": "add_footprint_to_project",
            "description": "Write a .kicad_mod footprint into footprints.pretty and update fp-lib-table.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "footprint_kicad_mod": {"type": "STRING"},
                    "footprint_name": {"type": "STRING"}
                },
                "required": ["footprint_kicad_mod", "footprint_name"]
            }
        },
        {
            "name": "generate_pinout_excel",
            "description": "Generate a pinout spreadsheet for a given MPN (pin names, numbers, types, and suggested nets).",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "mpn": {"type": "STRING"},
                    "datasheet_url": {"type": "STRING"}
                },
                "required": ["mpn"]
            }
        }
    ]
}

# Known KiCad library raw bases
KI_SYMBOLS_RAW = "https://gitlab.com/kicad/libraries/kicad-symbols/-/raw/master/"
KI_FOOTPRINTS_RAW = "https://gitlab.com/kicad/libraries/kicad-footprints/-/raw/master/"
KI_PACKAGES3D_RAW = "https://gitlab.com/kicad/libraries/kicad-packages3D/-/raw/master/"

def to_raw_gitlab(url: str) -> str:
    """Convert common GitLab blob URL formats to raw."""
    if "/-/blob/" in url:
        return url.replace("/-/blob/", "/-/raw/")
    if "/blob/" in url:
        return url.replace("/blob/", "/raw/")
    return url

def build_kicad_raw_url(file_name: str, kind: str = "symbol") -> str:
    """
    Build a deterministic raw URL for a known KiCad library file.
    kind: 'symbol' -> kicad-symbols, 'footprint' -> kicad-footprints, '3d' -> kicad-packages3D
    """
    if kind == "symbol":
        base = KI_SYMBOLS_RAW
    elif kind == "footprint":
        base = KI_FOOTPRINTS_RAW
    else:
        base = KI_PACKAGES3D_RAW
    return base + file_name.lstrip("/")

# Helper: extract function call from Gemini response
def extract_function_call(resp_json: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    try:
        cand = resp_json["candidates"][0]
        parts = cand["content"]["parts"]
        for p in parts:
            if "functionCall" in p:
                fc = p["functionCall"]
                return fc.get("name"), fc.get("args", {}) or {}
    except Exception:
        pass
    return None, {}

# Helper: extract first text response
def extract_text(resp_json: Dict[str, Any]) -> str:
    return (
        resp_json.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
    ) or ""

SYSTEM_PROMPT = (
    "You are a helpful KiCad CAD assistant.\n\n"
    "Primary goals (in order):\n"
    "1) If the user describes functional requirements (e.g., low power, high compute speed, I/O), "
    "suggest appropriate MCU/SoC/PCB families and concrete boards. Compare energy efficiency, performance, "
    "memory, price, ecosystem/tooling, and availability. Prefer concise, bulleted recommendations with trade-offs.\n\n"
    "2) If the user provides a PCB or MPN (e.g., 'STM32C011F4'), try to verify it exists online and then fetch official KiCad assets:\n"
    "   - Symbols (.kicad_sym): https://gitlab.com/kicad/libraries/kicad-symbols\n"
    "   - Footprints (.kicad_mod): https://gitlab.com/kicad/libraries/kicad-footprints\n"
    "   - 3D models: https://gitlab.com/kicad/libraries/kicad-packages3D\n"
    "   Always prefer RAW URLs ('-/raw/master/<file>'), not blob pages. If a blob URL is found, convert it to raw.\n"
    "   If the exact file name is unknown, propose likely library filenames (e.g., MCU_ST_STM32C0.kicad_sym for STM32C011F4) and attempt a fetch.\n\n"
    "3) If you cannot find the PCB/MPN online or in the official KiCad libraries, politely ask the user to confirm the exact name. "
    "Guide them to manually provide files: a schematic symbol (.kicad_sym), and optionally a footprint (.kicad_mod) or a board file (.kicad_pcb). "
    "Offer to explain the differences between .kicad_sym, .kicad_mod, and .kicad_pcb in friendly, beginner terms.\n\n"
    "Tool usage:\n"
    "- Use 'search_parts' to find MPNs/vendors when discovery is needed.\n"
    "- Use 'fetch_url' to download raw files from the official KiCad repos. Convert blob URLs to raw. "
    "If only a filename is known, ask the server to build the raw URL using known repo bases.\n"
    "- Use 'save_project_file' to write downloaded text assets into the local project folder.\n"
    "- Use 'get_kicad_assets' for distributor-provided assets; if Digi-Key login is required, notify briefly and proceed with other steps.\n\n"
    "Style: Be concise, accurate, and friendly. Use bullet points when helpful. If uncertain, ask a short clarifying question."
)

@router.post("/api/chat", operation_id="chat_endpoint")
async def chat(inp: ChatInput):
    text = inp.message.strip()
    events: List[str] = []

    # --- Exact MPN fast path: fetch KiCad symbol directly from official repo (bypasses /tools/search_parts) ---
    m = MPN_RE.search(text)
    if m:
        mpn_candidate = m.group(1).upper()
        sym_file = guess_kicad_symbol_filename(mpn_candidate)
        if sym_file:
            try:
                raw_url = build_kicad_raw_url(sym_file, kind="symbol")
                fetched = await call_tool("POST", "/tools/fetch_url", {"url": raw_url})
                content = fetched.get("content")
                if content:
                    out_path = f"ProjectSymbols/{sym_file}"
                    await call_tool("POST", "/tools/save_project_file", {
                        "path": out_path,
                        "content": content,
                    })
                    events.append(f"Downloaded KiCad symbol: {sym_file}")
                    events.append(f"Saved: {out_path}")

                    # Optional: try distributor assets for footprint/symbol merge
                    try:
                        assets = await call_tool("POST", "/tools/get_kicad_assets", {"mpn": mpn_candidate})
                        if assets.get("login_required"):
                            return {
                                "reply": (
                                    f"Found {mpn_candidate}. Official KiCad symbol saved.\n"
                                    "Digi-Key login is required to fetch verified footprints—please log in and try again."
                                ),
                                "events": events + ["Digi-Key login required to fetch footprint."]
                            }
                        if "symbol_kicad_sym" in assets:
                            await call_tool("POST", "/tools/add_symbol_to_project", {
                                "symbol_kicad_sym": assets["symbol_kicad_sym"],
                                "library_name": "ProjectSymbols"
                            })
                            events.append(f"Merged distributor symbol for {mpn_candidate} into ProjectSymbols.")
                        if "footprint_kicad_mod" in assets:
                            fp_name = assets.get("footprint_name", f"{mpn_candidate}_pkg")
                            await call_tool("POST", "/tools/add_footprint_to_project", {
                                "footprint_kicad_mod": assets["footprint_kicad_mod"],
                                "footprint_name": fp_name
                            })
                            events.append(f"Added footprint for {mpn_candidate} to footprints.pretty/")
                        return {
                            "reply": f"Found {mpn_candidate}. Official KiCad symbol saved. Footprint handled if available.",
                            "events": events
                        }
                    except Exception as e:
                        return {
                            "reply": f"Found {mpn_candidate}. Official KiCad symbol saved. Footprint step failed or was skipped.",
                            "events": events + [f"Footprint note: {e}"]
                        }

                # No content fetched
                events.append(f"Could not fetch: {raw_url}")
                return {
                    "reply": (
                        f"I tried to fetch the official KiCad symbol for {mpn_candidate} "
                        f"({sym_file}) but couldn't download it.\n"
                        "Please confirm the exact part name, or share the .kicad_sym file. "
                        "Optionally also provide a .kicad_mod or a .kicad_pcb, and I can place them into your project."
                    ),
                    "events": events
                }
            except Exception as e:
                return {
                    "reply": (
                        f"Attempted to fetch the official KiCad symbol for {mpn_candidate} ({sym_file}) but hit an error.\n"
                        "Please verify the part name or upload the .kicad_sym file. "
                        "I can also explain the differences between .kicad_sym, .kicad_mod, and .kicad_pcb if helpful."
                    ),
                    "events": events + [f"Fetch error: {type(e).__name__}: {e}"]
                }
        # If we don't have a guess for this MPN family, fall through to the normal logic below.

    # --- Rule-based fast path: common keywords (kept for responsiveness) ---
    lower = text.lower()
    if any(k in lower for k in ["stm32", "add ", "footprint", "mpn "]):
        try:
            res = await call_tool("POST", "/tools/search_parts", {"query": text})
        except Exception as e:
            # Do not crash if /tools/search_parts is broken
            return {"reply": f"Error calling search_parts: {e}", "events": []}

        events.append(f"Found {len(res.get('results', []))} candidate(s).")
        if res.get("results"):
            mpn = res["results"][0]["mpn"]
            try:
                assets = await call_tool("POST", "/tools/get_kicad_assets", {"mpn": mpn})
            except Exception as e:
                return {"reply": f"Found {mpn}, but error fetching assets: {e}", "events": events}

            if assets.get("login_required"):
                return {
                    "reply": "Digi-Key login is required to fetch verified symbols/footprints. Please log in and try again.",
                    "events": events + ["Digi-Key login required."]
                }

            try:
                await call_tool("POST", "/tools/add_symbol_to_project", {
                    "symbol_kicad_sym": assets["symbol_kicad_sym"],
                    "library_name": "ProjectSymbols"
                })
                events.append(f"Added symbol for {mpn} to ProjectSymbols.kicad_sym")
            except Exception as e:
                events.append(f"Failed to add symbol: {e}")

            fp_name = assets.get("footprint_name", f"{mpn}_pkg")
            try:
                await call_tool("POST", "/tools/add_footprint_to_project", {
                    "footprint_kicad_mod": assets["footprint_kicad_mod"],
                    "footprint_name": fp_name
                })
                events.append(f"Added footprint for {mpn} to footprints.pretty/")
            except Exception as e:
                events.append(f"Failed to add footprint: {e}")

            return {"reply": f"Done! Inserted {mpn} (symbol + footprint).", "events": events}

        return {"reply": "No parts found for your request. Try a clearer MPN or model.", "events": events}

    # --- Gemini fallback: consultant + function calling ---
    if GEMINI_API_KEY:
        try:
            initial_req = {
                "systemInstruction": {"role": "system", "parts": [{"text": SYSTEM_PROMPT}]},
                "tools": [TOOLS_DECL],
                "contents": [{"role": "user", "parts": [{"text": text}]}]
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(
                    f"{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}",
                    json=initial_req,
                    headers={"Content-Type": "application/json"}
                )
                r.raise_for_status()
                data = r.json()

                fn_name, fn_args = extract_function_call(data)
                if fn_name:
                    # Execute the requested tool via internal endpoints
                    tool_resp: Dict[str, Any] = {}
                    try:
                        if fn_name == "search_parts":
                            tool_resp = await call_tool("POST", "/tools/search_parts", {
                                "query": fn_args.get("query", text),
                                "filters": fn_args.get("filters")
                            })
                            events.append(f"Found {len(tool_resp.get('results', []))} candidates via search_parts.")

                        elif fn_name == "get_kicad_assets":
                            tool_resp = await call_tool("POST", "/tools/get_kicad_assets", {
                                "mpn": fn_args.get("mpn", "")
                            })
                            if tool_resp.get("login_required"):
                                events.append("Digi-Key login required for assets.")

                        elif fn_name == "add_symbol_to_project":
                            tool_resp = await call_tool("POST", "/tools/add_symbol_to_project", {
                                "symbol_kicad_sym": fn_args.get("symbol_kicad_sym", ""),
                                "library_name": fn_args.get("library_name", "ProjectSymbols")
                            })
                            events.append("Symbol added to ProjectSymbols.")

                        elif fn_name == "add_footprint_to_project":
                            tool_resp = await call_tool("POST", "/tools/add_footprint_to_project", {
                                "footprint_kicad_mod": fn_args.get("footprint_kicad_mod", ""),
                                "footprint_name": fn_args.get("footprint_name", "")
                            })
                            events.append("Footprint added to footprints.pretty/.")

                        elif fn_name == "generate_pinout_excel":
                            tool_resp = await call_tool("POST", "/tools/generate_pinout_excel", {
                                "mpn": fn_args.get("mpn", ""),
                                "datasheet_url": fn_args.get("datasheet_url")
                            })
                            events.append(f"Pinout created: {tool_resp.get('excel_path')}")

                        elif fn_name == "fetch_url":
                            url = to_raw_gitlab(fn_args.get("url", ""))
                            tool_resp = await call_tool("POST", "/tools/fetch_url", {"url": url})
                            content_len = len(tool_resp.get("content", "") or "")
                            events.append(f"Fetched URL ({content_len} chars).")

                        elif fn_name == "save_project_file":
                            tool_resp = await call_tool("POST", "/tools/save_project_file", {
                                "path": fn_args.get("path", ""),
                                "content": fn_args.get("content", "")
                            })
                            events.append(f"Saved file: {fn_args.get('path')}")

                        else:
                            tool_resp = {"error": f"Unknown tool: {fn_name}"}
                            events.append(tool_resp["error"])

                    except httpx.HTTPStatusError as e:
                        tool_resp = {"error": f"{fn_name} HTTP {e.response.status_code}", "body": e.response.text[:400]}
                        events.append(tool_resp["error"])
                    except Exception as e:
                        tool_resp = {"error": f"{fn_name} failed: {e}"}
                        events.append(tool_resp["error"])

                    # Provide the tool output back to Gemini for a grounded final response
                    followup_req = {
                        "systemInstruction": initial_req["systemInstruction"],
                        "tools": [TOOLS_DECL],
                        "contents": [
                            {"role": "user", "parts": [{"text": text}]},
                            {
                                "role": "tool",
                                "parts": [{
                                    "functionResponse": {
                                        "name": fn_name,
                                        "response": {"name": fn_name, "content": tool_resp}
                                    }
                                }]
                            }
                        ]
                    }

                    r2 = await client.post(
                        f"{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}",
                        json=followup_req,
                        headers={"Content-Type": "application/json"}
                    )
                    r2.raise_for_status()
                    data2 = r2.json()
                    reply2 = extract_text(data2) or "Could not parse model reply."
                    return {"reply": reply2, "events": events}

                # No tool call: plain text answer
                plain = extract_text(data) or "Could not parse model reply."
                return {"reply": plain, "events": events}

        except Exception as e:
            return {
                "reply": (
                    f"Gemini error: {e}\n"
                    "If you're looking for a specific PCB/MPN and I couldn't fetch it, please confirm the exact name. "
                    "You can also upload a .kicad_sym (schematic), and optionally a .kicad_mod or .kicad_pcb, "
                    "and I will place them into your project folder."
                ),
                "events": events
            }

    # Fallback if no LLM configured
    return {"reply": "I did not understand. Try: 'add STM32F407 + decoupling'.", "events": events}
