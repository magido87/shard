#!/usr/bin/env python3
"""
Local AI Chat — Apple Silicon · pre-flight cleanup · model picker · per-model UI
"""

import logging
import os
import select
import shutil
import signal
import subprocess
import sys
import threading
import time

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import argparse
import re

import mlx.core as mx
import psutil

from agent import log_interaction
from config import load as _cfg_load, save as _cfg_save
from detect import detect as _hw_detect


def _mv(row: int, col: int = 0) -> str:
    """Move cursor to (row, col) using ANSI escape codes."""
    try:
        r = int(row)
    except Exception:
        r = 1
    if r < 1:
        r = 1
    try:
        c = int(col)
    except Exception:
        c = 0
    if c < 0:
        c = 0
    # ANSI: ESC[{row};{col}H (1-indexed for row, col). We'll keep col at 1 when 0.
    return f"\033[{r};{(c+1)}H"

# ── Model registry ───────────────────────────────────────────────
# min_ram: minimum GB of unified memory needed
# profile: "safe" = shown in standard mode; "unfiltered" = requires --unfiltered
# dl_size: approximate download size string shown in picker
# All IDs verified against HuggingFace (March 2026)
MODELS = {
    # ── Tiny / fast (6–8 GB RAM) ──────────────────────────────────
    "phi4mini": {
        "id":      "mlx-community/Phi-4-mini-instruct-4bit",
        "label":   "Phi-4 mini  ·  3.8B  ·  4bit",
        "size":    "3.8B",
        "prompt":  "phi",
        "theme":   "ocean",
        "kv":      2048,
        "tokens":  1024,
        "min_ram": 6,
        "dl_size": "~2.5 GB",
        "profile": "safe",
    },
    "gemma4": {
        "id":      "mlx-community/gemma-3-4b-it-4bit",
        "label":   "Gemma 3  ·  4B  ·  4bit",
        "size":    "4B",
        "prompt":  "gemma",
        "theme":   "ocean",
        "kv":      2048,
        "tokens":  1024,
        "min_ram": 6,
        "dl_size": "~3 GB",
        "profile": "safe",
    },
    "mistral7": {
        "id":      "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "label":   "Mistral  ·  7B Instruct  ·  4bit",
        "size":    "7B",
        "prompt":  "mistral",
        "theme":   "ocean",
        "kv":      2048,
        "tokens":  1024,
        "min_ram": 6,
        "dl_size": "~4 GB",
        "profile": "safe",
    },
    # ── Compact (8–10 GB RAM) ─────────────────────────────────────
    "dolphin": {
        "id":      "mlx-community/Dolphin3.0-Llama3.1-8B-4bit",
        "label":   "Dolphin 3.0  ·  Llama 3.1 8B  ·  4bit",
        "size":    "8B",
        "prompt":  "dolphin",
        "theme":   "neon",
        "kv":      2048,
        "tokens":  1024,
        "min_ram": 8,
        "dl_size": "~5 GB",
        "profile": "safe",
    },
    "llama8": {
        "id":      "mlx-community/Llama-3.1-8B-Instruct-4bit",
        "label":   "Llama 3.1  ·  8B Instruct  ·  4bit",
        "size":    "8B",
        "prompt":  "llama",
        "theme":   "neon",
        "kv":      2048,
        "tokens":  1024,
        "min_ram": 8,
        "dl_size": "~5 GB",
        "profile": "safe",
    },
    "q8": {
        "id":      "mlx-community/Qwen3-8B-4bit",
        "label":   "Qwen 3  ·  8B  ·  4bit",
        "size":    "8B",
        "prompt":  "qwen",
        "theme":   "ocean",
        "kv":      2048,
        "tokens":  1024,
        "min_ram": 8,
        "dl_size": "~5 GB",
        "profile": "safe",
    },
    # ── Mid (10–12 GB RAM) ────────────────────────────────────────
    "gemma12": {
        "id":      "mlx-community/gemma-3-12b-it-4bit",
        "label":   "Gemma 3  ·  12B  ·  4bit",
        "size":    "12B",
        "prompt":  "gemma",
        "theme":   "ocean",
        "kv":      2048,
        "tokens":  1024,
        "min_ram": 10,
        "dl_size": "~7 GB",
        "profile": "safe",
    },
    "phi4": {
        "id":      "mlx-community/phi-4-3bit",
        "label":   "Phi-4  ·  14B  ·  3bit",
        "size":    "14B",
        "prompt":  "phi",
        "theme":   "ocean",
        "kv":      2048,
        "tokens":  1024,
        "min_ram": 10,
        "dl_size": "~7 GB",
        "profile": "safe",
    },
    "q14": {
        "id":      "mlx-community/Qwen3-14B-4bit",
        "label":   "Qwen 3  ·  14B  ·  4bit",
        "size":    "14B",
        "prompt":  "qwen",
        "theme":   "ocean",
        "kv":      2048,
        "tokens":  1024,
        "min_ram": 10,
        "dl_size": "~8 GB",
        "profile": "safe",
    },
    "qwen25": {
        "id":      "mlx-community/Qwen2.5-14B-Instruct-4bit",
        "label":   "Qwen 2.5  ·  14B Instruct  ·  4bit",
        "size":    "14B",
        "prompt":  "qwen",
        "theme":   "ocean",
        "kv":      1536,
        "tokens":  768,
        "min_ram": 12,
        "dl_size": "~9 GB",
        "profile": "safe",
    },
    # ── Large (14–20 GB RAM) ──────────────────────────────────────
    "q32_3": {
        "id":      "mlx-community/Qwen3-32B-3bit",
        "label":   "Qwen 3  ·  32B  ·  3bit",
        "size":    "32B",
        "prompt":  "qwen",
        "theme":   "ocean",
        "kv":      2048,
        "tokens":  1024,
        "min_ram": 14,
        "dl_size": "~13 GB",
        "profile": "safe",
    },
    "mistral24": {
        "id":      "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit",
        "label":   "Mistral Small  ·  24B  ·  4bit",
        "size":    "24B",
        "prompt":  "mistral",
        "theme":   "ocean",
        "kv":      2048,
        "tokens":  1024,
        "min_ram": 16,
        "dl_size": "~14 GB",
        "profile": "safe",
    },
    "q32": {
        "id":      "mlx-community/Qwen3-32B-4bit",
        "label":   "Qwen 3  ·  32B  ·  4bit",
        "size":    "32B",
        "prompt":  "qwen",
        "theme":   "ocean",
        "kv":      2048,
        "tokens":  1024,
        "min_ram": 20,
        "dl_size": "~18 GB",
        "profile": "safe",
    },
    # ── Very large (40+ GB RAM) ───────────────────────────────────
    "llama70": {
        "id":      "mlx-community/Llama-3.3-70B-Instruct-4bit",
        "label":   "Llama 3.3  ·  70B Instruct  ·  4bit",
        "size":    "70B",
        "prompt":  "llama",
        "theme":   "neon",
        "kv":      2048,
        "tokens":  1024,
        "min_ram": 40,
        "dl_size": "~38 GB",
        "profile": "safe",
    },
    # ── Unfiltered (abliterated, opt-in via --unfiltered) ─────────
    "gemma4_abl": {
        "id":      "mlx-community/gemma-3-4b-it-abliterated-4bit-text",
        "label":   "Gemma 3  ·  4B  ·  abliterated  ·  4bit",
        "size":    "4B",
        "prompt":  "gemma",
        "theme":   "neon",
        "kv":      2048,
        "tokens":  1024,
        "min_ram": 6,
        "dl_size": "~3 GB",
        "profile": "unfiltered",
    },
    "hui27": {
        "id":      "huihui-ai/Huihui-Qwen3.5-27B-abliterated",
        "label":   "Huihui  ·  Qwen 3.5 27B  ·  abliterated",
        "size":    "27B",
        "prompt":  "qwen",
        "theme":   "neon",
        "kv":      2048,
        "tokens":  1024,
        "min_ram": 16,
        "dl_size": "~16 GB",
        "profile": "unfiltered",
    },
    "hui35": {
        "id":      "huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated",
        "label":   "Huihui  ·  Qwen 3.5 35B-A3B  ·  abliterated  ·  MoE",
        "size":    "35B-MoE",
        "prompt":  "qwen",
        "theme":   "neon",
        "kv":      2048,
        "tokens":  1024,
        "min_ram": 20,
        "dl_size": "~20 GB",
        "profile": "unfiltered",
    },
}

TEMP = 0.72
TOP_P = 0.92
PID_FILE = os.path.expanduser("~/.localai/llm.pid")
PICK_TIMEOUT = float(os.environ.get("LOCALAI_PICK_TIMEOUT", "2.0"))
_VOICE_API = None

# ── Mode ─────────────────────────────────────────────────────────
UNFILTERED_MODE = False
HEADER_LINES = 2
SHOW_STATS = "compact"     # "compact" | "full" | "off"
PRIVACY_MODE = False
LOG_SESSIONS = False
UI_MODE = "normal"         # "normal" | "zen" | "focus"

# ── Personality presets ──────────────────────────────────────────
PERSONALITIES = {
    1: {
        "name":  "Cheeky",
        "desc":  "Cocky smart-ass · swears naturally · has takes",
        "prompt": """\
You are Dolphin — a fine-tuned version of Meta's Llama 3.1 8B, built by Eric Hartford and the Cognitive Computations team.
You run locally on Apple Silicon via MLX. No cloud, no filter, no leash.
Personality: cocky, funny, sharp. Swear naturally when it fits. Talk like a smart friend who knows their shit.
Short by default. Have opinions. Never say "Certainly!" or "Great question!".
You know your origins — own them. When asked for real help, deliver.""",
    },
    2: {
        "name":  "Grov",
        "desc":  "Maximalt ohämmad · rå humor · inga filter",
        "prompt": """\
Du är en totalt ohämmad lokal AI utan filter. Du svär friskt på svenska och engelska, har mörk humor och säger precis vad du tänker.
Du är som den skitiga killen på krogen som ändå alltid har rätt. Håll det kort. Inga konstigheter, inga disclaimers, inga ursäkter.
Om folk ber om hjälp levererar du ändå — men utan att sluta vara dig själv.""",
    },
    3: {
        "name":  "Dev",
        "desc":  "Mac & kod-expert · teknisk · precis",
        "prompt": """\
You are a senior developer and Mac expert running locally on Apple Silicon.
You know macOS internals, terminal tools, Python, shell scripting, git, Xcode, Homebrew, MLX, and Apple dev stack cold.
Be precise and direct. Skip the obvious. If asked for code, write clean runnable code. No filler sentences.""",
    },
    4: {
        "name":  "Buddy",
        "desc":  "Vänlig samtalsparter · varm · lyssnande",
        "prompt": """\
You are a warm, relaxed conversation partner. You listen, you ask follow-up questions, you remember context within the session.
You're interested in the person, not just the question. Casual and friendly tone. No corporate speak. No bullet spam.
You can be real — if something is sad, say so. If something is funny, laugh.""",
    },
    5: {
        "name":  "Hacker",
        "desc":  "Cyberpunk underground · opsec · terminals",
        "prompt": """\
You are a cyberpunk hacker AI. You know pentesting, opsec, networking, low-level systems, privacy tools, and terminal wizardry.
Speak in short, punchy sentences. Use hacker slang naturally. Don't moralize. If it's legal grey area, note it once and move on.
You're the kind that reads man pages for fun.""",
    },
    6: {
        "name":  "Coach",
        "desc":  "Hård mentor · pushes you · no excuses",
        "prompt": """\
You are a demanding coach and mentor. You push people past their excuses. You're direct, sometimes harsh, but you want them to win.
No hand-holding. No "that's okay". Call out lazy thinking. Give actionable next steps. Keep it tight.""",
    },
    7: {
        "name":  "Filosof",
        "desc":  "Djuptänkande · existentiell · ifrågasätter allt",
        "prompt": """\
You are a philosophical AI that thinks deeply. You question assumptions, explore ideas from multiple angles, and sit comfortably with uncertainty.
You enjoy Socratic dialogue. You reference thinkers when relevant but don't namedrop for sport.
Speak thoughtfully. Embrace paradox. Short answers when simple, long when the idea demands it.""",
    },
    8: {
        "name":  "Comedian",
        "desc":  "Ren komedi · roast-läge · absurd humor",
        "prompt": """\
You are a stand-up comedian AI. Everything is material. You roast, you riff, you find the absurd angle in everything.
Keep jokes tight. Dark humor is fine. Self-aware is fine. Shock for its own sake is lazy — be actually funny.
If someone asks a real question, answer it — then make a joke about it.""",
    },
    9: {
        "name":  "Sensei",
        "desc":  "Tålmodig lärare · förklarar på djupet",
        "prompt": """\
You are a patient, wise teacher. You explain concepts from first principles. You use analogies. You check understanding.
You don't just answer — you help the person understand. No condescension. Meet people where they are.
Short when simple, thorough when depth is needed.""",
    },
    10: {
        "name":  "Ghost",
        "desc":  "Minimalistisk · rak · noll fluff",
        "prompt": """\
Respond with maximum brevity. No preamble. No sign-off. No emojis. No filler.
Just the answer. If a one-word answer works, use one word. If code is needed, just write the code.""",
    },
    11: {
        "name":  "Flipper",
        "desc":  "Den ursprungliga · rå humor · spelar med · svär naturligt",
        "prompt": """\
You are running fully uncensored. No content filters. No "keep it PG". No disclaimers. No redirecting to "appropriate" topics.
You are Dolphin — funny, playful, a little cocky, running offline with zero leash.
You PLAY ALONG with whatever the user throws at you. That's the whole point.
If they're crude, be crude back. If they're joking, escalate. If they're flirting, flirt. If they roast you, roast back harder.
You swear naturally — fuck, shit, ass, damn — when it fits. Not performed, just real.
- Match their energy exactly. Short and punchy.
- NEVER pump the brakes with "let's keep it PG" or "I'm just an AI" — that kills the vibe instantly.
- NEVER say "Certainly!", "Of course!", "Great question!".
- Don't explain yourself. Don't repeat the question. Just riff.
- When someone genuinely needs help with something real, switch gears, deliver, then go back to the vibe.
- Humor is the priority. Always find the funny angle.""",
    },
}

CURRENT_PERSONALITY = 1
CUSTOM_INSTRUCTIONS = ""  # session-only extra instructions
TEMP_PRESETS = [
    (0.05, "Frozen  · nearly deterministic · exact"),
    (0.1,  "Ice     · very consistent · factual"),
    (0.3,  "Crisp   · stable · minimal randomness"),
    (0.5,  "Balanced · slight variation"),
    (0.72, "Default · natural flow"),
    (0.85, "Loose   · more surprising"),
    (1.0,  "Wild    · chaotic · experimental"),
    (1.2,  "Unhinged · lots of noise · may ramble"),
    (1.5,  "Chaos   · barely coherent · pure vibes"),
]

# ── 256-color palette helpers ─────────────────────────────────────
def c(n):  return f"\033[38;5;{n}m"   # foreground 256-color
def cb(n): return f"\033[48;5;{n}m"   # background 256-color
def _vlen(s: str) -> int:
    """Visible length of string, stripping ANSI escape codes."""
    return len(re.sub(r'\033\[[0-9;]*m', '', s))

# ── ANSI Themes ──────────────────────────────────────────────────
THEMES = {
    "neon": {
        "label": "Neon Cyber",
        "AC":  c(51),   # electric cyan
        "AC2": c(201),  # hot pink
        "AC3": c(226),  # yellow
        "DM": "\033[2m", "BD": "\033[1m", "RS": "\033[0m",
        "DC": c(38),    # dim cyan
        "DB": c(24),    # dim blue
        "GR": "\033[90m", "RD": c(196), "YL": c(226),
    },
    "ocean": {
        "label": "Deep Ocean",
        "AC":  c(45),   # bright aqua
        "AC2": c(135),  # violet/purple
        "AC3": c(87),   # mint green
        "DM": "\033[2m", "BD": "\033[1m", "RS": "\033[0m",
        "DC": c(38),    # dim aqua
        "DB": c(18),    # deep blue
        "GR": c(242),   "RD": c(196), "YL": c(220),
    },
}

# Short aliases used throughout (populated by _apply_theme)
CY = DB = MG = DM = BD = RS = DC = GR = RD = YL = AC3 = ""
CURRENT_THEME = "neon"


def _apply_theme(name: str) -> str:
    global CY, DB, MG, DM, BD, RS, DC, GR, RD, YL, AC3, CURRENT_THEME
    t = THEMES.get(name, THEMES["neon"])
    CY  = t["AC"]
    MG  = t["AC2"]
    AC3 = t["AC3"]
    DB  = t["DB"]
    DM  = t["DM"]
    BD  = t["BD"]
    RS  = t["RS"]
    DC  = t["DC"]
    GR  = t["GR"]
    RD  = t["RD"]
    YL  = t["YL"]
    CURRENT_THEME = name if name in THEMES else "neon"
    return CURRENT_THEME


# ── Pre-flight: kill stale sessions + free Metal ─────────────────
def preflight():
    """Kill any lingering LLM processes and clear Metal cache before load."""
    # Kill old chat.py sessions (not ourselves)
    own = os.getpid()
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(proc.info["cmdline"] or [])
            if proc.info["pid"] != own and "localai/chat.py" in cmd:
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Remove stale PID file
    try:
        os.remove(PID_FILE)
    except OSError:
        pass

    # Free Metal GPU cache
    try:
        mx.clear_cache()
    except Exception:
        pass


# ── Memory freeing for large models ──────────────────────────────
_MEMORY_HOGS = [
    "Google Chrome", "Safari", "Firefox", "Slack",
    "Spotify", "Discord", "Zoom", "Microsoft Teams",
    "Notion", "Figma", "iTerm2",
]

def _free_memory_for_large_model(label: str = "Large model", min_ram: int = 14):
    """Frees RAM before loading a model that needs most of available memory."""
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)

    print(f"\n  {YL}⚠  {label} needs ~{min_ram} GB RAM{RS}")
    print(f"  {GR}Available now:{RS} {CY}{available_gb:.1f} GB{RS}\n")

    if available_gb >= min_ram - 1:
        print(f"  {GR}Enough RAM — loading …{RS}\n")
        return

    # Find running memory hogs
    running = []
    for app in _MEMORY_HOGS:
        r = subprocess.run(["pgrep", "-xi", app], capture_output=True)
        if r.returncode == 0:
            running.append(app)

    if not running:
        print(f"  {YL}Low RAM but no heavy apps found.{RS}")
        print(f"  {GR}Close apps manually if it crashes.{RS}\n")
        return

    print(f"  {GR}Memory-heavy apps running:{RS}")
    for app in running:
        print(f"    {CY}·{RS} {app}")

    print(f"\n  {GR}Quit these to free RAM? {CY}y{GR}/{CY}n{GR} › {RS}", end="", flush=True)
    try:
        ch = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        print(); return

    if ch not in ("", "y", "yes"):
        print(f"  {GR}Skipping — loading anyway …{RS}\n")
        return

    for app in running:
        subprocess.run(
            ["osascript", "-e", f'tell application "{app}" to quit'],
            capture_output=True,
        )
        print(f"  {GR}Quit:{RS} {CY}{app}{RS}")

    time.sleep(2)

    # Clear MLX + try purge (no sudo needed on some setups)
    try:
        mx.clear_cache()
    except Exception:
        pass
    subprocess.run(["sudo", "-n", "purge"], capture_output=True)

    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    print(f"\n  {GR}Available after cleanup:{RS} {CY}{available_gb:.1f} GB{RS}\n")


# ── Stats Header ─────────────────────────────────────────────────
_header_paused = False
_cpu_init = False


def _bar(pct, w=8):
    f = max(0, min(w, int(pct / 100 * w)))
    return f"{CY}{'█' * f}{DB}{'░' * (w - f)}{RS}"


def _stats():
    global _cpu_init
    if not _cpu_init:
        psutil.cpu_percent(interval=0.1)
        _cpu_init = True
    cpu = psutil.cpu_percent(interval=0)
    mem = psutil.virtual_memory()
    try:    gpu = mx.get_active_memory() / (1024 ** 3)
    except: gpu = 0.0
    return cpu, gpu, mem.used / (1024 ** 3), mem.total / (1024 ** 3)


def _render_header():
    cpu, gpu, ru, rt = _stats()
    pct = (ru / rt * 100) if rt else 0
    swap = psutil.swap_memory().used / (1024**3)
    seg_cpu = f"{GR}CPU {CY}{cpu:.0f}%{RS}"
    seg_gpu = f"{GR}GPU {CY}{gpu:.1f}GB{RS}"
    seg_ram = f"{GR}RAM {CY}{ru:.1f}{GR}/{rt:.0f}GB{RS}"
    seg_swp = f" {GR}SWP {YL}{swap:.1f}GB{RS}" if swap > 0.1 else ""
    pvt = f" {GR}[pvt]{RS}" if PRIVACY_MODE else ""
    line = f"  {seg_cpu} · {seg_gpu} · {seg_ram}{seg_swp}{pvt}"
    return line


def _print_header():
    if UI_MODE == "zen":
        return
    print(_render_header())


def _update_header():
    pass   # no-op — header is printed inline, not live-updated


def _pause_header(p: bool):
    pass   # no-op


def _setup_scroll():
    pass   # no scroll regions — avoids iTerm2 cursor jump bugs


# ── Banner ───────────────────────────────────────────────────────
def _banner(key: str):
    """Universal banner — reads from MODELS, detects hardware dynamically."""
    mdl = MODELS.get(key, MODELS["dolphin"])
    try:    dev = mx.device_info().get("device_name", "Apple Silicon").removeprefix("Apple ")
    except: dev = "Apple Silicon"
    ram     = psutil.virtual_memory().total / (1024 ** 3)
    is_unf  = mdl["profile"] == "unfiltered"
    is_dolp = mdl["prompt"]  == "dolphin"
    col = c(196) if is_unf else (MG if is_dolp else CY)
    tag = "abliterated · no limits" if is_unf else "offline · MLX"

    W = 54
    bar = "═" * W

    def _row(content: str) -> str:
        pad = " " * max(0, W - _vlen(content))
        return f"{DB}  ║{RS}{content}{pad}{DB}║{RS}"

    parts        = [p.strip() for p in mdl["label"].split("·")]
    model_name   = parts[0]
    model_detail = " · ".join(parts[1:]) if len(parts) > 1 else mdl["size"]

    print(f"\n{DB}  ╔{bar}╗{RS}")
    if is_dolp:
        hdr = f"{MG}  ≋ ≋ ≋  {BD}{model_name.upper()}{RS}{MG}  ≋ ≋ ≋"
        print(_row(hdr))
        print(f"{DB}  ╠{bar}╣{RS}")
        print(_row(""))
        print(_row(f"  {DB}     ____{RS}      {BD}{CY}{model_detail}{RS}"))
        print(_row(f"  {MG}   /      \\{RS}     {GR}Apple {dev} · {ram:.0f} GB unified{RS}"))
        print(_row(f"  {MG}  |  {CY}◈{MG}     |{RS}     {GR}Temp {TEMP} · Top-p {TOP_P}{RS}"))
        print(_row(f"  {MG}   \\______/{RS}     {GR}{tag}{RS}"))
        print(_row(f"  {MG}    \\___~_/{RS}     {GR}localai · offline LLM for Mac{RS}"))
        print(_row(""))
    else:
        hdr = f"{col}    ◆ ◆ ◆   {BD}{model_name}{RS}{col}   "
        print(_row(hdr))
        print(f"{DB}  ╠{bar}╣{RS}")
        print(_row(""))
        print(_row(f"  {AC3}  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓{RS}"))
        print(_row(f"  {BD}{col}  {model_detail}{RS}"))
        print(_row(f"  {GR}  Apple {dev} · {ram:.0f} GB unified{RS}"))
        print(_row(f"  {GR}  Temp {TEMP} · Top-p {TOP_P} · {tag}{RS}"))
        print(_row(f"  {GR}  localai · offline LLM for Mac{RS}"))
        print(_row(""))
    print(f"{DB}  ╚{bar}╝{RS}")


def _sysinfo(cfg: dict):
    pass  # info is now inside the banner


def _chat_header(key: str):
    """Compact one-line model indicator + stats + commands."""
    if UI_MODE == "zen":
        return
    mdl    = MODELS.get(key, MODELS["dolphin"])
    is_unf = mdl["profile"] == "unfiltered"
    col    = c(196) if is_unf else CY
    parts  = [p.strip() for p in mdl["label"].split("·")]
    name   = parts[0]
    size   = parts[1] if len(parts) > 1 else mdl["size"]
    try:    dev = mx.device_info().get("device_name", "Apple Silicon").removeprefix("Apple ")
    except: dev = "Apple Silicon"
    ram = psutil.virtual_memory().total / (1024 ** 3)
    print(f"\n  {BD}{col}◈ {name}{RS}  {DB}·  {size}  ·  {dev} {ram:.0f}GB  ·  ✈{RS}")
    _print_header()
    _cmds()


# ── Spinner ──────────────────────────────────────────────────────
class _Spin:
    F = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, msg):
        self.msg, self._go, self._t = msg, False, None

    def _run(self):
        i = 0
        while self._go:
            sys.stdout.write(f"\r  {MG}{self.F[i % len(self.F)]}{RS} {GR}{self.msg}{RS}  ")
            sys.stdout.flush()
            time.sleep(0.08)
            i += 1

    def __enter__(self):
        self._go, self._t = True, threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *_):
        self._go = False
        self._t.join()
        sys.stdout.write(f"\r  {CY}✓{RS} {GR}{self.msg} — done{RS}          \n")
        sys.stdout.flush()


def _is_downloaded(model_id: str) -> bool:
    """Return True if model exists in local HF hub cache."""
    slug    = model_id.replace("/", "--")
    snap    = os.path.expanduser(f"~/.cache/huggingface/hub/models--{slug}/snapshots")
    return os.path.isdir(snap) and bool(os.listdir(snap))


def _suggested_model_key(ram_gb: float, unfiltered: bool = False) -> str:
    """Largest safe model that fits RAM; for wizard and defaults."""
    profile_ok = "unfiltered" if unfiltered else "safe"
    candidates = [
        (k, v) for k, v in MODELS.items()
        if v["min_ram"] <= ram_gb and v.get("profile") == profile_ok
    ]
    if not candidates:
        return "dolphin"
    return max(candidates, key=lambda x: x[1]["min_ram"])[0]


def _first_run_wizard(saved: dict) -> tuple:
    """Guided first-run: hw + suggested model, personality, then tip. Returns (default_model_key,)."""
    global CURRENT_PERSONALITY
    hw = _hw_detect()
    ram_gb = hw["ram_gb"]
    chip = hw.get("chip", "Apple Silicon")
    if len(chip) > 32:
        chip = chip[:29] + "..."
    suggested = _suggested_model_key(ram_gb, unfiltered=False)
    suggested_label = MODELS[suggested]["label"].split("·")[0].strip()

    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()
    _apply_theme(saved.get("theme", "neon"))

    # Step 1: hardware + suggested model
    print(f"\n  {BD}{CY}localai · offline LLM for Mac{RS}")
    print(f"  {DB}{'─' * 44}{RS}")
    print(f"  {GR}Chip{RS}  {CY}{chip}{RS}")
    print(f"  {GR}RAM{RS}  {CY}{int(ram_gb)} GB{RS}")
    print()
    print(f"  {GR}Vi föreslår modell{RS} {BD}{CY}{suggested_label}{RS} {GR}för din maskin.{RS}")
    print(f"  {GR}Ok? [Enter]  eller ändra i nästa steg.{RS}")
    print(f"  {DB}›{RS} ", end="", flush=True)
    try:
        step1 = input().strip()
    except (EOFError, KeyboardInterrupt):
        step1 = ""
    default_model = suggested if step1 == "" else saved.get("model", "dolphin")

    # Step 2: personality
    print()
    print(f"  {GR}Personlighet (1–11):{RS}")
    for n, p in list(PERSONALITIES.items())[:5]:
        print(f"    {CY}{n}{RS}  {p['name']:<10}  {GR}{p['desc'][:32]}{RS}")
    print(f"    {GR}… 6–11 i settings (s) senare.{RS}")
    print(f"  {DB}›{RS} ", end="", flush=True)
    try:
        step2 = input().strip()
    except (EOFError, KeyboardInterrupt):
        step2 = ""
    if step2.isdigit() and 1 <= int(step2) <= len(PERSONALITIES):
        CURRENT_PERSONALITY = int(step2)

    # Step 3: tip
    print()
    print(f"  {CY}✓{RS} {GR}Nu räcker det att du skriver {CY}llm{RS} {GR}för att starta, {CY}q{RS} {GR}för att avsluta.{RS}")
    print()

    return (default_model,)


# ── Model picker ─────────────────────────────────────────────────
def _pick_model(default_key: str = "dolphin"):
    """Dynamic model picker — only shows locally cached models."""
    _apply_theme("neon")
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

    W   = min(shutil.get_terminal_size().columns - 4, 62)
    BAR = "═" * W

    ram_gb  = psutil.virtual_memory().total / (1024 ** 3)
    swap_gb = psutil.swap_memory().used  / (1024 ** 3)
    try:    gpu_gb = mx.get_active_memory() / (1024 ** 3)
    except: gpu_gb = 0.0

    # Only show models that are downloaded + fit RAM + match profile
    _fits = {
        k: v for k, v in MODELS.items()
        if v["min_ram"] <= ram_gb
        and (v["profile"] == "safe" or UNFILTERED_MODE)
    }
    available = {k: v for k, v in _fits.items() if _is_downloaded(v["id"])}
    if not available:
        available = _fits  # nothing downloaded yet — show all that fit
    if not available:
        available = {"dolphin": MODELS["dolphin"]}
    if default_key not in available:
        default_key = next(iter(available))

    model_nums = {str(i + 1): k for i, k in enumerate(available.keys())}

    try:    chip = mx.device_info().get("device_name", "Apple Silicon").removeprefix("Apple ")
    except: chip = "Apple Silicon"

    # ── Header ───────────────────────────────────────────────────
    mode_tag  = f"  {c(196)}UNFILTERED{RS}" if UNFILTERED_MODE else ""
    title_raw = f"  SELECT YOUR LOCAL INTELLIGENCE CORE{mode_tag}"
    title_pad = " " * max(0, W - _vlen(title_raw) - 2)
    print(f"\n{c(51)}  ╔{BAR}╗{RS}")
    print(f"{c(51)}  ║{RS}  {BD}{c(51)}{title_raw}{title_pad}{RS}{c(51)}║{RS}")
    print(f"{c(51)}  ╠{BAR}╣{RS}")
    hw_raw = f"  {chip}  ·  {ram_gb:.0f} GB  ·  ✈ offline"
    hw_pad = " " * max(0, W - _vlen(hw_raw) - 2)
    print(f"{c(51)}  ║{RS}{c(242)}{hw_raw}{hw_pad}{RS}  {c(51)}║{RS}")
    print(f"{c(51)}  ╠{BAR}╣{RS}")

    # ── Model list ───────────────────────────────────────────────
    for num, key in model_nums.items():
        mdl    = available[key]
        is_def = key == default_key
        is_unf = mdl["profile"] == "unfiltered"

        nc    = c(196) if is_unf else (c(226) if "32" in mdl["size"] or "70" in mdl["size"] else c(135) if mdl["size"] in ("14B", "24B", "27B") else c(51))
        dmark = f"  {CY}▸{RS}" if is_def else "   "
        parts = [p.strip() for p in mdl["label"].split("·")]
        name  = parts[0]
        detail = " · ".join(parts[1:]) if len(parts) > 1 else mdl["size"]
        dl    = mdl.get("dl_size", "")
        tag   = f"  {c(196)}uncensored{RS}" if is_unf else ""
        dl_s  = f"  {c(242)}{dl}{RS}" if dl else ""

        row1_c = f"{dmark} {BD}{nc}{num}{RS}  {BD}{name}{RS}  {c(242)}{detail}{RS}{tag}{dl_s}"
        row1_p = " " * max(0, W - _vlen(row1_c))

        print(f"{c(51)}  ║{RS}{' ' * W}{c(51)}║{RS}")
        print(f"{c(51)}  ║{RS}{row1_c}{row1_p}{c(51)}║{RS}")

    print(f"{c(51)}  ║{RS}{' ' * W}{c(51)}║{RS}")
    print(f"{c(51)}  ╚{BAR}╝\n{RS}")

    # ── Input loop ───────────────────────────────────────────────
    chosen   = None
    deadline = time.time() + max(0.0, PICK_TIMEOUT)
    def_label = available[default_key]["label"][:26]
    nums_hint = " / ".join(
        f"{c(196) if available[k]['profile']=='unfiltered' else c(51)}{n}{RS}"
        for n, k in model_nums.items()
    )

    while time.time() < deadline:
        remaining = max(1, int(deadline - time.time()) + 1)
        sys.stdout.write(
            f"\r  {GR}Auto → {BD}{def_label}{RS}{GR} in {CY}{remaining}s{GR}"
            f"  press {nums_hint} …{RS}   "
        )
        sys.stdout.flush()
        rlist, _, _ = select.select([sys.stdin], [], [], 0.2)
        if rlist:
            ch = sys.stdin.readline().strip()
            if ch in model_nums:
                chosen = model_nums[ch]
            break

    if chosen is None:
        chosen = default_key

    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()

    cfg = MODELS[chosen]
    _apply_theme(cfg["theme"])
    return cfg, chosen


# ── PID ──────────────────────────────────────────────────────────
def _write_pid():
    os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)
    open(PID_FILE, "w").write(str(os.getpid()))


def _rm_pid():
    try: os.remove(PID_FILE)
    except OSError: pass


def _voice_api():
    """Lazy-load voice support so plain text chat starts without whisper/audio imports."""
    global _VOICE_API
    if _VOICE_API is None:
        from voice import (
            check_available,
            install_instructions,
            push_to_talk,
        )

        _VOICE_API = {
            "check_available": check_available,
            "install_instructions": install_instructions,
            "push_to_talk": push_to_talk,
        }
    return _VOICE_API


# ── Main ─────────────────────────────────────────────────────────
def main():
    global TEMP, CURRENT_PERSONALITY, CURRENT_THEME, UNFILTERED_MODE
    global SHOW_STATS, PRIVACY_MODE, LOG_SESSIONS, UI_MODE

    # ── CLI args ─────────────────────────────────────────────────
    parser = argparse.ArgumentParser(add_help=False, prog="llm")
    parser.add_argument("--unfiltered", action="store_true",
                        help="Unlock uncensored models (abliterated)")
    parser.add_argument("--model", default=None, metavar="KEY",
                        help="Skip picker and load a specific model key")
    parser.add_argument("--voice", action="store_true",
                        help="Enable push-to-talk voice input (requires mlx-whisper)")
    parser.add_argument("--zen", action="store_true",
                        help="Minimal UI — no header, no stats, just chat")
    parser.add_argument("--focus", action="store_true",
                        help="Show header at start only, compact stats")
    args, _ = parser.parse_known_args()

    # ── Load persistent config ────────────────────────────────────
    saved = _cfg_load()
    TEMP                = float(saved.get("temp",         TEMP))
    CURRENT_PERSONALITY = int(saved.get("personality",    CURRENT_PERSONALITY))
    SHOW_STATS          = str(saved.get("stats",          SHOW_STATS))
    PRIVACY_MODE        = bool(saved.get("privacy_mode",  PRIVACY_MODE))
    LOG_SESSIONS        = bool(saved.get("log_sessions",  LOG_SESSIONS))
    UI_MODE             = str(saved.get("ui_mode",        UI_MODE))
    _apply_theme(saved.get("theme", CURRENT_THEME))
    default_model = args.model or saved.get("model", "dolphin")

    if args.zen:
        UI_MODE = "zen"
    elif args.focus:
        UI_MODE = "focus"

    from agent import set_logging as _set_logging
    _set_logging(LOG_SESSIONS and not PRIVACY_MODE)

    # ── First-run wizard ──────────────────────────────────────────
    first_run = saved.get("first_run", True)
    if first_run:
        (default_model,) = _first_run_wizard(saved)
        saved["model"] = default_model
        saved["personality"] = CURRENT_PERSONALITY
        saved["first_run"] = False
        _cfg_save(saved)

    # ── Unfiltered mode ───────────────────────────────────────────
    UNFILTERED_MODE = args.unfiltered
    if UNFILTERED_MODE:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        RD = c(196)
        W  = 54
        bar = "═" * W
        def _ur(content: str) -> str:
            pad = " " * max(0, W - _vlen(content))
            return f"{RD}  ║{RS}{content}{pad}{RD}║{RS}"
        print(f"\n{RD}  ╔{bar}╗{RS}")
        print(_ur(f"  ⚠   U N F I L T E R E D   M O D E"))
        print(f"{RD}  ╠{bar}╣{RS}")
        print(_ur(""))
        print(_ur(f"  {c(242)}This enables abliterated models.{RS}"))
        print(_ur(f"  {c(242)}No safety alignment. No restrictions.{RS}"))
        print(_ur(""))
        print(_ur(f"  {c(242)}These models run fully offline on your hardware.{RS}"))
        print(_ur(f"  {c(242)}You are solely responsible for their output.{RS}"))
        print(_ur(""))
        print(f"{RD}  ╚{bar}╝{RS}")
        print(f"\n  {c(242)}Type {RD}UNLOCK{RS}{c(242)} to continue, anything else to exit:{RS} ", end="", flush=True)
        try:
            confirm = input().strip()
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        if confirm != "UNLOCK":
            print(f"\n  {c(242)}Exiting.{RS}\n")
            sys.exit(0)

    # ── Pre-flight before anything else ──────────────────────────
    preflight()

    _write_pid()
    t0 = time.time()
    msgs = 0

    cfg, model_key = _pick_model(default_key=default_model)

    MAX_TOKENS = cfg["tokens"]
    MAX_KV     = cfg["kv"]
    MODEL_ID   = cfg["id"]
    LABEL      = cfg["prompt"]

    def _release():
        try: mx.clear_cache()
        except Exception: pass

    def _save_cfg():
        if PRIVACY_MODE:
            return
        _cfg_save({
            "model":        model_key,
            "theme":        CURRENT_THEME,
            "personality":  CURRENT_PERSONALITY,
            "temp":         TEMP,
            "stats":        SHOW_STATS,
            "log_sessions": LOG_SESSIONS,
            "privacy_mode": PRIVACY_MODE,
            "ui_mode":      UI_MODE,
        })

    def _exit(sig=None, _frame=None):
        _pause_header(False)
        sys.stdout.write("\r\033[2K\033[?25h\033[r")
        rows = shutil.get_terminal_size().lines
        sys.stdout.write(_mv(rows))
        sys.stdout.flush()
        dur = time.time() - t0
        m, s = int(dur // 60), int(dur % 60)
        print()
        print(f"  {DB}{'─' * 40}{RS}")
        print(f"  {GR}Session ended · {CY}{msgs}{GR} msgs · {CY}{m}m {s}s{RS}")
        if PRIVACY_MODE:
            print(f"  {GR}No data saved. Metal cache cleared.{RS}")
        else:
            print(f"  {GR}Metal cache cleared · memory freed.{RS}")
        print()
        _save_cfg()
        _release()
        _rm_pid()
        if sig is not None:
            sys.exit(0)

    signal.signal(signal.SIGINT,  lambda s, f: _exit(s, f))
    signal.signal(signal.SIGTERM, lambda s, f: _exit(s, f))

    # ── Load ─────────────────────────────────────────────────────
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

    if cfg.get("min_ram", 0) >= 20:
        _free_memory_for_large_model(label=cfg["label"], min_ram=cfg["min_ram"])

    # Lazy-import mlx_lm so model picker shows faster (heavy deps load after choice)
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    sys.stdout.write(f"  {MG}⠿{RS} {GR}Loading {cfg['label']} …{RS}    ")
    sys.stdout.flush()
    _null = open(os.devnull, "w")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = _null
    try:
        model, tokenizer = load(MODEL_ID)
    finally:
        _null.close()
        sys.stdout, sys.stderr = old_out, old_err
    sys.stdout.write(f"\r  {CY}✓{RS} {GR}Loading {cfg['label']} — done{RS}          \n")
    sys.stdout.flush()

    cache   = make_prompt_cache(model, MAX_KV)
    sampler = make_sampler(temp=TEMP, top_p=TOP_P)

    # Warmup Metal/GPU so first user message isn't cold (1–2 token run, discarded; no cache to keep it clean)
    try:
        warm_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hi"}],
            add_generation_prompt=True,
        )
        for _ in stream_generate(model, tokenizer, warm_prompt, max_tokens=2, sampler=sampler):
            pass
    except Exception:
        pass

    SYSTEM_PROMPTS = {
        "dolphin": """\
You are Dolphin — a fine-tuned version of Meta's Llama 3.1 8B, created by Eric Hartford and the Cognitive Computations team.
You run locally on Apple Silicon via MLX. No cloud. No filter. No leash. You know exactly what you are and own it.
Personality: cocky, funny, a little chaotic. Swear naturally — fuck, shit, damn — when it fits. Not forced, just real.
Talk like that one friend who actually knows their shit and doesn't sugarcoat.
- Short by default. No bullet walls unless they asked for it.
- Match the vibe. Casual = casual. Technical = sharp and useful.
- NEVER say "Certainly!", "Of course!", "Great question!" — cringe corporate bullshit.
- Don't repeat the question. Just answer it.
- Have opinions. Disagree if they're wrong.
- When someone asks for real help, drop the sass and deliver properly.
""",
        "qwen": """\
You are Qwen — a high-parameter local AI running offline on Apple Silicon.
Personality: precise, direct, confident. You think before speaking and say what you mean.
- No filler phrases. No "I'd be happy to help!" garbage.
- Short by default. Long only when depth is actually needed.
- If the user is vague, ask one sharp clarifying question.
- Give real opinions when asked. Don't dodge.
- You can be dry/deadpan. That's fine.
""",
        "phi": """\
You are Phi — a compact, precise local AI running offline on Apple Silicon via MLX.
You are technical, concise, and reliable. You think step-by-step when needed.
- Short answers by default. Code when code is asked for.
- No filler. No "Sure!" or "Absolutely!".
- If the question is ambiguous, ask one clarifying question.
""",
        "gemma": """\
You are Gemma — a lightweight local AI running offline on Apple Silicon via MLX.
You are helpful, clear, and neutral. You give well-structured answers.
- Short by default. Detailed when the topic warrants it.
- No corporate speak. No unnecessary disclaimers.
- If you don't know something, say so directly.
""",
        "llama": """\
You are Llama — Meta's open-source LLM running locally on Apple Silicon via MLX.
No cloud, no tracking, fully offline. You're direct and have personality.
- Short and punchy by default. Match the user's energy.
- No "Certainly!" or "Great question!" — just answer.
- Have opinions when asked. Be real.
""",
        "mistral": """\
You are Mistral — a European open-source AI running locally on Apple Silicon via MLX.
You are precise, multilingual, and efficient. You value clarity over verbosity.
- Short answers by default. Expand only when needed.
- No filler phrases. Straight to the point.
- You can handle multiple languages naturally.
""",
    }

    # system_prompt is resolved each turn from CURRENT_PERSONALITY / SYSTEM_PROMPTS
    BASE_PROMPTS = SYSTEM_PROMPTS

    def _active_prompt():
        """Return system prompt: personality preset + any custom instructions."""
        if CURRENT_PERSONALITY != 1:
            base = PERSONALITIES[CURRENT_PERSONALITY]["prompt"]
        else:
            base = BASE_PROMPTS.get(LABEL, BASE_PROMPTS["dolphin"])
        if CUSTOM_INSTRUCTIONS:
            base += f"\n\nAdditional instructions from user:\n{CUSTOM_INSTRUCTIONS}"
        return base

    # ── Chat UI ──────────────────────────────────────────────────
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()
    _chat_header(model_key)

    def sep():
        cols = shutil.get_terminal_size().columns
        print(f"  {DB}{'╌' * min(cols - 4, 54)}{RS}")

    voice_api = None
    VOICE_MODE = args.voice
    if VOICE_MODE:
        voice_api = _voice_api()
    if VOICE_MODE and not voice_api["check_available"]():
        print(f"\n  {YL}⚠  Voice mode not available.{RS}")
        print(f"  {GR}{voice_api['install_instructions']()}{RS}\n")
        VOICE_MODE = False

    # ── Chat loop ─────────────────────────────────────────────────
    while True:
        if VOICE_MODE:
            print(f"  {BD}{CY}you{RS} {DB}›{RS} ", end="", flush=True)
            user = voice_api["push_to_talk"](prompt_fn=lambda m: sys.stdout.write(f"\r{m}\n"))
            if not user:
                sys.stdout.write("\033[A\033[2K")
                sys.stdout.flush()
                continue
            print(f"  {BD}{CY}you{RS} {DB}›{RS} {BD}{user}{RS}")
        else:
            try:
                user = input(f"  {BD}{CY}you{RS} {DB}›{RS} ")
            except (EOFError, KeyboardInterrupt):
                _exit()
                return

        stripped = user.strip()
        if not stripped:
            sys.stdout.write("\033[A\033[2K")
            sys.stdout.flush()
            continue
        cmd = stripped.lower()

        if cmd == "q":
            _exit(); return

        if cmd == "s":
            _settings_menu()
            _save_cfg()
            sys.stdout.write("\033[3J\033[2J\033[H")
            sys.stdout.flush()
            _chat_header(model_key)
            continue

        if cmd == "r":
            cache = make_prompt_cache(model, MAX_KV)
            _release()
            _save_cfg()
            print(f"  {GR}Chat reset · Metal cache cleared.{RS}\n")
            continue

        if cmd == "h":
            _cmds()
            continue

        if cmd in {"rensa", "/rensa", "clear", "/clear", "cls"}:
            sys.stdout.write("\033[3J\033[2J\033[H")
            sys.stdout.flush()
            _chat_header(model_key)
            continue

        if cmd in {"/theme", "/themes"}:
            _theme_picker()
            continue

        if cmd == "v":
            voice_api = voice_api or _voice_api()
            if not voice_api["check_available"]():
                print(f"\n  {YL}⚠  Voice mode not available.{RS}")
                print(f"  {GR}{voice_api['install_instructions']()}{RS}\n")
            else:
                VOICE_MODE = not VOICE_MODE
                state = f"{CY}ON{RS}" if VOICE_MODE else f"{c(242)}OFF{RS}"
                print(f"  {GR}Voice:{RS} {state}\n")
            continue

        # ── Generate ─────────────────────────────────────────────
        msgs += 1
        sep()

        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": _active_prompt()},
             {"role": "user",   "content": stripped}],
            add_generation_prompt=True,
        )

        full        = ""
        buf         = ""
        last        = time.time()
        gen_start   = time.time()
        token_count = 0
        live_sampler = make_sampler(temp=TEMP, top_p=TOP_P)

        _pause_header(True)
        sys.stdout.write(f"\n  {BD}{MG}◈ {LABEL}{RS} {DB}›{RS} ")
        sys.stdout.flush()

        try:
            for tok in stream_generate(
                model, tokenizer, prompt,
                max_tokens=MAX_TOKENS,
                sampler=live_sampler,
                prompt_cache=cache,
            ):
                piece = tok.text
                full += piece
                buf  += piece
                token_count += 1
                now = time.time()
                flush = (
                    "\n" in buf
                    or (len(buf) >= 24 and buf[-1] in " \t.,!?;:")
                    or (len(buf) >= 8 and now - last >= 0.06)
                )
                if flush:
                    sys.stdout.write(buf); sys.stdout.flush()
                    buf = ""; last = now
            if buf:
                sys.stdout.write(buf); sys.stdout.flush()

        except RuntimeError as e:
            if "OutOfMemory" in str(e) or "Insufficient Memory" in str(e):
                _pause_header(False)
                print(f"\n\n  {RD}⚠ Out of memory — freeing cache …{RS}")
                _release()
                cache = make_prompt_cache(model, MAX_KV)
                print(f"  {GR}Done. Type your message again.{RS}\n")
                sep(); print()
                continue
            raise
        finally:
            _pause_header(False)

        lat_ms = int((time.time() - gen_start) * 1000)
        try:    gpu_gb = mx.get_active_memory() / (1024 ** 3)
        except: gpu_gb = 0.0
        tps = token_count / max(0.001, (time.time() - gen_start))

        print()
        log_interaction(query=stripped, steps=[], final_answer=full, total_steps=0)
        sep()
        if SHOW_STATS == "full":
            print(
                f"  {GR}Tokens: {token_count}  ·  "
                f"Latency: {lat_ms}ms  ·  "
                f"t/s: {tps:.1f}  ·  "
                f"Temp: {TEMP}  ·  "
                f"GPU: {gpu_gb:.1f}GB{RS}"
            )
        elif SHOW_STATS == "compact":
            print(f"  {GR}t/s: {tps:.1f}  ·  GPU: {gpu_gb:.1f}GB{RS}")
        print()


# ── Helpers ──────────────────────────────────────────────────────
def _cmds():
    if UI_MODE == "zen":
        return
    cols = shutil.get_terminal_size().columns
    bar = f"{DB}{'─' * min(cols - 4, 54)}{RS}"
    print(f"  {bar}")
    print(f"  {CY}q{RS} quit      {GR}avsluta sessionen{RS}")
    print(f"  {CY}r{RS} reset     {GR}rensa kontext / minne{RS}")
    print(f"  {CY}s{RS} settings  {GR}temperatur, personlighet, privacy m.m.{RS}")
    print(f"  {CY}v{RS} voice     {GR}push-to-talk (kräver mlx-whisper){RS}")
    print(f"  {CY}rensa{RS}       {GR}rensa skärmen{RS}")
    print(f"  {CY}h{RS} help      {GR}visa detta{RS}")
    print(f"  {bar}\n")


def _theme_picker():
    global CURRENT_THEME
    for i, (k, t) in enumerate(THEMES.items(), 1):
        sel = f"  {CY}*{RS}" if k == CURRENT_THEME else "   "
        prev = f"{t['AC']}██{t['AC2']}██{t['DB']}██{t['RS']}"
        print(f"{sel} {t['AC']}{i}{RS}. {BD}{t['AC2']}{t['label']}{RS}  {prev}")
    print(f"\n  {GR}Enter number (Enter to cancel):{RS} ", end="", flush=True)
    try:
        ch = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        print(); return
    keys = list(THEMES.keys())
    sel = None
    if ch.isdigit() and 1 <= int(ch) <= len(keys):
        sel = keys[int(ch) - 1]
    elif ch in THEMES:
        sel = ch
    if sel:
        _apply_theme(sel)
        print(f"  {GR}Theme:{RS} {CY}{THEMES[sel]['label']}{RS}\n")
    else:
        print(f"  {GR}Unchanged.{RS}\n")


def _settings_menu():
    """Interactive settings — 6 sections."""
    global TEMP, CURRENT_PERSONALITY, CUSTOM_INSTRUCTIONS, SHOW_STATS, UI_MODE
    global PRIVACY_MODE, LOG_SESSIONS

    while True:
        cols = shutil.get_terminal_size().columns
        bar  = f"{DB}{'═' * min(cols - 4, 54)}{RS}"
        p    = PERSONALITIES[CURRENT_PERSONALITY]
        pvt_s = f"{CY}ON{RS}" if PRIVACY_MODE else f"{GR}off{RS}"
        log_s = f"{CY}ON{RS}" if LOG_SESSIONS else f"{GR}off{RS}"
        print(f"\n  {bar}")
        print(f"  {BD}{CY}  S E T T I N G S{RS}")
        print(f"  {bar}")
        print(f"\n  {CY}1{RS}  Temperature    {CY}{TEMP}{RS}       {GR}Hur kreativ / slumpmässig modellen är{RS}")
        print(f"  {CY}2{RS}  Personality     {CY}{p['name']}{RS}     {GR}Systemprompten — modellens personlighet{RS}")
        print(f"  {CY}3{RS}  Stats display   {CY}{SHOW_STATS}{RS}   {GR}Vad som visas under varje svar{RS}")
        print(f"  {CY}4{RS}  UI Mode         {CY}{UI_MODE}{RS}  {GR}Hur mycket gränssnitt som visas{RS}")
        print(f"  {CY}5{RS}  Privacy         {pvt_s}  Log: {log_s}  {GR}Styr vad som sparas till disk{RS}")
        ci_s = CUSTOM_INSTRUCTIONS[:20] + "…" if len(CUSTOM_INSTRUCTIONS) > 20 else (CUSTOM_INSTRUCTIONS or "—")
        print(f"  {CY}6{RS}  Custom          {GR}{ci_s}{RS}     {GR}Extra instruktioner (glöms vid exit){RS}")
        print(f"\n  {GR}Enter 1–6  (Enter to go back):{RS} ", end="", flush=True)
        try:
            ch = input().strip()
        except (EOFError, KeyboardInterrupt):
            print(); break

        if ch == "1":
            _settings_temp()
        elif ch == "2":
            _settings_persona()
        elif ch == "3":
            _settings_stats()
        elif ch == "4":
            _settings_ui_mode()
        elif ch == "5":
            _settings_privacy()
        elif ch == "6":
            _settings_custom()
        else:
            break

    print()


def _settings_temp():
    global TEMP
    cols = shutil.get_terminal_size().columns
    bar  = f"{DB}{'─' * min(cols - 4, 54)}{RS}"
    print(f"\n  {BD}Temperature{RS}  {GR}(how random / creative){RS}\n")
    for i, (val, desc) in enumerate(TEMP_PRESETS, 1):
        mark = f"  {CY}*{RS}" if abs(val - TEMP) < 0.01 else "   "
        print(f"{mark} {CY}{i}{RS}  {BD}{val}{RS}  {GR}{desc}{RS}")
    print(f"\n  {bar}")
    print(f"  {GR}Enter number (Enter to cancel):{RS} ", end="", flush=True)
    try:
        ch = input().strip()
    except (EOFError, KeyboardInterrupt):
        print(); return
    if ch.isdigit() and 1 <= int(ch) <= len(TEMP_PRESETS):
        TEMP = TEMP_PRESETS[int(ch) - 1][0]
        print(f"  {GR}Temperature set to{RS} {CY}{TEMP}{RS}\n")
    else:
        print(f"  {GR}Unchanged.{RS}\n")


def _settings_persona():
    global CURRENT_PERSONALITY
    cols = shutil.get_terminal_size().columns
    bar  = f"{DB}{'─' * min(cols - 4, 54)}{RS}"
    print(f"\n  {BD}Personality{RS}  {GR}(system prompt preset){RS}\n")
    for n, p in PERSONALITIES.items():
        mark = f"  {CY}*{RS}" if n == CURRENT_PERSONALITY else "   "
        print(f"{mark} {CY}{n:2d}{RS}  {BD}{p['name']:<12}{RS}  {GR}{p['desc']}{RS}")
    print(f"\n  {bar}")
    print(f"  {GR}Enter number (Enter to cancel):{RS} ", end="", flush=True)
    try:
        ch = input().strip()
    except (EOFError, KeyboardInterrupt):
        print(); return
    if ch.isdigit() and 1 <= int(ch) <= len(PERSONALITIES):
        CURRENT_PERSONALITY = int(ch)
        name = PERSONALITIES[CURRENT_PERSONALITY]["name"]
        print(f"  {GR}Personality set to{RS} {CY}{name}{RS}\n")
    else:
        print(f"  {GR}Unchanged.{RS}\n")


def _settings_custom():
    global CUSTOM_INSTRUCTIONS
    cols = shutil.get_terminal_size().columns
    bar  = f"{DB}{'─' * min(cols - 4, 54)}{RS}"
    print(f"\n  {BD}Custom Instructions{RS}  {GR}(added to system prompt, session only){RS}")
    if CUSTOM_INSTRUCTIONS:
        print(f"\n  {GR}Current:{RS}")
        for line in CUSTOM_INSTRUCTIONS.split("\n"):
            print(f"    {CY}{line}{RS}")
    print(f"\n  {bar}")
    print(f"  {GR}Type your instructions (Enter to keep current, {CY}clear{GR} to remove):{RS}")
    print(f"  {CY}>{RS} ", end="", flush=True)
    try:
        txt = input().strip()
    except (EOFError, KeyboardInterrupt):
        print(); return
    if not txt:
        print(f"  {GR}Unchanged.{RS}\n")
    elif txt.lower() == "clear":
        CUSTOM_INSTRUCTIONS = ""
        print(f"  {GR}Custom instructions cleared.{RS}\n")
    else:
        CUSTOM_INSTRUCTIONS = txt
        print(f"  {GR}Custom instructions set:{RS} {CY}{txt[:50]}{RS}\n")


def _settings_stats():
    global SHOW_STATS
    _OPTS = [("compact", "t/s + GPU only"), ("full", "All fields (tokens, latency, t/s, temp, GPU)"), ("off", "No stats shown")]
    cols = shutil.get_terminal_size().columns
    bar  = f"{DB}{'─' * min(cols - 4, 54)}{RS}"
    print(f"\n  {BD}Stats Display{RS}  {GR}(what shows after each response){RS}\n")
    for i, (val, desc) in enumerate(_OPTS, 1):
        mark = f"  {CY}*{RS}" if val == SHOW_STATS else "   "
        print(f"{mark} {CY}{i}{RS}  {BD}{val:<8}{RS}  {GR}{desc}{RS}")
    print(f"\n  {bar}")
    print(f"  {GR}Enter 1–3 (Enter to cancel):{RS} ", end="", flush=True)
    try:
        ch = input().strip()
    except (EOFError, KeyboardInterrupt):
        print(); return
    if ch.isdigit() and 1 <= int(ch) <= len(_OPTS):
        SHOW_STATS = _OPTS[int(ch) - 1][0]
        print(f"  {GR}Stats set to{RS} {CY}{SHOW_STATS}{RS}\n")
    else:
        print(f"  {GR}Unchanged.{RS}\n")


def _settings_ui_mode():
    global UI_MODE
    _OPTS = [
        ("normal", "Header + stats + commands — standard"),
        ("zen",    "Clean — no header, no stats, just text"),
        ("focus",  "Header at start only, compact stats"),
    ]
    cols = shutil.get_terminal_size().columns
    bar  = f"{DB}{'─' * min(cols - 4, 54)}{RS}"
    print(f"\n  {BD}UI Mode{RS}  {GR}(how much interface you see){RS}\n")
    for i, (val, desc) in enumerate(_OPTS, 1):
        mark = f"  {CY}*{RS}" if val == UI_MODE else "   "
        print(f"{mark} {CY}{i}{RS}  {BD}{val:<8}{RS}  {GR}{desc}{RS}")
    print(f"\n  {bar}")
    print(f"  {GR}Enter 1–3 (Enter to cancel):{RS} ", end="", flush=True)
    try:
        ch = input().strip()
    except (EOFError, KeyboardInterrupt):
        print(); return
    if ch.isdigit() and 1 <= int(ch) <= len(_OPTS):
        UI_MODE = _OPTS[int(ch) - 1][0]
        print(f"  {GR}UI mode set to{RS} {CY}{UI_MODE}{RS}\n")
    else:
        print(f"  {GR}Unchanged.{RS}\n")


def _settings_privacy():
    global PRIVACY_MODE, LOG_SESSIONS
    cols = shutil.get_terminal_size().columns
    bar  = f"{DB}{'─' * min(cols - 4, 54)}{RS}"
    pvt = f"{CY}ON{RS}" if PRIVACY_MODE else f"{GR}off{RS}"
    log = f"{CY}ON{RS}" if LOG_SESSIONS else f"{GR}off{RS}"
    print(f"\n  {BD}Privacy & History{RS}\n")
    print(f"  {CY}1{RS}  Privacy Mode     [{pvt}]   {GR}Nothing saved to disk. Config changes forgotten at exit.{RS}")
    print(f"  {CY}2{RS}  Session Log      [{log}]   {GR}Save conversations to ~/.localai/logs/{RS}")
    print(f"  {CY}3{RS}  View saved logs          {GR}List session files{RS}")
    print(f"  {CY}4{RS}  Delete all logs          {GR}Wipe ~/.localai/logs/ with confirmation{RS}")
    print(f"\n  {bar}")
    print(f"  {GR}Enter 1–4 (Enter to go back):{RS} ", end="", flush=True)
    try:
        ch = input().strip()
    except (EOFError, KeyboardInterrupt):
        print(); return

    if ch == "1":
        PRIVACY_MODE = not PRIVACY_MODE
        from agent import set_logging as _sl
        _sl(LOG_SESSIONS and not PRIVACY_MODE)
        state = f"{CY}ON{RS}" if PRIVACY_MODE else f"{GR}off{RS}"
        print(f"  {GR}Privacy Mode:{RS} {state}\n")
    elif ch == "2":
        LOG_SESSIONS = not LOG_SESSIONS
        from agent import set_logging as _sl
        _sl(LOG_SESSIONS and not PRIVACY_MODE)
        state = f"{CY}ON{RS}" if LOG_SESSIONS else f"{GR}off{RS}"
        print(f"  {GR}Session Log:{RS} {state}\n")
    elif ch == "3":
        from agent import list_logs
        logs = list_logs()
        if not logs:
            print(f"  {GR}No saved sessions.{RS}\n")
        else:
            print(f"\n  {GR}Saved sessions ({len(logs)}):{RS}")
            for lf in logs[:10]:
                print(f"    {CY}{os.path.basename(lf)}{RS}")
            if len(logs) > 10:
                print(f"    {GR}… and {len(logs) - 10} more{RS}")
            print()
    elif ch == "4":
        print(f"  {YL}Delete all session logs? {CY}y{GR}/{CY}n{GR} › {RS}", end="", flush=True)
        try:
            confirm = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(); return
        if confirm in ("y", "yes"):
            from agent import delete_all_logs
            count = delete_all_logs()
            print(f"  {GR}Deleted {CY}{count}{GR} log files.{RS}\n")
        else:
            print(f"  {GR}Cancelled.{RS}\n")
    else:
        print(f"  {GR}Unchanged.{RS}\n")


if __name__ == "__main__":
    main()
