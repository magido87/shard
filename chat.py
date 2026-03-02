#!/usr/bin/env python3
"""
Local AI Chat — Apple Silicon · pre-flight cleanup · model picker · per-model UI
"""

import logging
import os
import select
import shutil
import signal
import sys
import threading
import time

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import mlx.core as mx
import psutil
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from agent import log_interaction


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
MODELS = {
    "dolphin": {
        "id":     "mlx-community/Dolphin3.0-Llama3.1-8B-4bit",
        "label":  "Dolphin 3.0  ·  Llama 3.1 8B",
        "size":   "8B",
        "prompt": "dolphin",
        "theme":  "neon",
        "kv":     2048,
        "tokens": 1024,
    },
    "qwen": {
        "id":     "mlx-community/Qwen2.5-14B-Instruct-4bit",
        "label":  "Qwen 2.5  ·  14B Instruct",
        "size":   "14B",
        "prompt": "qwen",
        "theme":  "ocean",
        "kv":     1536,
        "tokens": 768,
    },
    "qwen35": {
        "id":     "mlx-community/Qwen3.5-27B-4bit",
        "label":  "Qwen 3.5  ·  27B Dense",
        "size":   "27B",
        "prompt": "qwen",
        "theme":  "ocean",
        "kv":     1024,
        "tokens": 512,
    },
}

TEMP = 0.72
TOP_P = 0.92
PID_FILE = os.path.expanduser("~/.localai/llm.pid")
HEADER_LINES = 2

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
    cols = shutil.get_terminal_size().columns
    seg_cpu = f"{GR}CPU{RS} {_bar(cpu,6)} {CY}{cpu:3.0f}%{RS}"
    seg_gpu = f"{GR}GPU{RS} {CY}{gpu:.1f}GB{RS}"
    seg_ram = f"{GR}RAM{RS} {_bar(pct,6)} {CY}{ru:.1f}{GR}/{rt:.0f}GB{RS}"
    seg_swp = f"  {DB}│{RS}  {GR}SWP{RS} {YL}{swap:.1f}GB{RS}" if swap > 0.1 else ""
    div = f"  {DB}│{RS}  "
    line = "  " + div.join([seg_cpu, seg_gpu, seg_ram]) + seg_swp
    sep  = f"  {DB}{'─' * min(cols - 4, 72)}{RS}"
    return line, sep


def _print_header():
    l1, l2 = _render_header()
    print(l1)
    print(l2)


def _update_header():
    pass   # no-op — header is printed inline, not live-updated


def _pause_header(p: bool):
    pass   # no-op


def _setup_scroll():
    pass   # no scroll regions — avoids iTerm2 cursor jump bugs


# ── Banners ──────────────────────────────────────────────────────
def _banner_dolphin():
    try:    dev = mx.device_info().get("device_name", "M3")
    except: dev = "M3"
    ram = psutil.virtual_memory().total / (1024 ** 3)
    print(f"""
{DB}  ╔{'═'*54}╗{RS}
{DB}  ║{RS}{MG}  ≋ ≋ ≋  {BD}D O L P H I N   3 . 0{RS}{MG}  ≋ ≋ ≋          {DB}║{RS}
{DB}  ╠{'═'*54}╣{RS}
{DB}  ║{RS}                                                      {DB}║{RS}
{DB}  ║{RS}  {DB}     ____{RS}      {BD}{CY}Llama 3.1 · 8B · 4-bit{RS}          {DB}║{RS}
{DB}  ║{RS}  {MG}   /      \\{RS}     {GR}Apple {dev} · {ram:.0f} GB unified{RS}          {DB}║{RS}
{DB}  ║{RS}  {MG}  |  {CY}◈{MG}     |{RS}     {GR}Temp {TEMP} · Top-p {TOP_P}{RS}               {DB}║{RS}
{DB}  ║{RS}  {MG}   \\______/{RS}     {GR}offline · no cloud · MLX{RS}               {DB}║{RS}
{DB}  ║{RS}  {MG}    \\___~_/{RS}                                       {DB}║{RS}
{DB}  ║{RS}                                                      {DB}║{RS}
{DB}  ╚{'═'*54}╝{RS}
""")


def _banner_qwen():
    try:    dev = mx.device_info().get("device_name", "M3")
    except: dev = "M3"
    ram = psutil.virtual_memory().total / (1024 ** 3)
    print(f"""
{DB}  ╔{'═'*54}╗{RS}
{DB}  ║{RS}{CY}    ◆ ◆ ◆   {BD}Q W E N   2 . 5   ◆ ◆ ◆{RS}{CY}           {DB}║{RS}
{DB}  ╠{'═'*54}╣{RS}
{DB}  ║{RS}                                                      {DB}║{RS}
{DB}  ║{RS}  {AC3}  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓{RS}            {DB}║{RS}
{DB}  ║{RS}  {BD}{CY}  Instruct · 14B · 4-bit{RS}                         {DB}║{RS}
{DB}  ║{RS}  {GR}  Apple {dev} · {ram:.0f} GB unified{RS}                    {DB}║{RS}
{DB}  ║{RS}  {GR}  Temp {TEMP} · Top-p {TOP_P} · offline · MLX{RS}            {DB}║{RS}
{DB}  ║{RS}  {MG}  ◈ multilingual · 20 languages{RS}                  {DB}║{RS}
{DB}  ║{RS}                                                      {DB}║{RS}
{DB}  ╚{'═'*54}╝{RS}
""")


def _banner_qwen35():
    try:    dev = mx.device_info().get("device_name", "M3")
    except: dev = "M3"
    ram = psutil.virtual_memory().total / (1024 ** 3)
    print(f"""
{DB}  ╔{'═'*54}╗{RS}
{DB}  ║{RS}{CY}    ◆ ◆ ◆   {BD}Q W E N   3 . 5   ◆ ◆ ◆{RS}{CY}           {DB}║{RS}
{DB}  ╠{'═'*54}╣{RS}
{DB}  ║{RS}                                                      {DB}║{RS}
{DB}  ║{RS}  {AC3}  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓{RS}            {DB}║{RS}
{DB}  ║{RS}  {BD}{CY}  Dense · 27B · 4-bit{RS}                            {DB}║{RS}
{DB}  ║{RS}  {GR}  Apple {dev} · {ram:.0f} GB unified{RS}                    {DB}║{RS}
{DB}  ║{RS}  {GR}  Temp {TEMP} · Top-p {TOP_P} · offline · MLX{RS}            {DB}║{RS}
{DB}  ║{RS}  {MG}  ◈ latest Qwen · deep reasoning{RS}                  {DB}║{RS}
{DB}  ║{RS}                                                      {DB}║{RS}
{DB}  ╚{'═'*54}╝{RS}
""")


def _banner(key: str):
    if key == "qwen35":
        _banner_qwen35()
    elif key == "qwen":
        _banner_qwen()
    else:
        _banner_dolphin()


def _sysinfo(cfg: dict):
    pass  # info is now inside the banner


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


# ── Model picker ─────────────────────────────────────────────────
def _pick_model():
    _apply_theme("neon")
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

    W = min(shutil.get_terminal_size().columns - 4, 58)
    bar = "═" * W

    ram  = psutil.virtual_memory().total / (1024**3)
    swap = psutil.swap_memory().used / (1024**3)
    try:    gpu = mx.get_active_memory() / (1024**3)
    except: gpu = 0.0

    print(f"""
{c(51)}  ╔{bar}╗{RS}
{c(51)}  ║{RS}{BD}{c(51)}{'  L O C A L   A I   —   M O D E L   S E L E C T':^{W}}{RS}{c(51)}║{RS}
{c(51)}  ╠{bar}╣{RS}
{c(51)}  ║{RS}  {c(242)}RAM {ram:.0f} GB  ·  GPU {gpu:.1f} GB  ·  Swap {swap:.1f} GB{RS}{' '*(W-40)}{c(51)}║{RS}
{c(51)}  ╠{bar}╣{RS}
{c(51)}  ║{RS}                                                           {c(51)}║{RS}
{c(51)}  ║{RS}   {BD}{c(51)}1{RS}  {BD}Dolphin 3.0{RS}  {c(242)}Llama 3.1 · 8B  · Neon Cyber{RS}    {c(51)}║{RS}
{c(51)}  ║{RS}      {c(38)}≋ fast · uncensored · great for chat{RS}           {c(51)}║{RS}
{c(51)}  ║{RS}                                                           {c(51)}║{RS}
{c(51)}  ║{RS}   {BD}{c(135)}2{RS}  {BD}Qwen 2.5{RS}     {c(242)}Instruct · 14B  · Deep Ocean{RS}    {c(51)}║{RS}
{c(51)}  ║{RS}      {c(87)}◆ smarter · multilingual · better reasoning{RS}    {c(51)}║{RS}
{c(51)}  ║{RS}                                                           {c(51)}║{RS}
{c(51)}  ║{RS}   {BD}{c(226)}3{RS}  {BD}Qwen 3.5{RS}     {c(242)}Dense · 27B · Deep Ocean{RS}       {c(51)}║{RS}
{c(51)}  ║{RS}      {c(87)}◈ latest · deep reasoning · 16 GB tight{RS}      {c(51)}║{RS}
{c(51)}  ║{RS}                                                           {c(51)}║{RS}
{c(51)}  ╚{bar}╝{RS}
""")

    chosen = None
    deadline = time.time() + 10.0
    model_map = {"1": "dolphin", "2": "qwen", "3": "qwen35"}
    while time.time() < deadline:
        remaining = int(deadline - time.time()) + 1
        sys.stdout.write(f"\r  {GR}Auto-select {BD}Dolphin{RS}{GR} in {CY}{remaining}{GR}s  —  press {CY}1{GR} / {c(135)}2{GR} / {c(226)}3{GR} …{RS}  ")
        sys.stdout.flush()
        rlist, _, _ = select.select([sys.stdin], [], [], 0.2)
        if rlist:
            ch = sys.stdin.readline().strip()
            chosen = model_map.get(ch, "dolphin")
            break

    if chosen is None:
        chosen = "dolphin"

    sys.stdout.write("\r" + " " * 70 + "\r")
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


# ── Main ─────────────────────────────────────────────────────────
def main():
    # ── Pre-flight before anything else ──────────────────────────
    preflight()

    _write_pid()
    t0 = time.time()
    msgs = 0

    cfg, model_key = _pick_model()

    MAX_TOKENS = cfg["tokens"]
    MAX_KV     = cfg["kv"]
    MODEL_ID   = cfg["id"]
    LABEL      = cfg["prompt"]

    def _release():
        try: mx.clear_cache()
        except Exception: pass

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
        print(f"  {GR}Metal cache cleared · memory freed.{RS}")
        print()
        _release()
        _rm_pid()
        if sig is not None:
            sys.exit(0)

    signal.signal(signal.SIGINT,  lambda s, f: _exit(s, f))
    signal.signal(signal.SIGTERM, lambda s, f: _exit(s, f))

    # ── Load ─────────────────────────────────────────────────────
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()
    _banner(model_key)
    _sysinfo(cfg)

    with _Spin(f"Loading {cfg['label']}"):
        old_err, sys.stderr = sys.stderr, open(os.devnull, "w")
        try:
            model, tokenizer = load(MODEL_ID)
        finally:
            sys.stderr.close()
            sys.stderr = old_err

    cache   = make_prompt_cache(model, MAX_KV)
    sampler = make_sampler(temp=TEMP, top_p=TOP_P)

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
    }

    # system_prompt is resolved each turn from CURRENT_PERSONALITY / SYSTEM_PROMPTS
    BASE_PROMPTS = SYSTEM_PROMPTS

    def _active_prompt():
        """Return system prompt: personality preset + any custom instructions."""
        if CURRENT_PERSONALITY != 1:
            base = PERSONALITIES[CURRENT_PERSONALITY]["prompt"]
        else:
            base = BASE_PROMPTS.get(model_key, BASE_PROMPTS["dolphin"])
        if CUSTOM_INSTRUCTIONS:
            base += f"\n\nAdditional instructions from user:\n{CUSTOM_INSTRUCTIONS}"
        return base

    # ── Chat UI ──────────────────────────────────────────────────
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()
    _banner(model_key)
    _print_header()
    _cmds()

    def sep():
        cols = shutil.get_terminal_size().columns
        print(f"  {DB}{'╌' * min(cols - 4, 54)}{RS}")

    # ── Chat loop ─────────────────────────────────────────────────
    while True:
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
            sys.stdout.write("\033[3J\033[2J\033[H")
            sys.stdout.flush()
            _banner(model_key)
            _print_header()
            _cmds()
            continue

        if cmd == "r":
            cache = make_prompt_cache(model, MAX_KV)
            _release()
            print(f"  {GR}Chat reset · Metal cache cleared.{RS}\n")
            continue

        if cmd == "h":
            _cmds()
            continue

        if cmd in {"rensa", "/rensa", "clear", "/clear", "cls"}:
            sys.stdout.write("\033[3J\033[2J\033[H")
            sys.stdout.flush()
            _banner(model_key)
            _print_header()
            _cmds()
            continue

        if cmd in {"/theme", "/themes"}:
            _theme_picker()
            continue

        # ── Generate ─────────────────────────────────────────────
        msgs += 1
        sep()

        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": _active_prompt()},
             {"role": "user",   "content": stripped}],
            add_generation_prompt=True,
        )

        full = ""
        buf  = ""
        last = time.time()
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

        print()
        log_interaction(query=stripped, steps=[], final_answer=full, total_steps=0)
        sep()
        print()


# ── Helpers ──────────────────────────────────────────────────────
def _cmds():
    cols = shutil.get_terminal_size().columns
    bar = f"{DB}{'─' * min(cols - 4, 54)}{RS}"
    print(f"  {bar}")
    print(
        f"  {CY}q{RS} quit  "
        f"{CY}r{RS} reset  "
        f"{CY}s{RS} settings  "
        f"{CY}rensa{RS} clear  "
        f"{CY}h{RS} help"
    )
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
    """Interactive settings: temperature + personality + custom instructions."""
    global TEMP, CURRENT_PERSONALITY, CUSTOM_INSTRUCTIONS

    while True:
        cols = shutil.get_terminal_size().columns
        bar  = f"{DB}{'═' * min(cols - 4, 54)}{RS}"
        p    = PERSONALITIES[CURRENT_PERSONALITY]
        ci   = CUSTOM_INSTRUCTIONS[:40] + "…" if len(CUSTOM_INSTRUCTIONS) > 40 else CUSTOM_INSTRUCTIONS
        print(f"\n  {bar}")
        print(f"  {BD}{CY}  S E T T I N G S{RS}")
        print(f"  {bar}")
        print(f"  {GR}Personality:{RS} {BD}{CY}{p['name']}{RS}  {GR}· Temp:{RS} {CY}{TEMP}{RS}")
        if CUSTOM_INSTRUCTIONS:
            print(f"  {GR}Custom:{RS} {CY}{ci}{RS}")
        print(f"  {bar}")
        print(f"\n  {CY}1{RS}  Temperature")
        print(f"  {CY}2{RS}  Personality")
        print(f"  {CY}3{RS}  Custom instructions  {GR}(session only){RS}")
        print(f"\n  {GR}Enter 1 / 2 / 3  (Enter to go back):{RS} ", end="", flush=True)
        try:
            ch = input().strip()
        except (EOFError, KeyboardInterrupt):
            print(); break

        if ch == "1":
            _settings_temp()
        elif ch == "2":
            _settings_persona()
        elif ch == "3":
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


if __name__ == "__main__":
    main()
