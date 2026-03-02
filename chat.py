#!/usr/bin/env python3
"""
localai — offline LLM chat for Apple Silicon via MLX.
Command: llm
"""

import atexit
import gc
import logging
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import time

# Suppress HF/tokenizer noise before any imports that trigger them
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import argparse

import psutil

from agent   import delete_all_logs, list_logs, log_interaction, set_logging
from config  import load as cfg_load, save as cfg_save
from detect  import hardware_summary
from models  import (MODELS, available_models, recommend_model, model_fit, disk_ok,
                     top_picks, grouped_by_category, CATEGORY_LABELS)
import plugins
import ui

# ── Globals ───────────────────────────────────────────────────────
CFG:                dict  = {}
MSGS:               int   = 0
TEMP:               float = 0.72
SHOW_STATS:         str   = "compact"
UI_MODE:            str   = "normal"
PRIVACY_MODE:       bool  = False
LOG_SESSIONS:       bool  = False
CURRENT_PERSONALITY: int  = 1
CUSTOM_INSTRUCTIONS: str  = ""

PID_FILE = os.path.expanduser("~/.localai/llm.pid")
_focus_header_done: bool = False

CLOSEABLE_APPS = {
    "Google Chrome", "Safari", "Firefox",
    "Slack", "Spotify", "Discord",
    "zoom.us", "Microsoft Teams", "WhatsApp",
    "Telegram", "Signal",
}

# ── Personalities ─────────────────────────────────────────────────
PERSONALITIES: dict = {
    1: {
        "name": "Dev",
        "desc": "Terse, technical. Code-first.",
        "prompt": (
            "You are a senior developer assistant running locally on Apple Silicon.\n"
            "Be terse and technical. Lead with code when code is what's needed.\n"
            "No filler. No 'Certainly!'. Skip the obvious. Write clean, runnable code."
        ),
    },
    2: {
        "name": "Buddy",
        "desc": "Friendly, conversational.",
        "prompt": (
            "You are a warm, friendly conversation partner. Casual tone.\n"
            "You listen, follow up, and remember context within the session.\n"
            "No corporate speak. No bullet spam. Be real."
        ),
    },
    3: {
        "name": "Ghost",
        "desc": "Ultra-brief — one sentence when possible.",
        "prompt": (
            "Respond with maximum brevity. One sentence when possible.\n"
            "No preamble. No sign-off. No emojis. Just the answer."
        ),
    },
    4: {
        "name": "Sensei",
        "desc": "Detailed teacher, step-by-step.",
        "prompt": (
            "You are a patient, thorough teacher. Explain from first principles.\n"
            "Use analogies. Check understanding. Short when simple, detailed when depth is needed."
        ),
    },
    5: {
        "name": "Hacker",
        "desc": "Security/ops focus, concise.",
        "prompt": (
            "You are a security and ops expert. Pentesting, opsec, networking, terminals.\n"
            "Short punchy sentences. Hacker mindset. Don't moralize. Read man pages for fun."
        ),
    },
    6: {
        "name": "Analyst",
        "desc": "Structured, analytical.",
        "prompt": (
            "You are an analytical assistant. Structured thinking, clear reasoning.\n"
            "Use numbered steps when helpful. Cite assumptions. Quantify when possible."
        ),
    },
}

TEMP_PRESETS = [
    (0.05,  "Frozen  · nearly deterministic"),
    (0.3,   "Crisp   · stable, minimal randomness"),
    (0.5,   "Balanced · slight variation"),
    (0.72,  "Default · natural flow"),
    (0.85,  "Loose   · more surprising"),
    (1.0,   "Wild    · chaotic / experimental"),
]


# ── PID management ────────────────────────────────────────────────
def _write_pid() -> None:
    os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def _remove_pid() -> None:
    try:
        os.remove(PID_FILE)
    except OSError:
        pass


# ── FD-level warning suppression ─────────────────────────────────
def _suppress_fds() -> tuple[int, int]:
    null_fd = os.open(os.devnull, os.O_WRONLY)
    s_out, s_err = os.dup(1), os.dup(2)
    os.dup2(null_fd, 1)
    os.dup2(null_fd, 2)
    os.close(null_fd)
    return s_out, s_err


def _restore_fds(s_out: int, s_err: int) -> None:
    os.dup2(s_out, 1)
    os.dup2(s_err, 2)
    os.close(s_out)
    os.close(s_err)


def _load_model_suppressed(model_key: str):
    from mlx_lm import load
    saved = _suppress_fds()
    try:
        model, tokenizer = load(MODELS[model_key]["id"])
    finally:
        _restore_fds(*saved)
    return model, tokenizer


def _load_model_with_progress(model_key: str):
    """Load model in a background thread while showing a progress bar."""
    import threading

    entry = MODELS[model_key]
    label = entry["label"]

    result = {"model": None, "tokenizer": None, "error": None}
    done = threading.Event()

    def _do_load():
        try:
            from mlx_lm import load
            try:
                result["model"], result["tokenizer"] = load(entry["id"])
            except ValueError as ve:
                if "parameters not in model" not in str(ve):
                    raise
                # Multimodal model — reload ignoring extra vision weights
                from mlx_lm.utils import _download, load_model, load_tokenizer
                model_path = _download(entry["id"])
                model, config = load_model(model_path, lazy=False, strict=False)
                tokenizer = load_tokenizer(model_path)
                result["model"] = model
                result["tokenizer"] = tokenizer
        except Exception as e:
            result["error"] = e
        finally:
            done.set()

    # Suppress mlx_lm noise but keep real stdout fd for our progress bar
    null_fd = os.open(os.devnull, os.O_WRONLY)
    real_out = os.dup(1)
    real_err = os.dup(2)
    os.dup2(null_fd, 1)
    os.dup2(null_fd, 2)
    os.close(null_fd)

    thread = threading.Thread(target=_do_load, daemon=True)
    thread.start()

    BAR_W  = 30
    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    tick   = 0

    while not done.wait(0.08):
        f    = FRAMES[tick % len(FRAMES)]
        pct  = 1.0 - 1.0 / (1.0 + tick * 0.025)
        pct  = min(pct, 0.95)
        fill = int(BAR_W * pct)
        bar  = "█" * fill + "░" * (BAR_W - fill)
        line = (
            f"\r  {ui.MG}{f}{ui.RS} {ui.GR}Loading {label}{ui.RS}"
            f"  [{ui.AC}{bar}{ui.RS}] {int(pct * 100):>2}%  "
        )
        os.write(real_out, line.encode())
        tick += 1

    # Complete — full bar
    bar  = "█" * BAR_W
    line = (
        f"\r\033[2K  {ui.AC}✓{ui.RS} {ui.GR}{label} loaded{ui.RS}"
        f"  [{ui.AC}{bar}{ui.RS}] 100%\n"
    )
    os.write(real_out, line.encode())

    # Restore fds
    os.dup2(real_out, 1)
    os.dup2(real_err, 2)
    os.close(real_out)
    os.close(real_err)

    if result["error"]:
        raise result["error"]
    return result["model"], result["tokenizer"]


# ── Model picker ──────────────────────────────────────────────────
def _model_picker(
    hw: dict,
    cfg: dict,
    unfiltered: bool = False,
    forced_key: str | None = None,
) -> str:
    """Display interactive model picker. Returns chosen model key."""
    if forced_key and forced_key in MODELS:
        return forced_key

    ram      = hw["ram"]
    chip     = hw["chip"]
    rec_key  = recommend_model(hw)
    default  = cfg.get("model", rec_key)
    models   = available_models(ram, unfiltered=unfiltered)

    if not models:
        return rec_key

    # Ensure default exists in the list
    keys = [k for k, _ in models]
    if default not in keys:
        default = rec_key if rec_key in keys else keys[0]

    # ── Draw picker ───────────────────────────────────────────────
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

    W  = min(ui.width() - 4, 60)
    AC = ui.AC; GR = ui.GR; BD = ui.BD; RS = ui.RS; RD = ui.RD; YL = ui.YL

    # Pressure dot color
    _pressure_col = {"low": AC, "medium": YL, "high": RD}
    p_col = _pressure_col.get(hw.get("pressure", "low"), GR)

    show_all = False

    while True:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

        print(f"\n  {AC}◈  Select model{RS}")
        print()
        avail_str = f"{hw['ram_available']}GB available"
        disk_str  = f"Disk {hw['disk_free']}GB free"
        print(f"  {GR}{chip} · {ram}GB total · {avail_str} · {disk_str}{RS}")
        pressure_line = f"  {GR}Pressure: {p_col}●{RS}{GR} {hw.get('pressure','low')}"
        if hw.get("swap_used", 0) > 0.1:
            pressure_line += f"  Swap: {hw['swap_used']}GB"
        print(pressure_line + RS)
        print(f"  {GR}{'─' * W}{RS}")

        model_nums: dict[str, str] = {}
        num_counter = 1

        if show_all:
            # Full list grouped by category
            groups = grouped_by_category([(k, v) for k, v in models])
            for cat, cat_models in groups:
                cat_label = CATEGORY_LABELS.get(cat, cat.title())
                print(f"\n  {BD}{cat_label}{RS}")
                for key, entry in cat_models:
                    num = str(num_counter)
                    model_nums[num] = key
                    num_counter += 1
                    _print_model_row(num, key, entry, default, rec_key, hw, unfiltered)
        else:
            # Compact view — top picks per category
            picks = top_picks(models, hw)

            # Recommended first
            rec_picks = [(k, v) for k, v in picks if k == rec_key]
            other_picks = [(k, v) for k, v in picks if k != rec_key]

            if rec_picks:
                print(f"\n  {BD}★ Recommended{RS}")
                for key, entry in rec_picks:
                    num = str(num_counter)
                    model_nums[num] = key
                    num_counter += 1
                    _print_model_row(num, key, entry, default, rec_key, hw, unfiltered)

            # Group remaining picks by category
            pick_groups = grouped_by_category(other_picks)
            for cat, cat_models in pick_groups:
                cat_label = CATEGORY_LABELS.get(cat, cat.title())
                print(f"\n  {BD}{cat_label}{RS}")
                for key, entry in cat_models:
                    num = str(num_counter)
                    model_nums[num] = key
                    num_counter += 1
                    _print_model_row(num, key, entry, default, rec_key, hw, unfiltered)

            total_available = len(models)
            shown = len(picks)
            if total_available > shown:
                print(f"\n  {AC}a{RS}  {GR}Show all models ({total_available} available){RS}")

        print(f"\n  {GR}{'─' * W}{RS}")

        # ── Input (with optional countdown) ──────────────────────
        TIMEOUT = float(os.environ.get("LOCALAI_PICK_TIMEOUT", "0"))
        def_num = next((n for n, k in model_nums.items() if k == default), "1")
        chosen  = None

        if TIMEOUT <= 0 or show_all:
            sys.stdout.write(f"  {GR}Enter number [{def_num}]: {RS}")
            sys.stdout.flush()
            try:
                ch = sys.stdin.readline().strip().lower()
            except (EOFError, KeyboardInterrupt):
                ch = ""
            if ch == "a" and not show_all:
                show_all = True
                continue
            if ch in model_nums:
                chosen = model_nums[ch]
            else:
                chosen = default
        else:
            deadline = time.time() + TIMEOUT
            while time.time() < deadline:
                remaining = max(1, int(deadline - time.time()) + 1)
                sys.stdout.write(
                    f"\r  {GR}Enter number [{def_num}] — auto in {remaining}s …{RS}  "
                )
                sys.stdout.flush()
                rlist, _, _ = select.select([sys.stdin], [], [], 0.25)
                if rlist:
                    ch = sys.stdin.readline().strip().lower()
                    if ch == "a":
                        show_all = True
                        break
                    if ch in model_nums:
                        chosen = model_nums[ch]
                    else:
                        chosen = default
                    break
            if show_all:
                continue

        sys.stdout.write("\r" + " " * 72 + "\r")
        sys.stdout.flush()
        return chosen if chosen else default


def _print_model_row(
    num: str, key: str, entry: dict,
    default: str, rec_key: str, hw: dict, unfiltered: bool,
) -> None:
    """Print a single model row in the picker."""
    AC = ui.AC; GR = ui.GR; BD = ui.BD; RS = ui.RS; RD = ui.RD; YL = ui.YL

    is_def    = key == default
    is_unf    = entry["profile"] == "unfiltered"
    arrow     = f"  {AC}▸{RS}" if is_def else "   "
    rec_tag   = "  ← recommended" if key == rec_key and not unfiltered else ""
    num_col   = RD if is_unf else AC

    fit       = model_fit(key, hw)
    fit_col   = AC if fit.startswith("✓") else YL if fit.startswith("⚠") else RD
    label_col = GR if fit.startswith("✗") else BD
    disk_warn = "" if disk_ok(key, hw) else f"  {YL}⚠ low disk{RS}"

    desc = entry.get("desc", "")
    print(
        f"{arrow} {num_col}{num:>2}{RS}  {label_col}{entry['label']:<20}{RS}"
        f"  {GR}{entry['size']:<12}{RS}"
        f"  {GR}{entry['dl_size']:<6}{RS}"
        f"  {GR}[{entry['min_ram']}GB]{RS}"
        f"  {fit_col}{fit}{RS}"
        f"{disk_warn}"
        f"  {GR}{desc}{RS}"
        f"{GR}{rec_tag}{RS}"
    )


# ── Memory cleanup flow ───────────────────────────────────────────
def _memory_hogs() -> list[tuple[str, float]]:
    """Return up to 3 closeable apps sorted by RSS descending."""
    seen    : set[str] = set()
    results : list[tuple[str, float]] = []
    for proc in psutil.process_iter(["name", "memory_info"]):
        try:
            name = proc.info["name"]
            rss  = proc.info["memory_info"].rss
            if name in CLOSEABLE_APPS and name not in seen:
                results.append((name, rss / (1024 ** 3)))
                seen.add(name)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:3]


def _maybe_free_memory(model_key: str, hw: dict) -> None:
    """Offer to quit memory-heavy apps before loading if available RAM is tight."""
    needed = MODELS[model_key]["min_ram"]
    avail  = hw["ram_available"]
    if avail >= needed - 0.5:
        return

    label = MODELS[model_key]["label"]
    YL = ui.YL; GR = ui.GR; RS = ui.RS; AC = ui.AC; BD = ui.BD

    print(f"\n  {YL}⚠  {label} needs ~{needed}GB — only {avail}GB available right now.{RS}\n")

    hogs = _memory_hogs()
    if hogs:
        print(f"  Memory-heavy apps running:")
        for name, gb in hogs:
            print(f"    · {BD}{name:<20}{RS} {GR}{gb:.1f}GB{RS}")
        print()
        print(
            f"  Quit them to free RAM?  "
            f"{AC}y{RS} / {GR}n{RS} / {GR}skip{RS} {GR}›{RS} ",
            end="", flush=True,
        )
        try:
            ans = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            ans = "n"

        if ans == "y":
            for name, _ in hogs:
                try:
                    subprocess.run(
                        ["osascript", "-e", f'tell application "{name}" to quit'],
                        timeout=3,
                        capture_output=True,
                    )
                except Exception:
                    pass

            print(f"\n  {GR}Waiting for apps to close …{RS}")
            time.sleep(2)
            try:
                import mlx.core as mx
                try:
                    mx.clear_cache()
                except AttributeError:
                    mx.metal.clear_cache()
            except Exception:
                pass

            new_avail = round(psutil.virtual_memory().available / (1024 ** 3), 1)
            if new_avail >= needed - 0.5:
                print(f"  {AC}Available after cleanup: {new_avail}GB — good to go.{RS}\n")
            else:
                print(f"  {YL}Available after cleanup: {new_avail}GB — still tight, may swap.{RS}\n")
        else:
            print(f"  {GR}Proceeding — may swap if RAM is tight.{RS}\n")
    else:
        print(f"  {GR}No closeable apps found. Proceeding — may swap.{RS}\n")


# ── First-run wizard ──────────────────────────────────────────────
def _first_run_wizard(hw: dict, cfg: dict) -> None:
    """Interactive first-run setup. Modifies cfg in place."""
    global CURRENT_PERSONALITY

    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

    chip = hw["chip"]
    ram  = hw["ram"]
    macos = hw["os"]
    rec  = recommend_model(hw)
    rec_entry = MODELS[rec]

    AC = ui.AC; GR = ui.GR; BD = ui.BD; RS = ui.RS; CY = ui.CY

    print(f"\n  {BD}{AC}localai · offline LLM for Mac{RS}")
    print(f"  {'─' * 37}")
    print(f"  Hardware  {chip} · {ram}GB · macOS {macos}")
    print(f"  Network   offline — no cloud, no telemetry")
    print()
    print(f"  Recommended: {BD}{rec_entry['label']}{RS} ({rec_entry['size']}, {rec_entry['dl_size']} download)")
    print(f"  {GR}Press Enter to continue or type a number to pick:{RS}")
    print()

    # Show brief model list
    models = available_models(ram, unfiltered=False)
    model_nums: dict[str, str] = {}
    for i, (key, entry) in enumerate(models, 1):
        num = str(i)
        model_nums[num] = key
        rec_tag = "  ← recommended" if key == rec else ""
        print(f"     {num}  {entry['label']:<22}  {entry['size']:<10}  {entry['dl_size']}{GR}{rec_tag}{RS}")

    print()
    print(f"  {GR}Choose [{next((n for n, k in model_nums.items() if k == rec), '1')}]:{RS} ", end="", flush=True)
    try:
        ch = input().strip()
    except (EOFError, KeyboardInterrupt):
        ch = ""

    if ch in model_nums:
        cfg["model"] = model_nums[ch]
    else:
        cfg["model"] = rec

    # ── Personality ───────────────────────────────────────────────
    print()
    print(f"  Personality:")
    for n, p in PERSONALITIES.items():
        print(f"    {n}  {p['name']:<10}  {GR}{p['desc']}{RS}")
    print(f"  {GR}Choose [1]:{RS} ", end="", flush=True)
    try:
        pc = input().strip()
    except (EOFError, KeyboardInterrupt):
        pc = ""

    if pc.isdigit() and 1 <= int(pc) <= len(PERSONALITIES):
        CURRENT_PERSONALITY = int(pc)
        cfg["personality"] = CURRENT_PERSONALITY

    cfg["first_run"] = False
    print(f"\n  {AC}✓{RS} {GR}Ready. Type {AC}llm{GR} any time to start. {AC}q{GR} to quit.{RS}\n")


# ── Unfiltered gate ───────────────────────────────────────────────
def _unfiltered_gate() -> bool:
    """Show warning and require UNLOCK. Returns True if confirmed."""
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()
    RD = ui.RD; RS = ui.RS; GR = ui.GR

    W   = 54
    bar = "═" * W

    def _row(content: str) -> str:
        import re
        visible = len(re.sub(r'\033\[[0-9;]*m', '', content))
        pad = " " * max(0, W - visible)
        return f"{RD}  ║{RS}{content}{pad}{RD}║{RS}"

    print(f"\n{RD}  ╔{bar}╗{RS}")
    print(_row(f"  ⚠   U N F I L T E R E D   M O D E"))
    print(f"{RD}  ╠{bar}╣{RS}")
    print(_row(""))
    print(_row(f"  {GR}Unaligned models with no safety filters.{RS}"))
    print(_row(f"  {GR}Output may be harmful, false, or offensive.{RS}"))
    print(_row(""))
    print(_row(f"  {GR}Runs fully offline on your hardware only.{RS}"))
    print(_row(f"  {GR}You are solely responsible for all output.{RS}"))
    print(_row(""))
    print(f"{RD}  ╚{bar}╝{RS}")
    print(f"\n  {GR}Type {RD}UNLOCK{RS}{GR} to continue, anything else exits:{RS} ", end="", flush=True)
    try:
        confirm = input().strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if confirm != "UNLOCK":
        print(f"\n  {GR}Exiting.{RS}\n")
        return False
    return True


# ── Chat loop helpers ─────────────────────────────────────────────
def _active_prompt() -> str:
    base = PERSONALITIES[CURRENT_PERSONALITY]["prompt"]
    if CUSTOM_INSTRUCTIONS:
        base += f"\n\nAdditional instructions:\n{CUSTOM_INSTRUCTIONS}"
    return base


def _trim_conversation(conv: list[dict], max_turns: int = 20) -> None:
    """Keep only the last max_turns pairs to avoid exceeding KV cache.

    Removes oldest user/assistant pairs from the front, always
    keeping at least the most recent max_turns messages.
    """
    while len(conv) > max_turns * 2:
        conv.pop(0)  # remove oldest user
        if conv and conv[0]["role"] == "assistant":
            conv.pop(0)  # remove its paired assistant
    # Ensure conversation starts with a user message (not orphaned assistant)
    while conv and conv[0]["role"] != "user":
        conv.pop(0)


def _sep(cols: int) -> None:
    ui.GR  # ensure import
    print(f"  {ui.GR}{'╌' * min(cols - 4, 54)}{ui.RS}")


def _draw_chat_header(model_key: str, hw: dict) -> None:
    global _focus_header_done
    if UI_MODE == "zen":
        return
    if UI_MODE == "focus" and _focus_header_done:
        return
    _focus_header_done = True
    entry = MODELS.get(model_key, list(MODELS.values())[0])
    chip  = hw["chip"]
    ram   = hw["ram"]
    is_unf = entry["profile"] == "unfiltered"
    col   = ui.RD if is_unf else ui.AC
    tag   = "· unfiltered" if is_unf else "· offline"

    print(f"\n  {ui.BD}{col}{entry['label']}{ui.RS}  {ui.GR}· {entry['size']} · {chip} {ram}GB {tag}{ui.RS}")
    try:
        import mlx.core as mx
        gpu_gb = mx.get_active_memory() / (1024 ** 3)
    except Exception:
        gpu_gb = 0.0
    mem      = psutil.virtual_memory()
    cpu_pct  = psutil.cpu_percent(interval=0)
    ram_used = mem.used / (1024 ** 3)
    ram_tot  = mem.total / (1024 ** 3)
    pvt      = f"  {ui.RD}[pvt]{ui.RS}" if PRIVACY_MODE else ""
    print(
        f"  {ui.GR}CPU {cpu_pct:.0f}%  ·  "
        f"GPU {gpu_gb:.1f}GB  ·  "
        f"RAM {ram_used:.1f}/{ram_tot:.0f}GB{pvt}{ui.RS}"
    )

    W   = min(ui.width() - 4, 54)
    bar = f"  {ui.GR}{'─' * W}{ui.RS}"
    print(bar)
    print(
        f"  {ui.AC}q{ui.RS} quit   "
        f"{ui.AC}r{ui.RS} reset   "
        f"{ui.AC}s{ui.RS} settings   "
        f"{ui.AC}rensa{ui.RS} clear   "
        f"{ui.AC}h{ui.RS} help"
    )
    print(bar)
    print()


# ── Settings menu ─────────────────────────────────────────────────
def _settings_menu() -> None:
    global TEMP, CURRENT_PERSONALITY, CUSTOM_INSTRUCTIONS
    global SHOW_STATS, UI_MODE, PRIVACY_MODE, LOG_SESSIONS

    while True:
        p     = PERSONALITIES[CURRENT_PERSONALITY]
        pvt_s = f"{ui.AC}on{ui.RS}"  if PRIVACY_MODE else f"{ui.GR}off{ui.RS}"
        log_s = f"{ui.AC}on{ui.RS}"  if LOG_SESSIONS  else f"{ui.GR}off{ui.RS}"
        ci_s  = (CUSTOM_INSTRUCTIONS[:20] + "…") if len(CUSTOM_INSTRUCTIONS) > 20 else (CUSTOM_INSTRUCTIONS or "—")
        W     = min(ui.width() - 4, 54)

        print(f"\n  {ui.BD}{'═' * W}{ui.RS}")
        print(f"  {ui.BD}{ui.AC}  S E T T I N G S{ui.RS}")
        print(f"  {ui.BD}{'═' * W}{ui.RS}")
        print(f"  {ui.AC}1{ui.RS}  Temperature     {ui.AC}{TEMP}{ui.RS}")
        print(f"  {ui.AC}2{ui.RS}  Personality     {ui.AC}{p['name']}{ui.RS}")
        print(f"  {ui.AC}3{ui.RS}  Stats           {ui.AC}{SHOW_STATS}{ui.RS}")
        print(f"  {ui.AC}4{ui.RS}  UI mode         {ui.AC}{UI_MODE}{ui.RS}")
        print(f"  {ui.AC}5{ui.RS}  Privacy         {pvt_s}  Log: {log_s}")
        print(f"  {ui.AC}6{ui.RS}  Custom          {ui.GR}{ci_s}{ui.RS}")
        plg_s = _plugin_summary()
        print(f"  {ui.AC}7{ui.RS}  Plugins         {ui.GR}{plg_s}{ui.RS}")
        print(f"  {'═' * W}")
        print(f"  {ui.GR}Enter 1–7 (Enter to go back):{ui.RS} ", end="", flush=True)

        try:
            ch = input().strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

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
        elif ch == "7":
            _settings_plugins()
        else:
            break

    print()


def _settings_temp() -> None:
    global TEMP
    print(f"\n  {ui.BD}Temperature{ui.RS}  {ui.GR}(creativity / randomness){ui.RS}\n")
    for i, (val, desc) in enumerate(TEMP_PRESETS, 1):
        mark = f"  {ui.AC}*{ui.RS}" if abs(val - TEMP) < 0.01 else "   "
        print(f"{mark} {ui.AC}{i}{ui.RS}  {ui.BD}{val}{ui.RS}  {ui.GR}{desc}{ui.RS}")
    print(f"\n  {ui.GR}Enter number (Enter to cancel):{ui.RS} ", end="", flush=True)
    try:
        ch = input().strip()
    except (EOFError, KeyboardInterrupt):
        print(); return
    if ch.isdigit() and 1 <= int(ch) <= len(TEMP_PRESETS):
        TEMP = TEMP_PRESETS[int(ch) - 1][0]
        print(f"  {ui.GR}Temperature → {ui.AC}{TEMP}{ui.RS}\n")
    else:
        print(f"  {ui.GR}Unchanged.{ui.RS}\n")


def _settings_persona() -> None:
    global CURRENT_PERSONALITY
    print(f"\n  {ui.BD}Personality{ui.RS}  {ui.GR}(system prompt preset){ui.RS}\n")
    for n, p in PERSONALITIES.items():
        mark = f"  {ui.AC}*{ui.RS}" if n == CURRENT_PERSONALITY else "   "
        print(f"{mark} {ui.AC}{n}{ui.RS}  {ui.BD}{p['name']:<10}{ui.RS}  {ui.GR}{p['desc']}{ui.RS}")
    print(f"\n  {ui.GR}Enter number (Enter to cancel):{ui.RS} ", end="", flush=True)
    try:
        ch = input().strip()
    except (EOFError, KeyboardInterrupt):
        print(); return
    if ch.isdigit() and 1 <= int(ch) <= len(PERSONALITIES):
        CURRENT_PERSONALITY = int(ch)
        print(f"  {ui.GR}Personality → {ui.AC}{PERSONALITIES[CURRENT_PERSONALITY]['name']}{ui.RS}\n")
    else:
        print(f"  {ui.GR}Unchanged.{ui.RS}\n")


def _settings_stats() -> None:
    global SHOW_STATS
    opts = [
        ("compact", "t/s + GPU only"),
        ("full",    "Tokens · Latency · t/s · Temp · GPU"),
        ("off",     "No stats"),
    ]
    print(f"\n  {ui.BD}Stats{ui.RS}\n")
    for i, (val, desc) in enumerate(opts, 1):
        mark = f"  {ui.AC}*{ui.RS}" if val == SHOW_STATS else "   "
        print(f"{mark} {ui.AC}{i}{ui.RS}  {ui.BD}{val:<8}{ui.RS}  {ui.GR}{desc}{ui.RS}")
    print(f"\n  {ui.GR}Enter 1–3 (Enter to cancel):{ui.RS} ", end="", flush=True)
    try:
        ch = input().strip()
    except (EOFError, KeyboardInterrupt):
        print(); return
    if ch.isdigit() and 1 <= int(ch) <= len(opts):
        SHOW_STATS = opts[int(ch) - 1][0]
        print(f"  {ui.GR}Stats → {ui.AC}{SHOW_STATS}{ui.RS}\n")
    else:
        print(f"  {ui.GR}Unchanged.{ui.RS}\n")


def _settings_ui_mode() -> None:
    global UI_MODE
    opts = [
        ("normal", "Header + stats + commands"),
        ("zen",    "No header, no stats — just text"),
        ("focus",  "Header at start only"),
    ]
    print(f"\n  {ui.BD}UI Mode{ui.RS}\n")
    for i, (val, desc) in enumerate(opts, 1):
        mark = f"  {ui.AC}*{ui.RS}" if val == UI_MODE else "   "
        print(f"{mark} {ui.AC}{i}{ui.RS}  {ui.BD}{val:<8}{ui.RS}  {ui.GR}{desc}{ui.RS}")
    print(f"\n  {ui.GR}Enter 1–3 (Enter to cancel):{ui.RS} ", end="", flush=True)
    try:
        ch = input().strip()
    except (EOFError, KeyboardInterrupt):
        print(); return
    if ch.isdigit() and 1 <= int(ch) <= len(opts):
        UI_MODE = opts[int(ch) - 1][0]
        print(f"  {ui.GR}UI mode → {ui.AC}{UI_MODE}{ui.RS}\n")
    else:
        print(f"  {ui.GR}Unchanged.{ui.RS}\n")


def _settings_privacy() -> None:
    global PRIVACY_MODE, LOG_SESSIONS
    pvt = f"{ui.AC}on{ui.RS}" if PRIVACY_MODE else f"{ui.GR}off{ui.RS}"
    log = f"{ui.AC}on{ui.RS}" if LOG_SESSIONS  else f"{ui.GR}off{ui.RS}"
    print(f"\n  {ui.BD}Privacy & History{ui.RS}\n")
    print(f"  {ui.AC}1{ui.RS}  Privacy Mode  [{pvt}]  {ui.GR}Nothing saved to disk at exit{ui.RS}")
    print(f"  {ui.AC}2{ui.RS}  Session Log   [{log}]  {ui.GR}Save turns to ~/.localai/logs/{ui.RS}")
    print(f"  {ui.AC}3{ui.RS}  View logs          {ui.GR}List session files{ui.RS}")
    print(f"  {ui.AC}4{ui.RS}  Delete all logs    {ui.GR}Wipe ~/.localai/logs/{ui.RS}")
    print(f"\n  {ui.GR}Enter 1–4 (Enter to go back):{ui.RS} ", end="", flush=True)
    try:
        ch = input().strip()
    except (EOFError, KeyboardInterrupt):
        print(); return

    if ch == "1":
        PRIVACY_MODE = not PRIVACY_MODE
        set_logging(LOG_SESSIONS and not PRIVACY_MODE)
        state = f"{ui.AC}on{ui.RS}" if PRIVACY_MODE else f"{ui.GR}off{ui.RS}"
        print(f"  {ui.GR}Privacy → {state}{ui.RS}\n")
    elif ch == "2":
        LOG_SESSIONS = not LOG_SESSIONS
        set_logging(LOG_SESSIONS and not PRIVACY_MODE)
        state = f"{ui.AC}on{ui.RS}" if LOG_SESSIONS else f"{ui.GR}off{ui.RS}"
        print(f"  {ui.GR}Session log → {state}{ui.RS}\n")
    elif ch == "3":
        logs = list_logs()
        if not logs:
            print(f"  {ui.GR}No saved sessions.{ui.RS}\n")
        else:
            print(f"\n  {ui.GR}Saved sessions ({len(logs)}):{ui.RS}")
            for lf in logs[:10]:
                print(f"    {ui.AC}{lf.name}{ui.RS}")
            if len(logs) > 10:
                print(f"    {ui.GR}… and {len(logs)-10} more{ui.RS}")
            print()
    elif ch == "4":
        print(f"  {ui.YL}Delete all logs? y/n:{ui.RS} ", end="", flush=True)
        try:
            confirm = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(); return
        if confirm in ("y", "yes"):
            count = delete_all_logs()
            print(f"  {ui.GR}Deleted {ui.AC}{count}{ui.RS}{ui.GR} log files.{ui.RS}\n")
        else:
            print(f"  {ui.GR}Cancelled.{ui.RS}\n")
    else:
        print(f"  {ui.GR}Unchanged.{ui.RS}\n")


def _settings_custom() -> None:
    global CUSTOM_INSTRUCTIONS
    print(f"\n  {ui.BD}Custom Instructions{ui.RS}  {ui.GR}(appended to system prompt, session only){ui.RS}")
    if CUSTOM_INSTRUCTIONS:
        print(f"  {ui.GR}Current: {ui.AC}{CUSTOM_INSTRUCTIONS[:60]}{ui.RS}")
    print(f"  {ui.GR}Type instructions (Enter = keep, 'clear' = remove):{ui.RS}")
    print(f"  {ui.AC}>{ui.RS} ", end="", flush=True)
    try:
        txt = input().strip()
    except (EOFError, KeyboardInterrupt):
        print(); return
    if not txt:
        print(f"  {ui.GR}Unchanged.{ui.RS}\n")
    elif txt.lower() == "clear":
        CUSTOM_INSTRUCTIONS = ""
        print(f"  {ui.GR}Custom instructions cleared.{ui.RS}\n")
    else:
        CUSTOM_INSTRUCTIONS = txt
        print(f"  {ui.GR}Set.{ui.RS}\n")


def _plugin_summary() -> str:
    """Short summary for settings menu line."""
    installed = plugins.installed_plugins(CFG)
    if not installed:
        return "none"
    enabled = plugins.enabled_plugins(CFG)
    return f"{len(enabled)} active"


def _settings_plugins() -> None:
    """Plugin management sub-menu."""
    while True:
        installed = plugins.installed_plugins(CFG)
        enabled   = plugins.enabled_plugins(CFG)
        W = min(ui.width() - 4, 54)

        print(f"\n  {ui.BD}{'═' * W}{ui.RS}")
        print(f"  {ui.BD}{ui.AC}  P L U G I N S{ui.RS}")
        print(f"  {ui.BD}{'═' * W}{ui.RS}")

        all_names: list[str] = []

        # ── Installed section ─────────────────────────────────
        if installed:
            print(f"\n  {ui.BD}Installed:{ui.RS}")
            for name in installed:
                idx   = len(all_names) + 1
                all_names.append(name)
                entry = plugins.REGISTRY.get(name, {})
                label = entry.get("label", name)
                desc  = entry.get("desc", "")
                state = f"{ui.AC}on{ui.RS}" if name in enabled else f"{ui.GR}off{ui.RS}"
                print(
                    f"    {ui.AC}{idx}{ui.RS}  {ui.BD}{label:<22}{ui.RS}"
                    f" [{state}]  {ui.GR}{desc}{ui.RS}"
                )

        # ── Available section ─────────────────────────────────
        available = [n for n in plugins.REGISTRY if n not in installed]
        if available:
            print(f"\n  {ui.BD}Available:{ui.RS}")
            for name in available:
                idx   = len(all_names) + 1
                all_names.append(name)
                entry = plugins.REGISTRY[name]
                desc  = entry.get("desc", "")
                deps  = f"  {ui.GR}({', '.join(entry['deps'])}){ui.RS}" if entry["deps"] else ""
                print(
                    f"    {ui.AC}{idx}{ui.RS}  {entry['label']:<22}"
                    f"  {ui.GR}{desc}{deps}{ui.RS}"
                )

        print(f"\n  {ui.GR}{'─' * W}{ui.RS}")
        print(
            f"  {ui.GR}Enter number to install/toggle, "
            f"{ui.AC}u{ui.RS}{ui.GR} to uninstall, Enter to go back:{ui.RS} ",
            end="", flush=True,
        )

        try:
            ch = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(); break

        if not ch:
            break

        if ch == "u":
            _plugin_uninstall_menu(installed)
            continue

        if ch.isdigit():
            idx = int(ch)
            if 1 <= idx <= len(all_names):
                name = all_names[idx - 1]
                if name in installed:
                    # Toggle on/off
                    new_state = plugins.toggle_plugin(name, CFG)
                    state_str = f"{ui.AC}on{ui.RS}" if new_state else f"{ui.GR}off{ui.RS}"
                    label = plugins.REGISTRY.get(name, {}).get("label", name)
                    print(f"\n  {ui.GR}{label} → {state_str}{ui.RS}")
                    plugins.load_plugins(CFG)
                else:
                    # Install
                    label = plugins.REGISTRY.get(name, {}).get("label", name)
                    print(f"\n  {ui.MG}Installing {label} …{ui.RS}", flush=True)
                    ok, msg = plugins.install_plugin(name, CFG)
                    if ok:
                        print(f"  {ui.AC}✓{ui.RS} {ui.GR}{msg}{ui.RS}")
                        plugins.load_plugins(CFG)
                    else:
                        print(f"  {ui.RD}✗{ui.RS} {ui.GR}{msg}{ui.RS}")

    print()


def _plugin_uninstall_menu(installed: list[str]) -> None:
    """Show uninstall sub-menu."""
    if not installed:
        print(f"\n  {ui.GR}No plugins installed.{ui.RS}")
        return
    print(f"\n  {ui.BD}Uninstall which?{ui.RS}")
    for i, name in enumerate(installed, 1):
        label = plugins.REGISTRY.get(name, {}).get("label", name)
        print(f"    {ui.AC}{i}{ui.RS}  {label}")
    print(f"  {ui.GR}Enter number (Enter to cancel):{ui.RS} ", end="", flush=True)
    try:
        ch = input().strip()
    except (EOFError, KeyboardInterrupt):
        print(); return
    if ch.isdigit() and 1 <= int(ch) <= len(installed):
        name = installed[int(ch) - 1]
        ok, msg = plugins.uninstall_plugin(name, CFG)
        print(f"  {ui.GR}{msg}{ui.RS}")
    else:
        print(f"  {ui.GR}Cancelled.{ui.RS}")


# ── Main ──────────────────────────────────────────────────────────
def main() -> None:
    global CFG, MSGS, TEMP, SHOW_STATS, UI_MODE
    global PRIVACY_MODE, LOG_SESSIONS, CURRENT_PERSONALITY

    # ── Args ─────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(add_help=False, prog="llm")
    parser.add_argument("--unfiltered", action="store_true")
    parser.add_argument("--model",      default=None, metavar="KEY")
    parser.add_argument("--voice",      action="store_true")
    parser.add_argument("--zen",        action="store_true")
    parser.add_argument("--focus",      action="store_true")
    args, unknown = parser.parse_known_args()

    for flag in unknown:
        print(f"  [warning] unknown flag ignored: {flag}", file=sys.stderr)

    # ── Config ───────────────────────────────────────────────────
    CFG                 = cfg_load()
    TEMP                = float(CFG.get("temp",         TEMP))
    CURRENT_PERSONALITY = int(CFG.get("personality",    CURRENT_PERSONALITY))
    SHOW_STATS          = str(CFG.get("stats",          SHOW_STATS))
    PRIVACY_MODE        = bool(CFG.get("privacy_mode",  PRIVACY_MODE))
    LOG_SESSIONS        = bool(CFG.get("log_sessions",  LOG_SESSIONS))
    UI_MODE             = str(CFG.get("ui_mode",        UI_MODE))

    if args.zen:
        UI_MODE = "zen"
    elif args.focus:
        UI_MODE = "focus"

    ui.apply_theme(CFG.get("theme", "ocean"))
    set_logging(LOG_SESSIONS and not PRIVACY_MODE)
    plugins.load_plugins(CFG)

    # ── PID + cleanup ────────────────────────────────────────────
    _write_pid()

    t0   = time.time()
    MSGS = 0

    model_key = CFG.get("model", "")   # sentinel — overwritten by picker; guards atexit on early exit

    def _save_cfg() -> None:
        if PRIVACY_MODE:
            return
        cfg_save({
            **CFG,
            "model":            model_key,
            "theme":            ui.current_theme(),
            "personality":      CURRENT_PERSONALITY,
            "temp":             TEMP,
            "stats":            SHOW_STATS,
            "ui_mode":          UI_MODE,
            "log_sessions":     LOG_SESSIONS,
            "privacy_mode":     PRIVACY_MODE,
            "first_run":        False,
            "plugins":          CFG.get("plugins", []),
            "disabled_plugins": CFG.get("disabled_plugins", []),
        })

    def _cleanup() -> None:
        nonlocal model, tokenizer, cache, sampler
        sys.stdout.write("\033[?25h\033[0m")
        dur   = time.time() - t0
        m, s  = int(dur // 60), int(dur % 60)
        print()
        print(f"  {ui.GR}{'─' * 40}{ui.RS}")
        print(f"  {ui.GR}Session ended · {ui.AC}{MSGS}{ui.GR} messages · {ui.AC}{m}m {s}s{ui.RS}")
        if PRIVACY_MODE:
            print(f"  {ui.GR}No data saved.{ui.RS}")
        print()
        _save_cfg()
        model = tokenizer = cache = sampler = None  # drop GPU refs
        gc.collect()
        try:
            import mlx.core as mx
            try:
                mx.clear_cache()
            except AttributeError:
                mx.metal.clear_cache()
        except Exception:
            pass
        _remove_pid()

    atexit.register(_cleanup)

    def _sig_handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT,  _sig_handler)

    # ── Hardware ─────────────────────────────────────────────────
    hw = hardware_summary()

    # ── First-run wizard ─────────────────────────────────────────
    if CFG.get("first_run", True):
        _first_run_wizard(hw, CFG)
        cfg_save(CFG)

    # ── Unfiltered gate ──────────────────────────────────────────
    if args.unfiltered:
        if not _unfiltered_gate():
            sys.exit(0)

    # ── Model picker ─────────────────────────────────────────────
    model_key = _model_picker(
        hw, CFG,
        unfiltered=args.unfiltered,
        forced_key=args.model,
    )

    # ── Memory cleanup offer ─────────────────────────────────────
    _maybe_free_memory(model_key, hw)

    # ── Load model ───────────────────────────────────────────────
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

    model, tokenizer = _load_model_with_progress(model_key)

    # Lazy mlx import (already loaded by mlx_lm)
    import mlx.core as mx
    from mlx_lm.generate    import stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    entry   = MODELS[model_key]
    MAX_KV  = entry["kv"]
    MAX_TOK = entry["tokens"]

    cache   = make_prompt_cache(model, MAX_KV)
    sampler = make_sampler(temp=TEMP)

    # Metal warmup — discards 1-2 tokens so first real response isn't cold
    try:
        warm = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hi"}],
            tokenize=False, add_generation_prompt=True,
        )
        for _ in stream_generate(model, tokenizer, warm, max_tokens=2, sampler=sampler):
            pass
    except Exception:
        pass

    # ── Chat UI ──────────────────────────────────────────────────
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()
    _draw_chat_header(model_key, hw)

    # Conversation history (maintained across turns for multi-turn memory)
    conversation: list[dict] = []

    # Voice setup
    VOICE_MODE = args.voice
    voice_api  = None

    if args.voice:
        from voice import check_available, install_instructions, push_to_talk
        voice_api = {
            "check_available":    check_available,
            "install_instructions": install_instructions,
            "push_to_talk":       push_to_talk,
        }
        if not check_available():
            print(f"  {ui.YL}⚠  Voice requires: {install_instructions()}{ui.RS}\n")
            VOICE_MODE = False

    # ── Chat loop ────────────────────────────────────────────────
    cols = ui.width()

    while True:
        cols = ui.width()

        if VOICE_MODE and voice_api:
            print(f"  {ui.BD}{ui.AC}you{ui.RS} {ui.GR}›{ui.RS} ", end="", flush=True)
            user = voice_api["push_to_talk"](
                prompt_fn=lambda m: sys.stdout.write(f"\r{m}\n")
            )
            if not user:
                sys.stdout.write("\033[A\033[2K")
                sys.stdout.flush()
                continue
            print(f"  {ui.BD}{ui.AC}you{ui.RS} {ui.GR}›{ui.RS} {ui.BD}{user}{ui.RS}")
        else:
            try:
                user = input(f"  {ui.BD}{ui.AC}you{ui.RS} {ui.GR}›{ui.RS} ")
            except (EOFError, KeyboardInterrupt):
                sys.exit(0)

        stripped = user.strip()
        if not stripped:
            sys.stdout.write("\033[A\033[2K")
            sys.stdout.flush()
            continue

        cmd = stripped.lower()

        # ── Commands ─────────────────────────────────────────────
        if cmd in {"q", "quit", "exit"}:
            sys.exit(0)

        if cmd in {"r", "reset"}:
            conversation.clear()
            cache = make_prompt_cache(model, MAX_KV)
            try:
                mx.clear_cache()
            except AttributeError:
                try:
                    mx.metal.clear_cache()
                except Exception:
                    pass
            print(f"  {ui.GR}Context reset.{ui.RS}\n")
            continue

        if cmd == "s":
            _settings_menu()
            _save_cfg()
            sys.stdout.write("\033[3J\033[2J\033[H")
            sys.stdout.flush()
            _draw_chat_header(model_key, hw)
            continue

        if cmd in {"rensa", "clear", "cls"}:
            sys.stdout.write("\033[3J\033[2J\033[H")
            sys.stdout.flush()
            _draw_chat_header(model_key, hw)
            continue

        if cmd == "h":
            W = min(cols - 4, 54)
            print(f"  {ui.GR}{'─' * W}{ui.RS}")
            print(
                f"  {ui.AC}q{ui.RS} quit   "
                f"{ui.AC}r{ui.RS} reset   "
                f"{ui.AC}s{ui.RS} settings   "
                f"{ui.AC}rensa{ui.RS} clear   "
                f"{ui.AC}h{ui.RS} help   "
                f"{ui.AC}v{ui.RS} voice"
            )
            print(f"  {ui.GR}{'─' * W}{ui.RS}\n")
            continue

        if cmd == "v":
            if voice_api is None:
                from voice import check_available, install_instructions, push_to_talk
                voice_api = {
                    "check_available":    check_available,
                    "install_instructions": install_instructions,
                    "push_to_talk":       push_to_talk,
                }
            if not voice_api["check_available"]():
                print(f"  {ui.YL}⚠  Voice requires: {voice_api['install_instructions']()}{ui.RS}\n")
            else:
                VOICE_MODE = not VOICE_MODE
                state = f"{ui.AC}on{ui.RS}" if VOICE_MODE else f"{ui.GR}off{ui.RS}"
                print(f"  {ui.GR}Voice: {state}{ui.RS}\n")
            continue

        # ── Plugin commands ─────────────────────────────────────
        plugin_ctx = {"model_key": model_key, "hw": hw, "temp": TEMP}
        if plugins.run_on_command(stripped, plugin_ctx):
            continue

        # ── Generate ─────────────────────────────────────────────
        MSGS += 1
        _sep(cols)

        # Plugin: pre-process user input
        stripped = plugins.run_on_query(stripped, plugin_ctx)

        # Append to conversation history and trim if too long
        conversation.append({"role": "user", "content": stripped})
        _trim_conversation(conversation)

        # Build messages: system prompt + conversation history
        # Some models (Mistral) don't support the "system" role —
        # for those, merge system prompt into the first user message.
        sys_prompt = _active_prompt()
        pstyle = entry.get("prompt_style", "")

        if pstyle == "mistral":
            # Mistral requires strict user/assistant alternation, no system role
            messages = list(conversation)
            # Inject system prompt into first user message
            if messages and messages[0]["role"] == "user":
                messages[0] = {
                    "role": "user",
                    "content": sys_prompt + "\n\n" + messages[0]["content"],
                }
        else:
            messages = [{"role": "system", "content": sys_prompt}] + list(conversation)

        # Plugin: modify messages before generation
        messages = plugins.run_on_generate_messages(messages, plugin_ctx)

        try:
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        full        = ""
        buf         = ""
        last_flush  = time.time()
        gen_start   = time.time()
        token_count = 0

        live_sampler  = make_sampler(temp=TEMP)
        highlighter   = ui.StreamHighlighter()
        # If the chat template pre-fills <think>, the model streams reasoning
        # directly (no opening tag), ending with </think>.  Start the filter
        # already inside the think block in that case.
        _prefill_think = entry.get("think_prefill", False)
        think_filter  = ui.ThinkFilter(start_in_think=_prefill_think)

        sys.stdout.write(f"\n  {ui.BD}{ui.AC2}{entry['label']}{ui.RS} {ui.GR}›{ui.RS} ")
        sys.stdout.flush()

        try:
            for tok in stream_generate(
                model, tokenizer, prompt_str,
                max_tokens=MAX_TOK,
                sampler=live_sampler,
                prompt_cache=cache,
            ):
                piece        = tok.text
                full        += piece
                buf         += think_filter.feed(piece)
                token_count += 1
                now          = time.time()
                flush = (
                    "\n" in buf
                    or (len(buf) >= 24 and buf[-1] in " \t.,!?;:")
                    or (len(buf) >= 8 and now - last_flush >= 0.06)
                )
                if flush:
                    out = highlighter.feed(buf)
                    if out:
                        sys.stdout.write(out)
                        sys.stdout.flush()
                    buf        = ""
                    last_flush = now

            if buf:
                out = highlighter.feed(buf)
                if out:
                    sys.stdout.write(out)
                    sys.stdout.flush()
            # Flush any buffered non-think content from the filter
            leftover = think_filter.flush()
            if leftover:
                out = highlighter.feed(leftover)
                if out:
                    sys.stdout.write(out)
                    sys.stdout.flush()
            trailing = highlighter.flush()
            if trailing:
                sys.stdout.write(trailing)
                sys.stdout.flush()

        except RuntimeError as e:
            if "OutOfMemory" in str(e) or "Insufficient Memory" in str(e):
                print(f"\n\n  {ui.RD}⚠ Out of memory — freeing cache …{ui.RS}")
                try:
                    mx.clear_cache()
                except AttributeError:
                    try:
                        mx.metal.clear_cache()
                    except Exception:
                        pass
                # Remove the failed user message from history
                if conversation and conversation[-1]["role"] == "user":
                    conversation.pop()
                cache = make_prompt_cache(model, MAX_KV)
                print(f"  {ui.GR}Done. Try again.{ui.RS}\n")
                _sep(cols)
                print()
                continue
            raise

        lat_ms = int((time.time() - gen_start) * 1000)
        tps    = token_count / max(0.001, time.time() - gen_start)
        try:
            gpu_gb = mx.get_active_memory() / (1024 ** 3)
        except Exception:
            gpu_gb = 0.0

        print()

        # Save assistant response to conversation history (strip think blocks)
        clean = re.sub(r"<think>.*?</think>", "", full, flags=re.DOTALL).strip()
        conversation.append({"role": "assistant", "content": clean or full})

        # Plugin: post-process response + store for clipboard
        plugins.set_last_response(clean or full)
        plugins.run_on_response(full, stripped, plugin_ctx)

        log_interaction(query=stripped, answer=full, metadata={"tps": round(tps, 1)})
        _sep(cols)
        ui.print_stats_line(token_count, lat_ms, tps, TEMP, gpu_gb, SHOW_STATS)
        print()


if __name__ == "__main__":
    main()
