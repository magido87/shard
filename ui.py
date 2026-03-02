"""ANSI/color helpers, themes, and layout for localai."""

import shutil

# ── Theme definitions ──────────────────────────────────────────────
THEMES: dict = {
    "ocean": {
        "label": "Ocean",
        "AC":  "\033[38;5;45m",   # bright aqua
        "AC2": "\033[38;5;135m",  # violet/purple
        "DB":  "\033[38;5;18m",   # deep blue (dim)
        "BD":  "\033[1m",
        "GR":  "\033[90m",        # muted/gray
        "CY":  "\033[38;5;51m",   # cyan
        "RS":  "\033[0m",
        "RD":  "\033[38;5;196m",
        "YL":  "\033[38;5;220m",
        "MG":  "\033[38;5;201m",
    },
    "dusk": {
        "label": "Dusk",
        "AC":  "\033[38;5;141m",  # lavender
        "AC2": "\033[38;5;240m",  # gray
        "DB":  "\033[38;5;236m",  # dark gray
        "BD":  "\033[1m",
        "GR":  "\033[90m",
        "CY":  "\033[38;5;147m",  # soft purple
        "RS":  "\033[0m",
        "RD":  "\033[38;5;196m",
        "YL":  "\033[38;5;222m",
        "MG":  "\033[38;5;177m",
    },
    "mono": {
        "label": "Mono",
        "AC":  "\033[97m",        # bright white
        "AC2": "\033[37m",        # white
        "DB":  "\033[90m",        # dark gray
        "BD":  "\033[1m",
        "GR":  "\033[90m",
        "CY":  "\033[97m",
        "RS":  "\033[0m",
        "RD":  "\033[91m",
        "YL":  "\033[93m",
        "MG":  "\033[95m",
    },
    "forest": {
        "label": "Forest",
        "AC":  "\033[38;5;82m",   # bright green
        "AC2": "\033[38;5;64m",   # dark green
        "DB":  "\033[38;5;22m",   # deep green
        "BD":  "\033[1m",
        "GR":  "\033[90m",
        "CY":  "\033[38;5;119m",  # light green
        "RS":  "\033[0m",
        "RD":  "\033[38;5;196m",
        "YL":  "\033[38;5;227m",
        "MG":  "\033[38;5;120m",
    },
}

_current_theme: str = "ocean"

# Module-level color shortcuts (updated by apply_theme)
AC = AC2 = DB = BD = GR = CY = RS = RD = YL = MG = ""


def apply_theme(name: str) -> None:
    global _current_theme, AC, AC2, DB, BD, GR, CY, RS, RD, YL, MG
    t = THEMES.get(name, THEMES["ocean"])
    _current_theme = name if name in THEMES else "ocean"
    AC  = t["AC"]
    AC2 = t["AC2"]
    DB  = t["DB"]
    BD  = t["BD"]
    GR  = t["GR"]
    CY  = t["CY"]
    RS  = t["RS"]
    RD  = t["RD"]
    YL  = t["YL"]
    MG  = t["MG"]


def current_theme() -> str:
    return _current_theme


def c(code: int) -> str:
    """Return ANSI 256-color foreground escape for the given color code."""
    return f"\033[38;5;{code}m"


def width() -> int:
    """Return current terminal width."""
    return shutil.get_terminal_size().columns


def trunc(s: str, n: int) -> str:
    """Truncate string to n visible chars, appending … if needed."""
    if len(s) <= n:
        return s
    return s[:n - 1] + "…"


def hline(char: str = "─", margin: int = 2) -> str:
    """Return a horizontal rule padded by margin on each side."""
    w = max(0, width() - margin * 2)
    return " " * margin + char * w


def print_header(
    model_key: str,
    chip: str,
    ram: int,
    ui_mode: str,
    stats: str,
    privacy: bool,
) -> None:
    """Print the top status bar (skipped in zen mode)."""
    if ui_mode == "zen":
        return
    pvt = f"  {RD}[pvt]{RS}" if privacy else ""
    line = (
        f"  {GR}{chip} · {ram}GB · offline{pvt}{RS}"
    )
    print(line)


def print_help_bar(voice_available: bool = False) -> None:
    """Print the one-line command reference bar."""
    voice_hint = f"  {AC}v{RS} voice" if voice_available else ""
    print(
        f"  {AC}q{RS} quit   "
        f"{AC}r{RS} reset   "
        f"{AC}s{RS} settings   "
        f"{AC}rensa{RS} clear   "
        f"{AC}h{RS} help"
        f"{voice_hint}"
    )


def print_stats_line(
    tokens: int,
    latency_ms: int,
    tps: float,
    temp: float,
    gpu_gb: float,
    mode: str,
) -> None:
    """Print per-response stats line. Skips output if mode is 'off'."""
    if mode == "off":
        return
    if mode == "full":
        print(
            f"  {GR}Tokens: {tokens} · "
            f"Latency: {latency_ms}ms · "
            f"t/s: {tps:.1f} · "
            f"Temp: {temp} · "
            f"GPU: {gpu_gb:.1f}GB{RS}"
        )
    else:  # compact
        print(f"  {GR}{tps:.0f} t/s · {gpu_gb:.1f}GB GPU{RS}")


# Apply the default theme on import
apply_theme("ocean")


class StreamHighlighter:
    """Detects fenced code blocks in streaming output and syntax-highlights them.

    Usage:
        h = StreamHighlighter()
        out = h.feed(chunk)     # returns text to print ("" while buffering code)
        trailing = h.flush()    # call after loop ends — handles unclosed fences
    """

    def __init__(self) -> None:
        self._in_fence: bool = False
        self._lang: str = ""
        self._code_lines: list = []
        self._pending: str = ""  # partial line not yet terminated by \n

    def feed(self, text: str) -> str:
        output = []
        for ch in text:
            if ch == "\n":
                line = self._pending + "\n"
                self._pending = ""
                output.append(self._process_line(line))
            else:
                self._pending += ch
        return "".join(output)

    def flush(self) -> str:
        output = []
        if self._pending:
            line = self._pending
            self._pending = ""
            output.append(self._process_line(line))
        if self._in_fence:
            output.append(self._render_code_block())
            self._in_fence = False
            self._lang = ""
            self._code_lines = []
        return "".join(output)

    def _process_line(self, line: str) -> str:
        stripped = line.rstrip("\n")
        if not self._in_fence:
            if stripped.startswith("```"):
                self._in_fence = True
                self._lang = stripped[3:].strip()
                self._code_lines = []
                return ""  # suppress opening fence
            return line
        else:
            if stripped == "```":
                rendered = self._render_code_block()
                self._in_fence = False
                self._lang = ""
                self._code_lines = []
                return rendered
            self._code_lines.append(line)
            return ""  # suppress while buffering

    def _render_code_block(self) -> str:
        gr = GR; rs = RS  # read at call time so current theme is respected
        code = "".join(self._code_lines)
        lang_name  = self._lang or "code"
        lang_label = f"  {gr}▸ {lang_name}{rs}\n"
        sep        = f"  {gr}{'─' * min(width() - 4, 54)}{rs}\n"

        try:
            from pygments import highlight
            from pygments.lexers import get_lexer_by_name, guess_lexer, TextLexer
            from pygments.formatters import Terminal256Formatter
            from pygments.util import ClassNotFound

            if self._lang:
                try:
                    lexer = get_lexer_by_name(self._lang, stripall=True)
                except ClassNotFound:
                    try:    lexer = guess_lexer(code)
                    except ClassNotFound: lexer = TextLexer()
            else:
                try:    lexer = guess_lexer(code)
                except ClassNotFound: lexer = TextLexer()

            highlighted = highlight(code, lexer, Terminal256Formatter(style="monokai"))
            body = "\n".join("  " + ln for ln in highlighted.splitlines()) + "\n"
        except ImportError:
            body = "\n".join(f"  {gr}{ln}{rs}" for ln in code.splitlines()) + "\n"

        return lang_label + body + sep


class ThinkFilter:
    """Strips <think>...</think> reasoning blocks from streaming LLM output.

    Usage:
        f = ThinkFilter()
        out = f.feed(chunk)   # returns displayable text (think blocks removed)
        tail = f.flush()      # call after loop — returns any buffered non-think text
    """

    OPEN  = "<think>"
    CLOSE = "</think>"

    def __init__(self, start_in_think: bool = False) -> None:
        self._in_think: bool = start_in_think
        self._buf: str = ""   # lookahead buffer for partial tag detection
        self._strip_nl: bool = False  # strip leading newlines on next feed

    def feed(self, chunk: str) -> str:
        out: list[str] = []
        self._buf += chunk
        # Strip leading newlines left over from a just-closed think block
        if self._strip_nl:
            self._buf = self._buf.lstrip("\n")
            self._strip_nl = False
        while self._buf:
            if not self._in_think:
                idx = self._buf.find(self.OPEN)
                if idx == -1:
                    # No opening tag — safe to output all but the last 6 chars
                    # (could be start of "<think>")
                    keep = len(self.OPEN) - 1
                    safe = len(self._buf) - keep
                    if safe > 0:
                        out.append(self._buf[:safe])
                        self._buf = self._buf[safe:]
                    break
                else:
                    out.append(self._buf[:idx])
                    self._buf = self._buf[idx + len(self.OPEN):]
                    self._in_think = True
            else:
                idx = self._buf.find(self.CLOSE)
                if idx == -1:
                    # Discard everything except possible partial closing tag
                    keep = len(self.CLOSE) - 1
                    self._buf = self._buf[-keep:] if len(self._buf) > keep else self._buf
                    break
                else:
                    self._buf = self._buf[idx + len(self.CLOSE):]
                    self._in_think = False
                    # Strip newlines immediately after </think>; if the buffer
                    # is now empty, carry the strip over to the next feed()
                    self._buf = self._buf.lstrip("\n")
                    if not self._buf:
                        self._strip_nl = True
        return "".join(out)

    def flush(self) -> str:
        """Return any buffered non-think text and reset state."""
        out = "" if self._in_think else self._buf
        self._buf = ""
        self._in_think = False
        self._strip_nl = False
        return out
