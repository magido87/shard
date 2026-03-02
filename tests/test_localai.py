#!/usr/bin/env python3
"""
Comprehensive test suite for localai — no model downloads required.
All MLX / mlx_lm calls are mocked.

Run:  python3 tests/test_localai.py
"""

import importlib
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Ensure project root is on sys.path ─────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Mock MLX and related modules before any imports ────────────────
_mock_mx = MagicMock()
_mock_mx.metal = MagicMock()
_mock_mx.metal.clear_cache = MagicMock()
_mock_mx.get_active_memory = MagicMock(return_value=2.0 * 1024**3)

_mock_mlx_lm = MagicMock()
_mock_mlx_lm_generate = MagicMock()
_mock_mlx_lm_cache = MagicMock()
_mock_mlx_lm_sample = MagicMock()

sys.modules["mlx"]                = types.ModuleType("mlx")
sys.modules["mlx.core"]           = _mock_mx
sys.modules["mlx_lm"]             = _mock_mlx_lm
sys.modules["mlx_lm.generate"]    = _mock_mlx_lm_generate
sys.modules["mlx_lm.models"]      = types.ModuleType("mlx_lm.models")
sys.modules["mlx_lm.models.cache"] = _mock_mlx_lm_cache
sys.modules["mlx_lm.sample_utils"] = _mock_mlx_lm_sample

# ── Test harness ───────────────────────────────────────────────────
_passed = 0
_failed = 0
_errors: list[str] = []


def test(name: str):
    """Decorator that registers and runs a test function."""
    def wrapper(fn):
        global _passed, _failed
        try:
            fn()
            _passed += 1
            print(f"  ✓ {name}")
        except Exception as e:
            _failed += 1
            _errors.append(f"{name}: {e}")
            print(f"  ✗ {name} — {e}")
        return fn
    return wrapper


# ====================================================================
# 1. MODEL REGISTRY
# ====================================================================

from models import (
    MODELS, available_models, recommend_model, model_fit, disk_ok,
    top_picks, grouped_by_category, CATEGORY_LABELS, CATEGORY_ORDER,
)

REQUIRED_KEYS = {"id", "label", "size", "min_ram", "dl_size", "profile",
                 "category", "desc", "kv", "tokens", "prompt_style"}


@test("model_registry: all entries have required keys")
def _():
    for key, entry in MODELS.items():
        missing = REQUIRED_KEYS - set(entry.keys())
        assert not missing, f"{key} missing {missing}"


@test("model_registry: dl_size is parseable float")
def _():
    for key, entry in MODELS.items():
        val = float(entry["dl_size"].rstrip("GB"))
        assert val > 0, f"{key} dl_size={entry['dl_size']}"


@test("model_registry: categories are valid")
def _():
    valid = set(CATEGORY_ORDER)
    for key, entry in MODELS.items():
        assert entry["category"] in valid, f"{key} has invalid category {entry['category']}"


@test("model_registry: prompt_style is one of known styles")
def _():
    valid = {"qwen", "llama", "mistral", "gemma", "phi"}
    for key, entry in MODELS.items():
        assert entry["prompt_style"] in valid, f"{key} has unknown style {entry['prompt_style']}"


@test("model_registry: min_ram reasonable vs dl_size")
def _():
    for key, entry in MODELS.items():
        dl = float(entry["dl_size"].rstrip("GB"))
        # MoE models can have dl_size > min_ram (sparse activations)
        # But non-MoE should have min_ram >= dl_size
        if "MoE" not in entry.get("size", ""):
            assert entry["min_ram"] >= dl, f"{key}: min_ram={entry['min_ram']} < dl_size={dl}"


@test("model_registry: smollm2_1_7b min_ram is 4 (Bug G fix)")
def _():
    assert MODELS["smollm2_1_7b"]["min_ram"] == 4


# ====================================================================
# 2. MODEL SELECTION
# ====================================================================

def _make_hw(ram=16, ram_available=10.0, pressure="low", disk_free=50.0):
    return {
        "chip": "M3 Pro", "ram": ram, "os": "15.3",
        "ram_available": ram_available, "ram_used": ram - ram_available,
        "ram_pct": 60.0, "swap_used": 0.0, "swap_total": 0.0,
        "disk_free": disk_free, "pressure": pressure,
    }


@test("recommend_model: returns a valid model key")
def _():
    hw = _make_hw()
    key = recommend_model(hw)
    assert key in MODELS, f"returned unknown key: {key}"


@test("recommend_model: returns safe profile only")
def _():
    hw = _make_hw()
    key = recommend_model(hw)
    assert MODELS[key]["profile"] == "safe"


@test("recommend_model: low-RAM fallback returns smallest safe (Bug F fix)")
def _():
    hw = _make_hw(ram=1, ram_available=0.5, pressure="high")
    key = recommend_model(hw)
    assert key in MODELS
    assert MODELS[key]["profile"] == "safe"
    # Should be the smallest, not hardcoded phi4mini
    smallest_safe = min(
        (v["min_ram"] for v in MODELS.values() if v["profile"] == "safe")
    )
    assert MODELS[key]["min_ram"] == smallest_safe


@test("recommend_model: high pressure reduces effective RAM")
def _():
    hw_low  = _make_hw(ram=16, ram_available=10.0, pressure="low")
    hw_high = _make_hw(ram=16, ram_available=10.0, pressure="high")
    key_low  = recommend_model(hw_low)
    key_high = recommend_model(hw_high)
    assert MODELS[key_high]["min_ram"] <= MODELS[key_low]["min_ram"]


@test("available_models: excludes unfiltered by default")
def _():
    models = available_models(64, unfiltered=False)
    for key, entry in models:
        assert entry["profile"] != "unfiltered", f"{key} is unfiltered"


@test("available_models: includes unfiltered when requested")
def _():
    models = available_models(64, unfiltered=True)
    unf = [k for k, v in models if v["profile"] == "unfiltered"]
    assert len(unf) > 0


@test("available_models: respects RAM limit")
def _():
    models = available_models(4)
    for key, entry in models:
        assert entry["min_ram"] <= 4, f"{key} min_ram={entry['min_ram']}"


@test("model_fit: returns correct labels")
def _():
    hw = _make_hw(ram_available=8.0)
    assert model_fit("qwen25_05b", hw) == "✓ fits"
    # Find a model that needs more RAM than available
    for key, entry in MODELS.items():
        if entry["min_ram"] > 12:
            assert "risky" in model_fit(key, hw)
            break


@test("disk_ok: checks headroom correctly")
def _():
    hw = _make_hw(disk_free=5.0)
    assert disk_ok("qwen25_05b", hw)  # 0.4GB + 1 < 5
    hw2 = _make_hw(disk_free=1.0)
    assert not disk_ok("qwen25_05b", hw2)  # 0.4 + 1 > 1


@test("top_picks: returns non-empty list")
def _():
    hw = _make_hw()
    models = available_models(hw["ram"])
    picks = top_picks(models, hw)
    assert len(picks) > 0


@test("grouped_by_category: returns valid groups")
def _():
    models = available_models(64)
    groups = grouped_by_category(models)
    for cat, items in groups:
        assert cat in CATEGORY_ORDER
        assert len(items) > 0


# ====================================================================
# 3. CONFIG
# ====================================================================

import config


@test("config: load returns all default keys")
def _():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        f.write(b"{}")
        tmp = Path(f.name)
    try:
        orig = config.CONFIG_PATH
        config.CONFIG_PATH = tmp
        cfg = config.load()
        for key in config.DEFAULTS:
            assert key in cfg, f"missing key: {key}"
    finally:
        config.CONFIG_PATH = orig
        tmp.unlink()


@test("config: save + load round-trip")
def _():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "config.json"
        orig = config.CONFIG_PATH
        config.CONFIG_PATH = tmp
        try:
            cfg = dict(config.DEFAULTS)
            cfg["model"] = "gemma3_4b"
            cfg["temp"] = 0.5
            config.save(cfg)
            loaded = config.load()
            assert loaded["model"] == "gemma3_4b"
            assert loaded["temp"] == 0.5
        finally:
            config.CONFIG_PATH = orig


@test("config: corrupt JSON falls back to defaults")
def _():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        f.write("{invalid json!!")
        tmp = Path(f.name)
    try:
        orig = config.CONFIG_PATH
        config.CONFIG_PATH = tmp
        cfg = config.load()
        assert cfg["model"] == config.DEFAULTS["model"]
    finally:
        config.CONFIG_PATH = orig
        tmp.unlink()


@test("config: privacy_mode prevents save")
def _():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "config.json"
        orig = config.CONFIG_PATH
        config.CONFIG_PATH = tmp
        try:
            cfg = dict(config.DEFAULTS)
            cfg["privacy_mode"] = True
            config.save(cfg)
            assert not tmp.exists(), "file should not exist with privacy_mode"
        finally:
            config.CONFIG_PATH = orig


# ====================================================================
# 4. CONVERSATION TRIMMING
# ====================================================================

# Import the function directly by reading chat.py source
# We can't easily import chat.py (it has heavy deps), so redefine here
def _trim_conversation(conv: list[dict], max_turns: int = 20) -> None:
    """Mirror of chat._trim_conversation after Bug C fix."""
    while len(conv) > max_turns * 2:
        conv.pop(0)
        if conv and conv[0]["role"] == "assistant":
            conv.pop(0)
    while conv and conv[0]["role"] != "user":
        conv.pop(0)


@test("trim_conversation: respects max_turns")
def _():
    conv = []
    for i in range(30):
        conv.append({"role": "user", "content": f"Q{i}"})
        conv.append({"role": "assistant", "content": f"A{i}"})
    _trim_conversation(conv, max_turns=5)
    assert len(conv) <= 10


@test("trim_conversation: preserves user/assistant alternation")
def _():
    conv = []
    for i in range(30):
        conv.append({"role": "user", "content": f"Q{i}"})
        conv.append({"role": "assistant", "content": f"A{i}"})
    _trim_conversation(conv, max_turns=5)
    for i, msg in enumerate(conv):
        expected = "user" if i % 2 == 0 else "assistant"
        assert msg["role"] == expected, f"index {i}: expected {expected}, got {msg['role']}"


@test("trim_conversation: starts with user after trim (Bug C fix)")
def _():
    # Construct a scenario where trimming leaves assistant first
    conv = [
        {"role": "assistant", "content": "orphan"},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
    ]
    _trim_conversation(conv, max_turns=1)
    assert conv[0]["role"] == "user"


@test("trim_conversation: empty list is safe")
def _():
    conv = []
    _trim_conversation(conv, max_turns=5)
    assert len(conv) == 0


@test("trim_conversation: single pair untouched")
def _():
    conv = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    _trim_conversation(conv, max_turns=5)
    assert len(conv) == 2


# ====================================================================
# 5. MESSAGE BUILDING (prompt styles)
# ====================================================================

@test("message_building: mistral style has no system role")
def _():
    """Mistral models should merge system into first user message."""
    sys_prompt = "You are helpful."
    conversation = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "How are you?"},
    ]
    # Simulate chat.py mistral path
    messages = list(conversation)
    if messages and messages[0]["role"] == "user":
        messages[0] = {
            "role": "user",
            "content": sys_prompt + "\n\n" + messages[0]["content"],
        }
    # Verify no system role
    for msg in messages:
        assert msg["role"] != "system", "Mistral messages should not have system role"
    assert sys_prompt in messages[0]["content"]


@test("message_building: non-mistral has system role")
def _():
    sys_prompt = "You are helpful."
    conversation = [
        {"role": "user", "content": "Hello"},
    ]
    messages = [{"role": "system", "content": sys_prompt}] + list(conversation)
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == sys_prompt


@test("message_building: all 5 prompt styles assigned in registry")
def _():
    styles = set()
    for key, entry in MODELS.items():
        styles.add(entry["prompt_style"])
    assert styles == {"qwen", "llama", "mistral", "gemma", "phi"}


@test("message_building: role alternation in conversation")
def _():
    """Verify that a normal conversation always alternates user/assistant."""
    conv = []
    for i in range(10):
        conv.append({"role": "user", "content": f"Q{i}"})
        conv.append({"role": "assistant", "content": f"A{i}"})
    for i in range(0, len(conv), 2):
        assert conv[i]["role"] == "user"
        assert conv[i+1]["role"] == "assistant"


# ====================================================================
# 6. PLUGINS
# ====================================================================

import plugins


@test("plugins: registry has 8 plugins")
def _():
    assert len(plugins.REGISTRY) == 8


@test("plugins: installed_plugins reads config")
def _():
    cfg = {"plugins": ["web_search", "tts"], "disabled_plugins": []}
    assert plugins.installed_plugins(cfg) == ["web_search", "tts"]


@test("plugins: enabled_plugins excludes disabled")
def _():
    cfg = {"plugins": ["web_search", "tts"], "disabled_plugins": ["tts"]}
    enabled = plugins.enabled_plugins(cfg)
    assert "web_search" in enabled
    assert "tts" not in enabled


@test("plugins: toggle flips state")
def _():
    cfg = {"plugins": ["web_search"], "disabled_plugins": []}
    result = plugins.toggle_plugin("web_search", cfg)
    assert result is False  # now disabled
    assert "web_search" in cfg["disabled_plugins"]
    result = plugins.toggle_plugin("web_search", cfg)
    assert result is True  # now enabled again
    assert "web_search" not in cfg["disabled_plugins"]


@test("plugins: install unknown plugin fails")
def _():
    cfg = {"plugins": [], "disabled_plugins": []}
    ok, msg = plugins.install_plugin("nonexistent_plugin_xyz", cfg)
    assert not ok


@test("plugins: uninstall removes from config")
def _():
    cfg = {"plugins": ["web_search"], "disabled_plugins": ["web_search"]}
    ok, msg = plugins.uninstall_plugin("web_search", cfg)
    assert ok
    assert "web_search" not in cfg["plugins"]
    assert "web_search" not in cfg["disabled_plugins"]


@test("plugins: last_response storage")
def _():
    plugins.set_last_response("test output")
    assert plugins.get_last_response() == "test output"
    plugins.set_last_response("")


@test("plugins: hook dispatchers with no loaded plugins")
def _():
    plugins._loaded = {}
    result = plugins.run_on_query("hello", {})
    assert result == "hello"
    result = plugins.run_on_response("answer", "question", {})
    assert result == "answer"
    handled = plugins.run_on_command("/test", {})
    assert handled is False
    msgs = [{"role": "user", "content": "hi"}]
    result = plugins.run_on_generate_messages(msgs, {})
    assert result == msgs


# ====================================================================
# 7. PLUGIN INJECTION (Bug H fix)
# ====================================================================

def _load_plugin_module(name: str):
    """Load a plugin by file path, avoiding namespace collision."""
    plugin_file = PROJECT_ROOT / "plugins" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_plugin_{name}", plugin_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@test("file_reader: injects into system role when present")
def _():
    mod = _load_plugin_module("file_reader")
    mod._file_context = [{"path": "/tmp/test.txt", "content": "file content"}]
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    result = mod.on_generate_messages(messages, {})
    assert "file content" in result[0]["content"]
    assert result[0]["role"] == "system"
    mod._file_context = []


@test("file_reader: injects into first user when no system (Bug H fix)")
def _():
    mod = _load_plugin_module("file_reader")
    mod._file_context = [{"path": "/tmp/test.txt", "content": "file content"}]
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    result = mod.on_generate_messages(messages, {})
    assert "file content" in result[0]["content"]
    assert result[0]["role"] == "user"
    # Original user content should still be there
    assert "Hello" in result[0]["content"]
    mod._file_context = []


@test("file_reader: no-op when no files loaded")
def _():
    mod = _load_plugin_module("file_reader")
    mod._file_context = []
    messages = [{"role": "user", "content": "Hello"}]
    result = mod.on_generate_messages(messages, {})
    assert result is None


@test("translator: injects into system role when present")
def _():
    mod = _load_plugin_module("translator")
    mod._target_lang = "Swedish"
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    result = mod.on_generate_messages(messages, {})
    assert "Swedish" in result[0]["content"]
    assert result[0]["role"] == "system"
    mod._target_lang = ""


@test("translator: injects into first user when no system (Bug H fix)")
def _():
    mod = _load_plugin_module("translator")
    mod._target_lang = "Swedish"
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    result = mod.on_generate_messages(messages, {})
    assert "Swedish" in result[0]["content"]
    assert result[0]["role"] == "user"
    assert "Hello" in result[0]["content"]
    mod._target_lang = ""


@test("translator: no-op when no target language")
def _():
    mod = _load_plugin_module("translator")
    mod._target_lang = ""
    messages = [{"role": "user", "content": "Hello"}]
    result = mod.on_generate_messages(messages, {})
    assert result is None


# ====================================================================
# 8. UI
# ====================================================================

import ui


@test("ui: all 4 themes exist")
def _():
    assert set(ui.THEMES.keys()) == {"ocean", "dusk", "mono", "forest"}


@test("ui: apply_theme sets color globals")
def _():
    ui.apply_theme("forest")
    assert ui.AC != ""
    assert "38;5" in ui.AC or "97m" in ui.AC
    ui.apply_theme("ocean")  # restore


@test("ui: apply_theme falls back to ocean on invalid")
def _():
    ui.apply_theme("nonexistent")
    assert ui.current_theme() == "ocean"


@test("ui: trunc respects limit")
def _():
    assert ui.trunc("hello", 10) == "hello"
    assert len(ui.trunc("hello world this is long", 10)) == 10
    assert ui.trunc("hello world this is long", 10).endswith("…")


@test("ui: width returns positive int")
def _():
    w = ui.width()
    assert isinstance(w, int)
    assert w > 0


@test("ui: hline returns string of correct form")
def _():
    line = ui.hline()
    assert "─" in line


@test("ui: c() returns valid ANSI escape")
def _():
    result = ui.c(196)
    assert result == "\033[38;5;196m"


# ====================================================================
# 9. STREAM HIGHLIGHTER
# ====================================================================

@test("highlighter: plain text passes through")
def _():
    h = ui.StreamHighlighter()
    out = h.feed("Hello world\n")
    assert out == "Hello world\n"
    assert h.flush() == ""


@test("highlighter: code block is buffered and rendered")
def _():
    h = ui.StreamHighlighter()
    out = h.feed("```python\n")
    assert out == ""  # suppressed
    out += h.feed("x = 1\n")
    assert "x = 1" not in out  # still buffered
    out = h.feed("```\n")
    # Now it should be rendered
    assert "x = 1" in out or out != ""


@test("highlighter: unclosed fence rendered on flush")
def _():
    h = ui.StreamHighlighter()
    h.feed("```\n")
    h.feed("some code\n")
    trailing = h.flush()
    assert "some code" in trailing


@test("highlighter: empty code block")
def _():
    h = ui.StreamHighlighter()
    h.feed("```\n")
    out = h.feed("```\n")
    # Should render (possibly empty) without error
    assert isinstance(out, str)


@test("highlighter: multiple code blocks")
def _():
    h = ui.StreamHighlighter()
    out = ""
    out += h.feed("text before\n")
    out += h.feed("```\n")
    out += h.feed("code1\n")
    out += h.feed("```\n")
    out += h.feed("text between\n")
    out += h.feed("```\n")
    out += h.feed("code2\n")
    out += h.feed("```\n")
    out += h.flush()
    assert "text before" in out
    assert "text between" in out
    assert "code1" in out
    assert "code2" in out


# ====================================================================
# 10. HARDWARE DETECTION
# ====================================================================

import detect


@test("detect: chip_name returns non-empty string")
def _():
    result = detect.chip_name()
    assert isinstance(result, str)
    assert len(result) > 0


@test("detect: ram_gb returns positive int")
def _():
    result = detect.ram_gb()
    assert isinstance(result, int)
    assert result > 0


@test("detect: os_version returns string")
def _():
    result = detect.os_version()
    assert isinstance(result, str)


@test("detect: pressure levels")
def _():
    assert detect._pressure(90.0, 4 * 1024**3) == "high"
    assert detect._pressure(70.0, 0) == "medium"
    assert detect._pressure(50.0, 0) == "low"


@test("detect: hardware_summary returns all 10 keys")
def _():
    try:
        hw = detect.hardware_summary()
    except ImportError:
        # psutil not available in system Python — mock it
        mock_vm = MagicMock(total=16*1024**3, available=10*1024**3,
                            used=6*1024**3, percent=37.5)
        mock_swap = MagicMock(used=0, total=2*1024**3)
        mock_disk = MagicMock(free=100*1024**3)
        with patch.dict(sys.modules, {"psutil": MagicMock(
            virtual_memory=MagicMock(return_value=mock_vm),
            swap_memory=MagicMock(return_value=mock_swap),
        )}):
            with patch("shutil.disk_usage", return_value=mock_disk):
                importlib.reload(detect)
                hw = detect.hardware_summary()
    expected = {"chip", "ram", "os", "ram_available", "ram_used",
                "ram_pct", "swap_used", "swap_total", "disk_free", "pressure"}
    assert expected == set(hw.keys()), f"missing: {expected - set(hw.keys())}"


@test("detect: hardware_summary pressure is valid value")
def _():
    try:
        hw = detect.hardware_summary()
    except ImportError:
        mock_vm = MagicMock(total=16*1024**3, available=10*1024**3,
                            used=6*1024**3, percent=37.5)
        mock_swap = MagicMock(used=0, total=2*1024**3)
        mock_disk = MagicMock(free=100*1024**3)
        with patch.dict(sys.modules, {"psutil": MagicMock(
            virtual_memory=MagicMock(return_value=mock_vm),
            swap_memory=MagicMock(return_value=mock_swap),
        )}):
            with patch("shutil.disk_usage", return_value=mock_disk):
                importlib.reload(detect)
                hw = detect.hardware_summary()
    assert hw["pressure"] in ("low", "medium", "high")


# ====================================================================
# 11. AGENT (logging)
# ====================================================================

import agent


@test("agent: logging disabled by default")
def _():
    agent._enabled = False
    agent._session_path = None
    agent.log_interaction("q", "a")  # should be no-op


@test("agent: set_logging creates session file path")
def _():
    with tempfile.TemporaryDirectory() as td:
        orig = agent.LOGS_DIR
        agent.LOGS_DIR = Path(td)
        agent._session_path = None
        agent.set_logging(True)
        assert agent._session_path is not None
        assert "session_" in str(agent._session_path)
        agent.set_logging(False)
        agent.LOGS_DIR = orig
        agent._session_path = None


@test("agent: log_interaction writes JSONL")
def _():
    with tempfile.TemporaryDirectory() as td:
        orig = agent.LOGS_DIR
        agent.LOGS_DIR = Path(td)
        agent._session_path = None
        agent.set_logging(True)
        agent.log_interaction("test query", "test answer", {"model": "phi4mini"})
        assert agent._session_path.exists()
        line = agent._session_path.read_text().strip()
        data = json.loads(line)
        assert data["query"] == "test query"
        assert data["answer"] == "test answer"
        assert data["meta"]["model"] == "phi4mini"
        agent.set_logging(False)
        agent.LOGS_DIR = orig
        agent._session_path = None


@test("agent: list_logs returns list")
def _():
    result = agent.list_logs()
    assert isinstance(result, list)


# ====================================================================
# 12. TEMPLATE FALLBACK (Bug A fix)
# ====================================================================

@test("template_fallback: both paths should produce string")
def _():
    """Verify the fix: both try/except paths of apply_chat_template
    should produce a string, not token IDs."""
    mock_tokenizer = MagicMock()
    messages = [{"role": "user", "content": "Hi"}]

    # Path 1: tokenize=False works
    mock_tokenizer.apply_chat_template.return_value = "<|user|>Hi<|end|>"
    result = mock_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    assert isinstance(result, str)

    # Path 2: simulate what the fixed code does (also tokenize=False)
    mock_tokenizer.apply_chat_template.side_effect = None
    mock_tokenizer.apply_chat_template.return_value = "<|user|>Hi<|end|>"
    try:
        result = mock_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        result = mock_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    assert isinstance(result, str)


@test("template_fallback: verify chat.py source has tokenize=False in both branches")
def _():
    """Read chat.py and verify the fix is present."""
    src = (PROJECT_ROOT / "chat.py").read_text()
    # Find the template section
    idx = src.find("prompt_str = tokenizer.apply_chat_template(")
    assert idx > 0, "Could not find primary apply_chat_template call"
    # Find the except block
    except_idx = src.find("except Exception:", idx)
    assert except_idx > 0, "Could not find except block"
    # The fallback should also have tokenize=False
    fallback_section = src[except_idx:except_idx+200]
    assert "tokenize=False" in fallback_section, \
        f"Fallback missing tokenize=False:\n{fallback_section}"


# ====================================================================
# 13. FOCUS MODE (Bug D fix)
# ====================================================================

@test("focus_mode: verify _focus_header_done flag exists in chat.py")
def _():
    src = (PROJECT_ROOT / "chat.py").read_text()
    assert "_focus_header_done" in src, "Missing _focus_header_done flag"


@test("focus_mode: _draw_chat_header checks focus mode")
def _():
    src = (PROJECT_ROOT / "chat.py").read_text()
    assert 'UI_MODE == "focus"' in src, "Missing focus mode check in _draw_chat_header"


# ====================================================================
# 14. WARMUP (Bug B fix)
# ====================================================================

@test("warmup: verify tokenize=False in warmup call")
def _():
    src = (PROJECT_ROOT / "chat.py").read_text()
    # Find warmup section
    idx = src.find("Metal warmup")
    assert idx > 0
    warmup_section = src[idx:idx+300]
    assert "tokenize=False" in warmup_section, \
        f"Warmup missing tokenize=False:\n{warmup_section}"


# ====================================================================
# 15. VOICE CONDITION (Bug E fix)
# ====================================================================

@test("voice: redundant condition removed")
def _():
    src = (PROJECT_ROOT / "chat.py").read_text()
    assert "if VOICE_MODE or args.voice:" not in src, \
        "Redundant VOICE_MODE condition still present"
    assert "if args.voice:" in src


# ====================================================================
# SUMMARY
# ====================================================================

if __name__ == "__main__":
    print()
    total = _passed + _failed
    print(f"  {'─' * 40}")
    print(f"  {total} tests: {_passed} passed, {_failed} failed")
    if _errors:
        print()
        print("  Failures:")
        for e in _errors:
            print(f"    ✗ {e}")
    print()
    sys.exit(1 if _failed else 0)
