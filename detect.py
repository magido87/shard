"""Hardware detection for localai — Apple Silicon."""

import os
import platform
import subprocess


def chip_name() -> str:
    """Return human-readable chip name, e.g. 'M3 Pro'. Falls back to 'Apple Silicon'."""
    try:
        raw = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        if raw:
            return raw
    except Exception:
        pass
    try:
        model = subprocess.check_output(
            ["sysctl", "-n", "hw.model"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        if model:
            return model
    except Exception:
        pass
    return "Apple Silicon"


def ram_gb() -> int:
    """Return total RAM in GB (integer)."""
    try:
        import psutil
        return psutil.virtual_memory().total // (1024 ** 3)
    except Exception:
        pass
    try:
        raw = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        return int(raw) // (1024 ** 3)
    except Exception:
        return 8


def os_version() -> str:
    """Return macOS version string, e.g. '15.3'."""
    return platform.mac_ver()[0] or "unknown"


def _pressure(ram_pct: float, swap_bytes: int) -> str:
    """Return memory pressure level: 'low', 'medium', or 'high'."""
    swap_gb = swap_bytes / (1024 ** 3)
    if ram_pct > 85 or swap_gb > 3:   return "high"
    if ram_pct > 65 or swap_gb > 0.5: return "medium"
    return "low"


def hardware_summary() -> dict:
    """Return dict with chip, ram, os, and extended health keys."""
    import psutil
    import shutil
    vm   = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = shutil.disk_usage(os.path.expanduser("~"))

    return {
        # existing
        "chip":          chip_name(),
        "ram":           int(vm.total // (1024 ** 3)),
        "os":            os_version(),
        # new
        "ram_available": round(vm.available / (1024 ** 3), 1),
        "ram_used":      round(vm.used      / (1024 ** 3), 1),
        "ram_pct":       vm.percent,
        "swap_used":     round(swap.used    / (1024 ** 3), 1),
        "swap_total":    round(swap.total   / (1024 ** 3), 1),
        "disk_free":     round(disk.free    / (1024 ** 3), 1),
        "pressure":      _pressure(vm.percent, swap.used),
    }
