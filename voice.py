"""Voice mode for localai — push-to-talk via mlx-whisper + sounddevice.

Gracefully degrades: if dependencies aren't installed, check_available()
returns False and push_to_talk() returns None.

Install:
    pip install mlx-whisper sounddevice
"""

import threading

VOICE_AVAILABLE: bool = False

try:
    import mlx_whisper
    import numpy as np
    import sounddevice as sd
    VOICE_AVAILABLE = True
except ImportError:
    pass

WHISPER_MODEL = "mlx-community/whisper-base-mlx"
SAMPLE_RATE   = 16_000

_stream    = None
_frames: list = []
_recording = False
_lock      = threading.Lock()


def check_available() -> bool:
    """Return True if mlx-whisper and sounddevice are importable."""
    return VOICE_AVAILABLE


def install_instructions() -> str:
    return "pip install mlx-whisper sounddevice"


def _cb(indata, frame_count, time_info, status):
    with _lock:
        if _recording:
            _frames.append(indata.copy())


def _start() -> bool:
    global _stream, _frames, _recording
    if not VOICE_AVAILABLE:
        return False
    with _lock:
        _frames = []
        _recording = True
    try:
        _stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1,
            dtype="float32", callback=_cb,
        )
        _stream.start()
        return True
    except Exception:
        _recording = False
        return False


def _stop() -> str:
    global _stream, _recording
    with _lock:
        _recording = False
    if _stream:
        try:
            _stream.stop()
            _stream.close()
        except Exception:
            pass
        _stream = None
    with _lock:
        frames = list(_frames)
    if not frames or not VOICE_AVAILABLE:
        return ""
    try:
        audio  = np.concatenate(frames).flatten()
        result = mlx_whisper.transcribe(audio, path_or_hf_repo=WHISPER_MODEL)
        return result.get("text", "").strip()
    except Exception:
        return ""


def push_to_talk(prompt_fn=None) -> str | None:
    """Record until Enter is pressed, then transcribe and return text.

    Returns None (not empty string) if voice is unavailable.
    """
    if not VOICE_AVAILABLE:
        return None
    if prompt_fn:
        prompt_fn("  \033[1m\033[38;5;196m● REC\033[0m  press Enter to stop …")
    if not _start():
        if prompt_fn:
            prompt_fn("  \033[33m⚠  Failed to start microphone\033[0m")
        return None
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        global _stream, _recording, _frames
        with _lock:
            _recording = False
            _frames = []
        if _stream:
            try:
                _stream.stop()
                _stream.close()
            except Exception:
                pass
            _stream = None
        return None
    return _stop()
