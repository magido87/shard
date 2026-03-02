# Contributing to localai

Thanks for your interest. This is a CLI-first, offline-only project — keep that spirit.

## Ground rules

- **No cloud calls** — nothing should phone home during runtime
- **No chat history** — conversations must not be written to disk by default
- **No GUI** — terminal UI only, ANSI colors are fine
- **Apple Silicon only** — MLX requires M1 or later; don't add CPU-only fallbacks that break the MLX path

## What's welcome

- New model entries in `MODELS` in `models.py` (with correct HuggingFace ID and all required keys)
- Bug fixes and edge case handling
- New personalities in `PERSONALITIES`
- Voice backend alternatives via `voice.py` (keep the interface: `push_to_talk() -> str`)
- Hardware detection improvements in `detect.py`
- Better README / docs

## What to avoid

- External API calls or cloud model providers
- GUI frameworks (tkinter, web UI, etc.)
- Persistent chat logging
- Dependencies beyond `mlx-lm`, `psutil`, and optional `mlx-whisper`/`sounddevice`

## How to add a model

1. Verify the model ID exists on Hugging Face and is MLX-compatible
2. Add an entry to `MODELS` in `models.py`:
   ```python
   "my_key": {
       "id":           "mlx-community/ModelName-4bit",
       "label":        "Model Name",
       "size":         "14B",
       "min_ram":      12,           # GB required to run safely
       "dl_size":      8.5,          # download size in GB (float)
       "profile":      "safe",       # "safe" or "unfiltered"
       "category":     "general",    # "general", "coding", "reasoning", "unfiltered"
       "desc":         "Short description",
       "kv":           4096,         # context window in tokens
       "tokens":       2048,         # max new tokens per response
       "prompt_style": "qwen",       # "qwen", "llama", "phi", "gemma", "deepseek", etc.
   },
   ```
3. Pick `category` from: `general`, `coding`, `reasoning`, `unfiltered`
4. `prompt_style` controls the chat template — match it to the model family

## Dev setup

```bash
git clone <repo>
cd localai
python3 -m venv venv
source venv/bin/activate
pip install mlx-lm psutil
python3 chat.py
```

## Pull requests

- One logical change per PR
- Test on at least one Apple Silicon machine
- Keep the ANSI CLI aesthetic

## Issues

Report bugs, suggest models, or request features via GitHub Issues.
