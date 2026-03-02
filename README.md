# localai

**Offline LLM chat for Apple Silicon. No cloud. No history. No bullshit.**

Run powerful open-source language models locally in your terminal — from MacBook Air M1 to M4 Max. One command to start, one to stop.

```
llm
```

---

## What it is

localai is a minimal terminal chat client for running open-source LLMs fully offline on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). It detects your chip and RAM, recommends a model that fits, downloads it once, and then runs entirely on your GPU with no network calls. Streaming output, 4 themes, 6 personalities, 8 plugins, and push-to-talk voice — all in a single `llm` command.

---

## Install

**One command:**

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/magido87/shard/master/install.sh)
```

Open a new terminal tab and run `llm`. The first launch walks you through model selection; after that just type `llm` and chat.

**Manual (git):**

```bash
git clone https://github.com/magido87/shard.git localai
cd localai
bash setup.sh
```

Setup detects your Mac, suggests a model that fits, and installs the `llm` command in `~/bin/`. Run `source ~/.zshrc` if `llm` is not found immediately.

---

## Requirements

| Requirement | Minimum |
|-------------|---------|
| macOS | 13 Ventura or later |
| Chip | Apple Silicon (M1 or later, any variant) |
| Python | 3.10+ |
| RAM | 8 GB (16 GB+ recommended) |

---

## Usage

```bash
llm                  # start with last used model and settings
llm --model phi4mini # skip picker, load a specific model
llm --unfiltered     # unlock uncensored models (confirmation required)
llm --zen            # minimal UI: no header, no stats
llm --focus          # header on start only
llm --voice          # enable push-to-talk voice input
llm-stop             # kill a running session
```

### CLI flags

| Flag | Effect |
|------|--------|
| `--model KEY` | Skip picker, load model by key (e.g. `phi4mini`, `llama8b`) |
| `--unfiltered` | Show uncensored models; prompts for confirmation |
| `--zen` | No header, no stats |
| `--focus` | Header shown once at startup only |
| `--voice` | Enable push-to-talk on launch |

---

## Models

38 models across 4 categories. Setup auto-recommends based on your RAM.

| Category | Count | RAM range | Examples |
|----------|-------|-----------|---------|
| General | 22 | 2–48 GB | Qwen3 0.5B → 235B MoE, Llama 3.3 70B, Gemma 3 27B |
| Coding | 5 | 3–24 GB | Qwen Coder 3B/7B/14B/32B, DeepSeek Coder 7B |
| Reasoning | 5 | 3–20 GB | DeepSeek R1 1.5B/8B/14B/32B, Phi-4 14B |
| Unfiltered | 6 | 5–24 GB | Dolphin, OpenHermes, Lexi variants |

### RAM tiers

| Tier | RAM needed | Example models |
|------|-----------|----------------|
| Tiny | 2–4 GB | Qwen3 0.5B, SmolLM2 1.7B, Qwen3 3B, DeepSeek R1 1.5B |
| Small | 4–8 GB | Phi-4 Mini, Gemma 3 4B, Llama 3.1 8B, Qwen3 8B |
| Medium | 10–14 GB | Gemma 3 12B, Mistral Nemo 12B, Phi-4 14B, Qwen3 14B |
| Large | 14–48 GB | Gemma 3 27B, Mistral Small 24B, Qwen3 30B MoE, Llama 3.3 70B |

All models are 4-bit quantized MLX builds from [mlx-community](https://huggingface.co/mlx-community) unless noted. They download once to `~/.cache/huggingface/` and run fully offline after that.

---

## Plugins

8 bundled plugins, installed and toggled from the settings menu (`s` → Plugins).

| Plugin | Commands | Deps |
|--------|----------|------|
| Web Search | `/search <query>`, `search:` prefix | duckduckgo-search |
| Code Runner | `/run <python>`, `/shell <cmd>` | none |
| File Reader | `/read <path>`, `/files`, `/clearfiles` | none |
| Text-to-Speech | `/tts` (toggle), `/say <text>` | none (macOS say) |
| Shell Assistant | `/sh <command>` | none |
| Summarizer | `/summarize <url\|text>`, `summarize:` prefix | requests |
| Translator | `/translate <lang>`, `/translate off` | none |
| Clipboard | `/copy` | none (macOS pbcopy) |

---

## In-chat commands

| Input | Action |
|-------|--------|
| `q` / `quit` / `exit` | Quit |
| `r` / `reset` | Reset context, clear GPU cache |
| `s` | Settings menu |
| `h` | Show help bar |
| `v` | Toggle voice mode |
| `rensa` / `clear` / `cls` | Clear screen |
| `/search`, `/run`, `/sh`, etc. | Plugin commands (if installed) |

---

## Themes

Switch theme in settings (`s`):

| Theme | Description |
|-------|-------------|
| `ocean` | Deep blue — default |
| `dusk` | Purple/gray |
| `mono` | Monochrome |
| `forest` | Green |

---

## Personalities

Switch personality in settings (`s`):

| # | Name | Style |
|---|------|-------|
| 1 | Dev | Terse, technical, code-first |
| 2 | Buddy | Friendly, conversational |
| 3 | Ghost | Ultra-brief, one sentence |
| 4 | Sensei | Detailed teacher, step-by-step |
| 5 | Hacker | Security and ops focus |
| 6 | Analyst | Structured, analytical |

---

## Voice mode

Requires optional dependencies (~500 MB):

```bash
bash setup.sh --voice
```

Then launch with `llm --voice`. Hold a key to record, release to transcribe via [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper). Everything runs offline.

---

## Unfiltered mode

Unlocks uncensored models not shown in the standard picker:

```bash
llm --unfiltered
```

You will be prompted to type `UNLOCK` to confirm. No extra config needed.

---

## Privacy

- **No logging** — conversations live only in RAM, gone on exit
- **No telemetry** — zero network calls during chat
- **No history file** — nothing is written to disk by default
- **Privacy mode** — enable in settings to block all config writes too

Session logging is opt-in via settings and writes JSONL to `~/.localai/logs/`.

---

## Project structure

```
localai/
├── chat.py          Entry point — model picker, chat loop, UI, commands
├── models.py        38-model registry, RAM matching, category grouping
├── plugins.py       Plugin registry, loader, installer, hook dispatchers
├── plugins/         8 bundled plugin files
├── config.py        Persistent config (~/.localai/config.json)
├── detect.py        Hardware detection — chip, RAM, disk, pressure
├── ui.py            4 themes, StreamHighlighter, ANSI helpers
├── voice.py         Optional push-to-talk (mlx-whisper)
├── agent.py         Session logging (opt-in JSONL)
├── setup.sh         Deploy to ~/.localai/ + write ~/bin/llm wrappers
└── install.sh       One-liner bootstrap (curl | bash)
```

---

## License

MIT — see [LICENSE](LICENSE)
