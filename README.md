# localai

A private AI chat that runs on your Mac. No cloud, no account, no subscription. Everything stays on your computer.

```
llm
```

---

## Quick Start (2 minutes)

**Step 1 — Open Terminal**

Press `Cmd + Space`, type **Terminal**, press Enter.

**Step 2 — Paste this one line**

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/magido87/shard/main/install.sh)
```

Press Enter. It installs everything automatically.

**Step 3 — Start chatting**

Close Terminal, open it again, then type:

```
llm
```

A model picker appears. Press Enter to accept the recommended model. It downloads once (a few minutes), then you're chatting.

**That's it. You're running AI locally.**

---

## What You Need

- A Mac with Apple Silicon (M1, M2, M3, or M4 — any model)
- macOS 13 or newer
- 8 GB RAM minimum (16 GB recommended for better models)

Not sure what chip you have? Click the Apple menu top-left, click **About This Mac**. It says "Chip: Apple M1/M2/M3/M4".

---

## How It Works

When you run `llm`, it checks your Mac's specs and recommends an AI model that fits. Smaller models are faster but less smart. Bigger models are smarter but need more RAM.

The model downloads from Hugging Face the first time (one-time only). After that, everything runs offline. No internet needed.

---

## Models

43 models across four categories. The picker auto-recommends based on your available RAM.

### Which model should I pick?

| Your Mac | Recommended | Why |
|----------|-------------|-----|
| 8 GB RAM | Qwen3 8B | Best balance of speed and quality |
| 8 GB RAM (tight) | Qwen 3.5 4B | Lighter, still good |
| 16 GB RAM | Qwen3 14B | Strong all-around |
| 24 GB+ RAM | Qwen 2.5 32B | Top quality |

Don't worry about picking wrong — you can always switch by running `llm` again.

### General

| Model | Size | Download | Min RAM |
|-------|------|----------|---------|
| Qwen 2.5 0.5B | 0.5B 4-bit | 0.4 GB | 2 GB |
| Qwen 3.5 0.8B | 0.8B 4-bit | 0.6 GB | 2 GB |
| SmolLM2 1.7B | 1.7B | 3.4 GB | 4 GB |
| Qwen 2.5 3B | 3B 4-bit | 1.8 GB | 4 GB |
| Qwen 3.5 2B | 2B 4-bit | 1.7 GB | 4 GB |
| Phi-4 Mini | 3.8B 4-bit | 2.3 GB | 4 GB |
| Phi-3.5 Mini | 3.8B 4-bit | 2.2 GB | 4 GB |
| Llama 3.2 3B | 3B 4-bit | 1.9 GB | 4 GB |
| Gemma 3 4B | 4B 4-bit | 3.0 GB | 5 GB |
| Qwen 3.5 4B | 4B 4-bit | 2.5 GB | 5 GB |
| Qwen3 8B | 8B 4-bit | 5.0 GB | 6 GB |
| Dolphin 7B | 7B 4-bit | 4.3 GB | 6 GB |
| Qwen 3.5 9B | 9B 4-bit | 6.0 GB | 8 GB |
| Mistral 7B | 7B 4-bit | 4.1 GB | 8 GB |
| Llama 3.1 8B | 8B 4-bit | 4.9 GB | 8 GB |
| Llama 3.2 8B | 8B 4-bit | 4.9 GB | 8 GB |
| Ministral 8B | 8B 4-bit | 4.7 GB | 8 GB |
| Mistral Nemo 12B | 12B 4-bit | 6.9 GB | 10 GB |
| Gemma 3 12B | 12B 4-bit | 7.0 GB | 10 GB |
| Qwen 2.5 14B | 14B 4-bit | 8.2 GB | 14 GB |
| Qwen3 14B | 14B 4-bit | 8.5 GB | 10 GB |
| Qwen3 30B MoE | 30B MoE 4-bit | 9.5 GB | 14 GB |
| Gemma 3 27B | 27B 4-bit | 15.0 GB | 20 GB |
| Mistral Small 24B | 24B 4-bit | 13.5 GB | 20 GB |
| Qwen 2.5 32B | 32B 4-bit | 18.4 GB | 24 GB |
| Llama 3.3 70B | 70B 4-bit | 38.0 GB | 48 GB |
| Qwen3 235B MoE | 235B MoE 4-bit | 50.0 GB | 48 GB |

### Coding

Models fine-tuned for writing and explaining code.

| Model | Size | Download | Min RAM |
|-------|------|----------|---------|
| Qwen Coder 3B | 3B 4-bit | 1.8 GB | 4 GB |
| Qwen Coder 7B | 7B 4-bit | 4.3 GB | 6 GB |
| DeepSeek Coder 7B | 6.7B 4-bit | 3.8 GB | 6 GB |
| Qwen Coder 14B | 14B 4-bit | 8.3 GB | 12 GB |
| Qwen Coder 32B | 32B 4-bit | 18.4 GB | 24 GB |

### Reasoning

Models that think step-by-step before answering. Great for math, logic, and complex questions. The thinking is hidden — you only see the final answer.

| Model | Size | Download | Min RAM |
|-------|------|----------|---------|
| DeepSeek R1 1.5B | 1.5B 4-bit | 0.9 GB | 3 GB |
| DeepSeek R1 8B | 8B 4-bit | 5.0 GB | 8 GB |
| Phi-4 14B | 14B 4-bit | 8.0 GB | 12 GB |
| DeepSeek R1 14B | 14B 4-bit | 8.3 GB | 12 GB |
| DeepSeek R1 32B | 32B 4-bit | 18.5 GB | 24 GB |

### Unfiltered

Models with no content filters. Available with `llm --unfiltered` (asks for confirmation first):

Dolphin 3.0 8B, Dolphin 2.9.4 8B, Dolphin Qwen2 7B, Dolphin Mistral 7B, OpenHermes 7B, Lexi Uncensored 8B.

---

## Commands

### Starting up

```bash
llm                  # start chatting (picks up where you left off)
llm --unfiltered     # unlock unfiltered models
llm --voice          # talk instead of type (needs mic)
llm --zen            # clean minimal UI
llm --focus          # header shown once, then hidden
llm --model KEY      # pick a specific model
llm-stop             # kill a running session
```

### While chatting

| Type this | What it does |
|-----------|-------------|
| `q` | Quit |
| `r` | Start fresh (clear conversation memory) |
| `s` | Open settings |
| `h` | Show help |
| `v` | Toggle voice mode on/off |
| `clear` | Clear the screen |

### Settings

Press `s` during a chat to change:

1. **Temperature** — how creative the AI is (low = precise, high = creative)
2. **Personality** — Dev, Buddy, Ghost, Sensei, Hacker, Analyst
3. **Stats** — show speed/memory info (compact, full, or off)
4. **UI mode** — normal, zen (minimal), focus
5. **Privacy** — toggle privacy mode (nothing saved to disk)
6. **Custom instructions** — tell the AI how to behave
7. **Plugins** — add extra features

Settings are saved between sessions.

---

## Plugins

Press `s` > `7` during chat to manage plugins:

| Plugin | What it does |
|--------|-------------|
| Web Search | Search DuckDuckGo from chat |
| Code Runner | Run Python/shell code from responses |
| File Reader | Load local files into the conversation |
| Text-to-Speech | AI reads responses out loud |
| Shell Assistant | Suggests and runs terminal commands |
| Summarizer | Summarize text or web pages |
| Translator | Auto-translate responses |
| Clipboard | Copy the last response |

---

## Themes

Four color themes: **Ocean** (default), **Dusk**, **Mono**, **Forest**. Change in settings.

## Voice Mode

Talk to the AI instead of typing. Requires a one-time setup:

```bash
pip install mlx-whisper sounddevice
```

Uses Whisper (runs locally, not cloud). Press Enter to stop recording.

---

## Privacy

- Zero telemetry — no data leaves your Mac
- No chat history saved (unless you turn on session logging)
- Privacy mode disables all config saving
- Models download from Hugging Face once, then it's fully offline

---

## Install from Source

If you prefer to clone the repo manually:

```bash
git clone https://github.com/magido87/shard.git ~/.localai
cd ~/.localai && bash setup.sh
```

## Uninstall

```bash
rm -rf ~/.localai ~/bin/llm ~/bin/llm-stop
```

## License

MIT
