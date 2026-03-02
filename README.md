# localai

Offline LLM chat for Apple Silicon. No cloud, no telemetry, no history. Runs entirely on your Mac via [MLX](https://github.com/ml-explore/mlx).

```
llm
```

## Install

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/magido87/shard/main/install.sh)
```

Open a new terminal and run `llm`. A first-run wizard picks the right model for your hardware.

**From source:**

```bash
git clone https://github.com/magido87/shard.git ~/.localai
cd ~/.localai && bash setup.sh
```

## Requirements

- macOS 13+ on Apple Silicon (M1 through M4, any variant)
- Python 3.10+
- 8 GB unified memory minimum (16 GB+ recommended)

## Models

43 models spanning four categories. The picker auto-recommends based on your available RAM.

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

| Model | Size | Download | Min RAM |
|-------|------|----------|---------|
| Qwen Coder 3B | 3B 4-bit | 1.8 GB | 4 GB |
| Qwen Coder 7B | 7B 4-bit | 4.3 GB | 6 GB |
| DeepSeek Coder 7B | 6.7B 4-bit | 3.8 GB | 6 GB |
| Qwen Coder 14B | 14B 4-bit | 8.3 GB | 12 GB |
| Qwen Coder 32B | 32B 4-bit | 18.4 GB | 24 GB |

### Reasoning

| Model | Size | Download | Min RAM |
|-------|------|----------|---------|
| DeepSeek R1 1.5B | 1.5B 4-bit | 0.9 GB | 3 GB |
| DeepSeek R1 8B | 8B 4-bit | 5.0 GB | 8 GB |
| Phi-4 14B | 14B 4-bit | 8.0 GB | 12 GB |
| DeepSeek R1 14B | 14B 4-bit | 8.3 GB | 12 GB |
| DeepSeek R1 32B | 32B 4-bit | 18.5 GB | 24 GB |

Reasoning models (DeepSeek R1, Qwen3) use chain-of-thought internally but only display the final answer. The thinking process is filtered automatically.

### Unfiltered

Available with `llm --unfiltered` (requires confirmation):

Dolphin 3.0 8B, Dolphin 2.9.4 8B, Dolphin Qwen2 7B, Dolphin Mistral 7B, OpenHermes 7B, Lexi Uncensored 8B.

## Usage

```bash
llm                  # start with last used model
llm --unfiltered     # unlock unfiltered models
llm --voice          # push-to-talk via mlx-whisper
llm --zen            # minimal UI, no header or stats
llm --focus          # header shown once, then hidden
llm --model KEY      # force a specific model key
llm-stop             # kill a running session
```

### In-chat commands

| Command | Action |
|---------|--------|
| `q` | Quit |
| `r` | Reset context |
| `s` | Settings menu |
| `h` | Help |
| `v` | Toggle voice mode |
| `rensa` / `clear` | Clear screen |

### Settings

Press `s` during chat to configure:

1. **Temperature** &mdash; 6 presets from 0.05 (frozen) to 1.0 (wild), default 0.72
2. **Personality** &mdash; Dev, Buddy, Ghost, Sensei, Hacker, Analyst
3. **Stats** &mdash; compact, full, or off
4. **UI mode** &mdash; normal, zen, focus
5. **Privacy** &mdash; toggle privacy mode (nothing saved) and session logging
6. **Custom instructions** &mdash; appended to system prompt for the session
7. **Plugins** &mdash; install, enable, disable, uninstall

All settings persist in `~/.localai/config.json` (excluded from git).

## Plugins

Built-in plugin system with install/toggle from the settings menu:

| Plugin | Description |
|--------|-------------|
| Web Search | Search via DuckDuckGo |
| Code Runner | Execute Python/shell from responses |
| File Reader | Load local files into context (RAG) |
| Text-to-Speech | Read responses via macOS `say` |
| Shell Assistant | Suggest and run shell commands |
| Summarizer | Summarize text or fetch URLs |
| Translator | Auto-translate responses |
| Clipboard | Copy last response to clipboard |

## Themes

Four color themes, switchable in settings: **Ocean** (default), **Dusk**, **Mono**, **Forest**.

## Voice Mode

Requires additional dependencies:

```bash
pip install mlx-whisper sounddevice
```

Runs Whisper locally via MLX. Press Enter to stop recording; transcription feeds directly into the chat.

## Privacy

- No telemetry or network calls during chat
- No chat history written to disk (unless session logging is explicitly enabled)
- Privacy mode disables all config saving
- Model weights download from Hugging Face on first use only

## Project structure

```
~/.localai/
  chat.py          Main app: model picker, chat loop, progress bar, UI
  models.py        Model registry (43 models) and hardware matching
  ui.py            ANSI themes, stream highlighter, think filter
  config.py        Persistent JSON config
  detect.py        Hardware detection (chip, RAM, disk, memory pressure)
  agent.py         Session logging (opt-in only)
  voice.py         Push-to-talk via mlx-whisper (optional)
  plugins.py       Plugin loader and hook dispatcher
  plugins/         Plugin implementations
  install.sh       One-command remote installer
  setup.sh         Local setup: venv, deps, ~/bin wrappers
  llm.sh           ~/bin/llm wrapper
  llm-stop.sh      ~/bin/llm-stop wrapper
  wipe_session.sh  Session cleanup
```

## License

MIT
