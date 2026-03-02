# LocalAI — MLX Chat Project

## Vad är det här?
Lokalt AI-chatgränssnitt för Apple Silicon. Kör LLM offline via MLX. Startas med `llm` i terminalen.

## Projektstruktur
- `chat.py` — huvudapp (modellval, chat-loop, UI, teman, personligheter)
- `agent.py` — interaction logging (disabled för privacy)
- `setup.sh` — installerar venv + deps + `~/bin/llm` wrappers
- `wipe_session.sh` — säker rensning av session-artefakter
- `llm.sh` / `llm-stop.sh` — bin-wrappers som körs via `~/bin/`

## Deploy-struktur (live)
```
~/.localai/          ← live-kopian som `llm` pekar på
  chat.py
  agent.py
  setup.sh
  wipe_session.sh
  venv/              ← Python venv (skapas av setup.sh, inte i git)

~/bin/
  llm                ← startar chat.py via venv
  llm-stop           ← dödar pågående session
```

## Modeller (MODELS dict i chat.py)
| Key | Model | VRAM |
|-----|-------|------|
| dolphin | mlx-community/Dolphin3.0-Llama3.1-8B-4bit | ~5 GB |
| qwen | mlx-community/Qwen2.5-14B-Instruct-4bit | ~9 GB |
| qwen35 | mlx-community/Qwen3.5-27B-4bit | ~16 GB |

## Nyckelkonstanter (chat.py)
- `TEMP = 0.72`, `TOP_P = 0.92`
- `MAX_KV` / `MAX_TOKENS` — per modell i `MODELS` dict
- `PERSONALITIES` — 11 systemprompt-presets (1–11)
- `THEMES` — neon / ocean (256-color ANSI)

## Deploy: uppdatera live-kopian
```bash
cp chat.py ~/.localai/chat.py
```

## Setup (ny maskin)
```bash
cd ~/.localai && bash setup.sh
```

## Kommandon i chatten
- `q` quit, `r` reset, `s` settings, `rensa` clear, `h` help

## Beroenden
- Python 3 + venv
- `mlx-lm`, `psutil`
- Apple Silicon (MLX kräver Metal GPU)
