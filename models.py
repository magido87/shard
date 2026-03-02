"""MODELS registry and helpers for localai."""

MODELS: dict = {
    # ── Tiny (1-3B) — fast, 8GB-friendly ─────────────────────────
    "qwen25_05b": {
        "id":           "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "label":        "Qwen 2.5 0.5B",
        "size":         "0.5B 4-bit",
        "min_ram":      2,
        "dl_size":      "0.4GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "ultra-tiny, basic tasks",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "qwen",
    },
    "qwen35_08b": {
        "id":           "mlx-community/Qwen3.5-0.8B-4bit",
        "label":        "Qwen 3.5 0.8B",
        "size":         "0.8B 4-bit",
        "min_ram":      2,
        "dl_size":      "0.6GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "tiny multimodal, Qwen 3.5",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "qwen",
    },
    "deepseek_r1_1_5b": {
        "id":           "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit",
        "label":        "DeepSeek R1 1.5B",
        "size":         "1.5B 4-bit",
        "min_ram":      3,
        "dl_size":      "0.9GB",
        "profile":      "safe",
        "category":     "reasoning",
        "desc":         "tiny reasoning, chain-of-thought",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "qwen",
        "think_prefill": True,
    },
    "smollm2_1_7b": {
        "id":           "mlx-community/SmolLM2-1.7B-Instruct",
        "label":        "SmolLM2 1.7B",
        "size":         "1.7B",
        "min_ram":      4,
        "dl_size":      "3.4GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "HuggingFace, lightweight",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "llama",
    },
    "qwen25_3b": {
        "id":           "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "label":        "Qwen 2.5 3B",
        "size":         "3B 4-bit",
        "min_ram":      4,
        "dl_size":      "1.8GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "fast general purpose",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "qwen",
    },
    "qwen35_2b": {
        "id":           "mlx-community/Qwen3.5-2B-4bit",
        "label":        "Qwen 3.5 2B",
        "size":         "2B 4-bit",
        "min_ram":      4,
        "dl_size":      "1.7GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "small & capable, Qwen 3.5",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "qwen",
    },
    # ── Small (4-8B) — workhorses ────────────────────────────────
    "phi4mini": {
        "id":           "mlx-community/Phi-4-mini-instruct-4bit",
        "label":        "Phi-4 Mini",
        "size":         "3.8B 4-bit",
        "min_ram":      4,
        "dl_size":      "2.3GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "fast, Microsoft reasoning",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "phi",
    },
    "phi35_mini": {
        "id":           "mlx-community/Phi-3.5-mini-instruct-4bit",
        "label":        "Phi-3.5 Mini",
        "size":         "3.8B 4-bit",
        "min_ram":      4,
        "dl_size":      "2.2GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "Microsoft, strong reasoning",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "phi",
    },
    "llama32_3b": {
        "id":           "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "label":        "Llama 3.2 3B",
        "size":         "3B 4-bit",
        "min_ram":      4,
        "dl_size":      "1.9GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "tiny, general purpose",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "llama",
    },
    "gemma3_4b": {
        "id":           "mlx-community/gemma-3-4b-it-4bit",
        "label":        "Gemma 3 4B",
        "size":         "4B 4-bit",
        "min_ram":      5,
        "dl_size":      "3.0GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "Google, well-balanced",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "gemma",
    },
    "qwen35_4b": {
        "id":           "mlx-community/Qwen3.5-4B-4bit",
        "label":        "Qwen 3.5 4B",
        "size":         "4B 4-bit",
        "min_ram":      5,
        "dl_size":      "2.5GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "balanced, Qwen 3.5",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "qwen",
    },
    "qwen25_coder_3b": {
        "id":           "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
        "label":        "Qwen Coder 3B",
        "size":         "3B 4-bit",
        "min_ram":      4,
        "dl_size":      "1.8GB",
        "profile":      "safe",
        "category":     "coding",
        "desc":         "code-specialized, tiny",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "qwen",
    },
    "qwen3_8b": {
        "id":           "mlx-community/Qwen3-8B-4bit",
        "label":        "Qwen3 8B",
        "size":         "8B 4-bit",
        "min_ram":      6,
        "dl_size":      "5.0GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "thinking mode, coding",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "qwen",
        "think_prefill": True,
    },
    "qwen35_9b": {
        "id":           "mlx-community/Qwen3.5-9B-4bit",
        "label":        "Qwen 3.5 9B",
        "size":         "9B 4-bit",
        "min_ram":      8,
        "dl_size":      "6.0GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "strong general, Qwen 3.5",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "qwen",
    },
    "dolphin7b": {
        "id":           "mlx-community/dolphin-2.9.2-qwen2-7b-4bit",
        "label":        "Dolphin 7B",
        "size":         "7B 4-bit",
        "min_ram":      6,
        "dl_size":      "4.3GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "uncensored, Qwen2 base",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "qwen",
    },
    "qwen25_coder_7b": {
        "id":           "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        "label":        "Qwen Coder 7B",
        "size":         "7B 4-bit",
        "min_ram":      6,
        "dl_size":      "4.3GB",
        "profile":      "safe",
        "category":     "coding",
        "desc":         "strong code completion",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "qwen",
    },
    "deepseek_coder_7b": {
        "id":           "mlx-community/deepseek-coder-6.7b-instruct-hf-4bit-mlx",
        "label":        "DeepSeek Coder 7B",
        "size":         "6.7B 4-bit",
        "min_ram":      6,
        "dl_size":      "3.8GB",
        "profile":      "safe",
        "category":     "coding",
        "desc":         "code generation specialist",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "llama",
    },
    "mistral7b": {
        "id":           "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "label":        "Mistral 7B",
        "size":         "7B 4-bit",
        "min_ram":      8,
        "dl_size":      "4.1GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "reliable workhorse",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "mistral",
    },
    "llama31_8b": {
        "id":           "mlx-community/Llama-3.1-8B-Instruct-4bit",
        "label":        "Llama 3.1 8B",
        "size":         "8B 4-bit",
        "min_ram":      8,
        "dl_size":      "4.9GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "solid all-rounder",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "llama",
    },
    "llama32_8b": {
        "id":           "mlx-community/Llama-3.2-8B-Instruct-4bit",
        "label":        "Llama 3.2 8B",
        "size":         "8B 4-bit",
        "min_ram":      8,
        "dl_size":      "4.9GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "Meta compact, efficient",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "llama",
    },
    "ministral_8b": {
        "id":           "mlx-community/Ministral-8B-Instruct-2410-4bit",
        "label":        "Ministral 8B",
        "size":         "8B 4-bit",
        "min_ram":      8,
        "dl_size":      "4.7GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "Mistral efficient series",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "mistral",
    },
    "deepseek_r1_8b": {
        "id":           "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit",
        "label":        "DeepSeek R1 8B",
        "size":         "8B 4-bit",
        "min_ram":      8,
        "dl_size":      "5.0GB",
        "profile":      "safe",
        "category":     "reasoning",
        "desc":         "chain-of-thought, math",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "llama",
        "think_prefill": True,
    },
    # ── Medium (12-14B) — strong performers ──────────────────────
    "mistral_nemo": {
        "id":           "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
        "label":        "Mistral Nemo 12B",
        "size":         "12B 4-bit",
        "min_ram":      10,
        "dl_size":      "6.9GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "Mistral/NVIDIA, versatile",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "mistral",
    },
    "gemma3_12b": {
        "id":           "mlx-community/gemma-3-12b-it-4bit",
        "label":        "Gemma 3 12B",
        "size":         "12B 4-bit",
        "min_ram":      10,
        "dl_size":      "7.0GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "Google, strong & balanced",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "gemma",
    },
    "phi4_14b": {
        "id":           "mlx-community/phi-4-4bit",
        "label":        "Phi-4 14B",
        "size":         "14B 4-bit",
        "min_ram":      12,
        "dl_size":      "8.0GB",
        "profile":      "safe",
        "category":     "reasoning",
        "desc":         "Microsoft, strong reasoning",
        "kv":           1536,
        "tokens":       768,
        "prompt_style": "phi",
    },
    "qwen25_coder_14b": {
        "id":           "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
        "label":        "Qwen Coder 14B",
        "size":         "14B 4-bit",
        "min_ram":      12,
        "dl_size":      "8.3GB",
        "profile":      "safe",
        "category":     "coding",
        "desc":         "best coding < 16GB",
        "kv":           1536,
        "tokens":       768,
        "prompt_style": "qwen",
    },
    "deepseek_r1_14b": {
        "id":           "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit",
        "label":        "DeepSeek R1 14B",
        "size":         "14B 4-bit",
        "min_ram":      12,
        "dl_size":      "8.3GB",
        "profile":      "safe",
        "category":     "reasoning",
        "desc":         "strong reasoning, math",
        "kv":           1536,
        "tokens":       768,
        "prompt_style": "qwen",
        "think_prefill": True,
    },
    "qwen25_14b": {
        "id":           "mlx-community/Qwen2.5-14B-Instruct-4bit",
        "label":        "Qwen 2.5 14B",
        "size":         "14B 4-bit",
        "min_ram":      14,
        "dl_size":      "8.2GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "coding + math",
        "kv":           1536,
        "tokens":       768,
        "prompt_style": "qwen",
    },
    "qwen3_14b": {
        "id":           "mlx-community/Qwen3-14B-4bit",
        "label":        "Qwen3 14B",
        "size":         "14B 4-bit",
        "min_ram":      10,
        "dl_size":      "8.5GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "best general < 16GB",
        "kv":           1536,
        "tokens":       768,
        "prompt_style": "qwen",
        "think_prefill": True,
    },
    # ── Large (20B+) — high-end Macs ────────────────────────────
    "qwen3_30b_moe": {
        "id":           "mlx-community/Qwen3-30B-A3B-4bit",
        "label":        "Qwen3 30B MoE",
        "size":         "30B MoE 4-bit",
        "min_ram":      14,
        "dl_size":      "9.5GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "MoE, fast for its size",
        "kv":           1536,
        "tokens":       768,
        "prompt_style": "qwen",
        "think_prefill": True,
    },
    "gemma3_27b": {
        "id":           "mlx-community/gemma-3-27b-it-4bit",
        "label":        "Gemma 3 27B",
        "size":         "27B 4-bit",
        "min_ram":      20,
        "dl_size":      "15.0GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "Google, premium quality",
        "kv":           1536,
        "tokens":       768,
        "prompt_style": "gemma",
    },
    "mistral24b": {
        "id":           "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit",
        "label":        "Mistral Small 24B",
        "size":         "24B 4-bit",
        "min_ram":      20,
        "dl_size":      "13.5GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "nuanced, powerful",
        "kv":           1536,
        "tokens":       768,
        "prompt_style": "mistral",
    },
    "qwen25_coder_32b": {
        "id":           "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
        "label":        "Qwen Coder 32B",
        "size":         "32B 4-bit",
        "min_ram":      24,
        "dl_size":      "18.4GB",
        "profile":      "safe",
        "category":     "coding",
        "desc":         "top-tier coding",
        "kv":           1024,
        "tokens":       512,
        "prompt_style": "qwen",
    },
    "qwen25_32b": {
        "id":           "mlx-community/Qwen2.5-32B-Instruct-4bit",
        "label":        "Qwen 2.5 32B",
        "size":         "32B 4-bit",
        "min_ram":      24,
        "dl_size":      "18.4GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "strong all-around 32B",
        "kv":           1024,
        "tokens":       512,
        "prompt_style": "qwen",
    },
    "deepseek_r1_32b": {
        "id":           "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit",
        "label":        "DeepSeek R1 32B",
        "size":         "32B 4-bit",
        "min_ram":      24,
        "dl_size":      "18.5GB",
        "profile":      "safe",
        "category":     "reasoning",
        "desc":         "powerful reasoning, math",
        "kv":           1024,
        "tokens":       512,
        "prompt_style": "qwen",
        "think_prefill": True,
    },
    "llama33_70b": {
        "id":           "mlx-community/Llama-3.3-70B-Instruct-4bit",
        "label":        "Llama 3.3 70B",
        "size":         "70B 4-bit",
        "min_ram":      48,
        "dl_size":      "38.0GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "flagship, near-frontier",
        "kv":           1024,
        "tokens":       512,
        "prompt_style": "llama",
    },
    "qwen3_235b_moe": {
        "id":           "mlx-community/Qwen3-235B-A22B-4bit",
        "label":        "Qwen3 235B MoE",
        "size":         "235B MoE 4-bit",
        "min_ram":      48,
        "dl_size":      "50.0GB",
        "profile":      "safe",
        "category":     "general",
        "desc":         "massive MoE, top quality",
        "kv":           1024,
        "tokens":       512,
        "prompt_style": "qwen",
        "think_prefill": True,
    },
    # ── Unfiltered (require --unfiltered + UNLOCK) ───────────────
    "dolphin8b": {
        "id":           "mlx-community/Dolphin3.0-Llama3.1-8B-4bit",
        "label":        "Dolphin 3.0 8B",
        "size":         "8B 4-bit",
        "min_ram":      8,
        "dl_size":      "4.9GB",
        "profile":      "unfiltered",
        "category":     "unfiltered",
        "desc":         "no filters, Llama 3.1",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "llama",
    },
    "dolphin294_8b": {
        "id":           "mlx-community/dolphin-2.9.4-llama3.1-8b-4bit",
        "label":        "Dolphin 2.9.4 8B",
        "size":         "8B 4-bit",
        "min_ram":      8,
        "dl_size":      "4.9GB",
        "profile":      "unfiltered",
        "category":     "unfiltered",
        "desc":         "no filters, creative",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "llama",
    },
    "dolphin_qwen_7b": {
        "id":           "mlx-community/dolphin-2.9.2-qwen2-7b-4bit",
        "label":        "Dolphin Qwen2 7B",
        "size":         "7B 4-bit",
        "min_ram":      6,
        "dl_size":      "4.3GB",
        "profile":      "unfiltered",
        "category":     "unfiltered",
        "desc":         "no filters, Qwen2 base",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "qwen",
    },
    "dolphin_mistral_7b": {
        "id":           "mlx-community/dolphin-2.8-mistral-7b-v02-4bit",
        "label":        "Dolphin Mistral 7B",
        "size":         "7B 4-bit",
        "min_ram":      8,
        "dl_size":      "4.1GB",
        "profile":      "unfiltered",
        "category":     "unfiltered",
        "desc":         "no filters, Mistral base",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "mistral",
    },
    "hermes7b": {
        "id":           "mlx-community/OpenHermes-2.5-Mistral-7B-4bit",
        "label":        "OpenHermes 7B",
        "size":         "7B 4-bit",
        "min_ram":      8,
        "dl_size":      "4.1GB",
        "profile":      "unfiltered",
        "category":     "unfiltered",
        "desc":         "no filters, roleplay",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "mistral",
    },
    "lexi_8b": {
        "id":           "mlx-community/Llama-3.1-8B-Lexi-Uncensored-v2-4bit",
        "label":        "Lexi Uncensored 8B",
        "size":         "8B 4-bit",
        "min_ram":      8,
        "dl_size":      "4.9GB",
        "profile":      "unfiltered",
        "category":     "unfiltered",
        "desc":         "fully uncensored Llama",
        "kv":           2048,
        "tokens":       1024,
        "prompt_style": "llama",
    },
}


def available_models(ram_gb: int, unfiltered: bool = False) -> list[tuple[str, dict]]:
    """Return list of (key, entry) tuples fitting RAM, filtered by profile."""
    out = []
    for key, entry in MODELS.items():
        if entry["min_ram"] > ram_gb:
            continue
        if entry["profile"] == "unfiltered" and not unfiltered:
            continue
        out.append((key, entry))
    return out


def recommend_model(hw: dict) -> str:
    """Return key of the largest safe model fitting available RAM with pressure penalty."""
    effective_ram = min(hw["ram"], hw["ram_available"] + 1.5)

    pressure = hw.get("pressure", "low")
    if pressure == "high":
        effective_ram *= 0.75
    elif pressure == "medium":
        effective_ram *= 0.88

    candidates = [(k, v) for k, v in MODELS.items()
                  if v["profile"] == "safe" and v["min_ram"] <= effective_ram]
    if not candidates:
        # Return the smallest safe model instead of hardcoding a specific key
        safe = [(k, v) for k, v in MODELS.items() if v["profile"] == "safe"]
        return min(safe, key=lambda x: x[1]["min_ram"])[0]
    # Prefer "general" category when models tie on min_ram
    return max(candidates, key=lambda x: (
        x[1]["min_ram"], x[1]["category"] == "general"
    ))[0]


def model_fit(key: str, hw: dict) -> str:
    """Return fit label for a model given current memory availability."""
    min_ram = MODELS[key]["min_ram"]
    avail   = hw["ram_available"]
    if min_ram <= avail - 0.5:
        return "✓ fits"
    if min_ram <= avail + 2:
        return "⚠ tight"
    return "✗ risky"


def disk_ok(key: str, hw: dict) -> bool:
    """Return True if there's at least 1GB of disk headroom above the model's download size."""
    dl = float(MODELS[key]["dl_size"].rstrip("GB"))
    return hw["disk_free"] > dl + 1.0


# ── Category helpers ─────────────────────────────────────────────

CATEGORY_LABELS: dict = {
    "general":    "General",
    "coding":     "Coding",
    "reasoning":  "Reasoning",
    "unfiltered": "Unfiltered",
}

CATEGORY_ORDER: list = ["general", "coding", "reasoning", "unfiltered"]


def top_picks(models: list[tuple[str, dict]], hw: dict) -> list[tuple[str, dict]]:
    """Select best model per category from available models for the compact picker view."""
    rec_key = recommend_model(hw)
    picks = []
    seen_cats: set = set()

    # Always include recommended first
    for k, v in models:
        if k == rec_key:
            picks.append((k, v))
            seen_cats.add(v["category"])
            break

    # Best (largest min_ram that fits) per category
    for cat in CATEGORY_ORDER:
        if cat == "unfiltered":
            continue  # shown separately
        cat_models = [(k, v) for k, v in models if v["category"] == cat and k != rec_key]
        if not cat_models:
            continue
        # Pick up to 2 per category — largest first
        cat_models.sort(key=lambda x: x[1]["min_ram"], reverse=True)
        added = 0
        for k, v in cat_models:
            if added >= 2:
                break
            if k not in [p[0] for p in picks]:
                picks.append((k, v))
                added += 1

    return picks


def grouped_by_category(models: list[tuple[str, dict]]) -> list[tuple[str, list[tuple[str, dict]]]]:
    """Group models by category in display order."""
    groups: dict = {}
    for k, v in models:
        cat = v["category"]
        groups.setdefault(cat, []).append((k, v))
    return [(cat, groups[cat]) for cat in CATEGORY_ORDER if cat in groups]
