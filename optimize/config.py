"""Settings for the AI advisor. Keys go into a .env file next to app.py (see .env.example)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"          # real public datasets (see data/SOURCES.md)


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env_file(ROOT / ".env")

# auto = every AI that has a key (Groq -> Gemini -> Anthropic -> OpenAI) or is running (Ollama); falls back to
# the next one when one fails, and to a rule-based summary when none is available. Or: groq, ollama, rules, ...
LLM_PROVIDER = os.environ.get("KAIZEN_LLM", "auto")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("KAIZEN_OLLAMA_MODEL", "qwen2.5:7b")
GROQ_MODEL = os.environ.get("KAIZEN_GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.environ.get("KAIZEN_GEMINI_MODEL", "gemini-flash-latest")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
ANTHROPIC_MODEL = os.environ.get("KAIZEN_ANTHROPIC_MODEL", "claude-haiku-5-5")
OPENAI_MODEL = os.environ.get("KAIZEN_OPENAI_MODEL", "gpt-4.1-mini")
LLM_TIMEOUT_S = int(os.environ.get("KAIZEN_LLM_TIMEOUT", "120"))
