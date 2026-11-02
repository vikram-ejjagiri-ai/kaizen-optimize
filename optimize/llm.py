"""Small wrapper around the language model backends (Ollama locally, or an API, or none)."""
from __future__ import annotations

import os
import time

import requests

from optimize import config


class LLMError(RuntimeError):
    pass


def ollama_running() -> bool:
    try:
        return requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=1.5).ok
    except requests.RequestException:
        return False


KEY_OF = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
          "openai": "OPENAI_API_KEY"}
COMPAT = {"groq": "https://api.groq.com/openai/v1", "gemini": config.GEMINI_URL}


def available_providers() -> list[str]:
    """Every AI that can be used, in order of preference. Puppy falls back to the next one when one fails
    (for example a Gemini key with quota 0)."""
    out = [p for p in ("groq", "gemini", "anthropic", "openai") if os.environ.get(KEY_OF[p], "").strip()]
    if ollama_running():
        out.append("ollama")
    return out


def detect_provider() -> str:
    av = available_providers()
    return av[0] if av else "rules"


class LLM:
    def __init__(self, provider: str | None = None):
        p = (provider or config.LLM_PROVIDER).lower()
        self.auto = p == "auto"
        self.chain = available_providers() if self.auto else ([p] if p != "rules" else [])
        self.provider = self.chain[0] if self.chain else "rules"
        self.models = {"groq": config.GROQ_MODEL, "gemini": config.GEMINI_MODEL,
                       "anthropic": config.ANTHROPIC_MODEL, "openai": config.OPENAI_MODEL}
        self.dead: dict[str, float] = {}       # provider -> time it failed (skipped for 10 minutes)
        self.last_error = ""
        self._checked = time.time()

    def refresh(self) -> None:
        """Ollama may be started after Puppy: in auto mode look again (at most every 20 s)."""
        if self.auto and self.provider == "rules" and time.time() - self._checked > 20:
            self._checked = time.time()
            self.chain = available_providers()
            self.provider = self.chain[0] if self.chain else "rules"

    @property
    def enabled(self) -> bool:
        return self.provider != "rules"

    @property
    def name(self) -> str:
        if self.provider == "rules":
            return "rules (no LLM)"
        if self.provider == "ollama":
            return f"ollama:{self.ollama_model()}"
        return f"{self.provider}:{self.models[self.provider]}"

    def ollama_model(self) -> str:
        """Use the configured model if it is installed, else any installed chat model (not an embedding model)."""
        if getattr(self, "_ollama_model", None):
            return self._ollama_model
        name = config.OLLAMA_MODEL
        try:
            have = [m["name"] for m in requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5).json()["models"]]
            chat = [m for m in have if "embed" not in m and "bge" not in m]
            if name not in have and f"{name}:latest" not in have and chat:
                print(f"[puppy] Ollama model {name} is not installed, using {chat[0]} (or run: ollama pull {name})")
                name = chat[0]
        except (requests.RequestException, KeyError, ValueError):
            pass
        self._ollama_model = name
        return name

    # ---- one provider -------------------------------------------------------------------------
    def _compat_models(self, p: str) -> list[str]:
        """Model names change often: list the chat models this key can use, best first."""
        r = requests.get(f"{COMPAT[p]}/models", timeout=20, headers={"Authorization": f"Bearer {os.environ[KEY_OF[p]]}"})
        r.raise_for_status()
        ids = [m["id"].removeprefix("models/") for m in r.json().get("data", [])]
        bad = ("image", "tts", "audio", "live", "whisper", "guard", "embed", "vision", "orpheus", "compound", "native")
        ids = [i for i in ids if not any(b in i for b in bad)]
        if p == "gemini":
            flash = sorted((i for i in ids if "flash" in i), reverse=True)        # newest first
            return [i for i in flash if "lite" not in i] + [i for i in flash if "lite" in i]
        pref = ("llama-3.3-70b", "gpt-oss-120b", "qwen", "gpt-oss-20b", "llama")
        return sorted(ids, key=lambda i: next((n for n, k in enumerate(pref) if k in i), 99))

    def _call_compat(self, p: str, system: str, user: str, max_tokens: int) -> str:
        hdr = {"Authorization": f"Bearer {os.environ[KEY_OF[p]]}"}
        extra = 600 if p == "gemini" else 0          # thinking models need room
        tried, candidates = set(), None
        for _ in range(4):
            model = self.models[p]
            tried.add(model)
            r = requests.post(f"{COMPAT[p]}/chat/completions", timeout=config.LLM_TIMEOUT_S, headers=hdr, json={
                "model": model, "temperature": 0, "max_tokens": max_tokens + extra,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
            if r.ok:
                return (r.json()["choices"][0]["message"].get("content") or "").strip()
            body = r.text
            quota_zero = r.status_code == 429 and ("limit: 0" in body or "free_tier" in body or "quota" in body.lower())
            if r.status_code in (400, 404) and "model" in body.lower() or quota_zero:
                # unknown model, or no free quota for this model: try the next model on this key
                candidates = candidates if candidates is not None else self._compat_models(p)
                nxt = next((m for m in candidates if m not in tried), None)
                if nxt:
                    self.models[p] = nxt
                    continue
            if r.status_code == 429 and not quota_zero:
                time.sleep(5)                          # per-minute limit: wait once
                continue
            raise LLMError(f"{p} {r.status_code}: {' '.join(body.split())[:300]}")
        raise LLMError(f"{p}: no model with free quota on this key")

    def _call(self, p: str, system: str, user: str, max_tokens: int) -> str:
        if p in COMPAT:
            return self._call_compat(p, system, user, max_tokens)
        if p == "ollama":
            r = requests.post(f"{config.OLLAMA_URL}/api/chat", timeout=config.LLM_TIMEOUT_S, json={
                "model": self.ollama_model(), "stream": False, "keep_alive": "30m",
                "options": {"temperature": 0, "num_ctx": 6144, "num_predict": max_tokens},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        if p == "anthropic":
            r = requests.post("https://api.anthropic.com/v1/messages", timeout=config.LLM_TIMEOUT_S, headers={
                "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"}, json={
                "model": self.models["anthropic"], "max_tokens": max_tokens, "temperature": 0, "system": system,
                "messages": [{"role": "user", "content": user}]})
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json()["content"]).strip()
        if p == "openai":
            base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            r = requests.post(f"{base}/chat/completions", timeout=config.LLM_TIMEOUT_S, headers={
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}, json={
                "model": self.models["openai"], "temperature": 0, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        raise LLMError("no language model configured")

    # ---- with fallback --------------------------------------------------------------------------
    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        errors = []
        order = [p for p in self.chain if time.time() - self.dead.get(p, 0) > 600] or self.chain
        for p in order:
            try:
                out = self._call(p, system, user, max_tokens)
                if p != self.provider:
                    print(f"[puppy] now using {p} (previous AI failed)")
                self.provider = p
                return out
            except (requests.RequestException, KeyError, ValueError, IndexError, LLMError) as e:
                msg = str(e)
                errors.append(msg)
                self.dead[p] = time.time()
                print(f"[puppy] {p} failed: {msg[:300]}")
        self.last_error = errors[-1] if errors else "no language model configured"
        raise LLMError(self.last_error)


def warm_up(llm: "LLM") -> None:
    """Load the local model into memory before the first question (the first load can take 10-30 s)."""
    if llm.provider == "ollama":
        try:
            llm.complete("Reply with OK.", "OK", max_tokens=2)
        except LLMError:
            pass
