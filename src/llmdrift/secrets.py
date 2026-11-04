"""Unified secrets loader for local (.env) and deployed (st.secrets) environments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def load_env_and_secrets(
    secrets_source: Any = None,
    session_overrides: dict[str, str] | None = None,
    dotenv_path: Path | None = None,
) -> dict[str, bool]:
    """Load API keys into os.environ from .env, Streamlit secrets, and session overrides.

    Returns a dict mapping provider names to boolean availability status.
    Never prints or logs secret values.
    """
    # 1. Load from .env locally (skipped in tests if LLMDRIFT_DOTENV=0)
    if os.environ.get("LLMDRIFT_DOTENV", "1") != "0":
        from dotenv import load_dotenv

        env_file = dotenv_path or (ROOT / ".env")
        if env_file.exists():
            load_dotenv(env_file)

    # 2. Load from Streamlit secrets if available
    source = secrets_source
    if source is None:
        try:
            import streamlit as st

            if hasattr(st, "secrets"):
                source = st.secrets
        except Exception:
            source = None

    if source is not None:
        try:
            for key in ("GEMINI_API_KEY", "GROQ_API_KEY", "OLLAMA_API_KEY"):
                val = getattr(source, "get", lambda k: None)(key)
                if val and isinstance(val, str) and val.strip():
                    os.environ[key] = val.strip()

            # Iterate over any other keys ending with _API_KEY
            items_fn = getattr(source, "items", None)
            if callable(items_fn):
                for k, v in items_fn():
                    if k.endswith("_API_KEY") and isinstance(v, str) and v.strip():
                        os.environ[k] = v.strip()
        except Exception:
            pass

    # 3. Apply session overrides if provided
    if session_overrides:
        for k, v in session_overrides.items():
            if v and isinstance(v, str) and v.strip():
                os.environ[k] = v.strip()

    return {
        "gemini": bool(os.environ.get("GEMINI_API_KEY")),
        "groq": bool(os.environ.get("GROQ_API_KEY")),
        "ollama": bool(os.environ.get("OLLAMA_API_KEY") or os.environ.get("OLLAMA_API_BASE")),
    }
