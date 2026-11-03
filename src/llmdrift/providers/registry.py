"""Build providers from config and check whether they can be used."""

from __future__ import annotations

import os
import urllib.error
import urllib.request

from llmdrift.config import ModelConfig
from llmdrift.datasets.schema import GoldenItem
from llmdrift.prompting import render_prompt
from llmdrift.providers.base import Provider
from llmdrift.providers.litellm_provider import LiteLLMProvider
from llmdrift.providers.mock import MockProvider


def check_available(cfg: ModelConfig, timeout_s: float = 2.0) -> tuple[bool, str]:
    """Return ``(usable, reason)`` without making a billable call."""
    if not cfg.enabled:
        return False, "disabled in config"
    if cfg.provider == "mock":
        return True, "ok"
    if cfg.api_key_env and not os.environ.get(cfg.api_key_env):
        return False, f"{cfg.api_key_env} not set"
    if cfg.provider == "ollama":
        base = os.environ.get("OLLAMA_API_BASE", cfg.api_base or "http://localhost:11434")
        try:
            with urllib.request.urlopen(f"{base.rstrip('/')}/api/tags", timeout=timeout_s):
                pass
        except (urllib.error.URLError, OSError) as exc:
            return False, f"Ollama not reachable at {base} ({exc.__class__.__name__})"
    return True, "ok"


def build_provider(cfg: ModelConfig, dataset: list[GoldenItem]) -> Provider:
    """Create the provider object for one model entry."""
    if cfg.provider == "mock":
        return MockProvider(cfg, dataset, render_prompt)
    return LiteLLMProvider(cfg)
