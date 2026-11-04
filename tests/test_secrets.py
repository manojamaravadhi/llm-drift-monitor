from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llmdrift.secrets import load_env_and_secrets


class MockSecrets:
    """Mock Streamlit secrets object supporting .get() and .items()."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def items(self) -> Any:
        return self._data.items()


def test_load_env_and_secrets_from_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("LLMDRIFT_DOTENV", "0")

    mock_sec = MockSecrets({"GEMINI_API_KEY": "sec_gem_123", "GROQ_API_KEY": "sec_groq_456"})
    status = load_env_and_secrets(secrets_source=mock_sec)

    assert status["gemini"] is True
    assert status["groq"] is True
    import os

    assert os.environ.get("GEMINI_API_KEY") == "sec_gem_123"
    assert os.environ.get("GROQ_API_KEY") == "sec_groq_456"


def test_load_env_and_secrets_with_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("LLMDRIFT_DOTENV", "0")

    status = load_env_and_secrets(
        secrets_source=None,
        session_overrides={"GEMINI_API_KEY": "override_key"},
    )
    assert status["gemini"] is True
    import os

    assert os.environ.get("GEMINI_API_KEY") == "override_key"


def test_load_env_and_secrets_from_dotenv_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("LLMDRIFT_DOTENV", "1")

    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=file_key_789\n")

    status = load_env_and_secrets(secrets_source=None, dotenv_path=env_file)
    assert status["gemini"] is True
    import os

    assert os.environ.get("GEMINI_API_KEY") == "file_key_789"
