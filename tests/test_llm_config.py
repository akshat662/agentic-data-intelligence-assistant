"""Tests for adia.agents.llm_config."""

import pytest

from adia.agents.llm_config import DEFAULT_MODEL, MissingApiKeyError, load_llm_settings

# A path guaranteed not to exist, so `load_dotenv` never picks up a real developer .env file
# and leaks a real key into these tests.
_NO_DOTENV = "/nonexistent/.env"


@pytest.fixture(autouse=True)
def _clear_openai_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)


class TestMissingApiKey:
    def test_raises_when_unset(self):
        with pytest.raises(MissingApiKeyError):
            load_llm_settings(dotenv_path=_NO_DOTENV)

    def test_raises_when_blank(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "   ")
        with pytest.raises(MissingApiKeyError):
            load_llm_settings(dotenv_path=_NO_DOTENV)


class TestLoadedSettings:
    def test_reads_api_key_and_model_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
        settings = load_llm_settings(dotenv_path=_NO_DOTENV)
        assert settings.api_key == "sk-test-key"
        assert settings.model == "gpt-4o"

    def test_defaults_model_when_unset(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        settings = load_llm_settings(dotenv_path=_NO_DOTENV)
        assert settings.model == DEFAULT_MODEL

    def test_api_key_not_in_repr(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret")
        settings = load_llm_settings(dotenv_path=_NO_DOTENV)
        assert "sk-super-secret" not in repr(settings)
