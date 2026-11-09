import pytest

from hireflow.core.config import Settings
from hireflow.llm import client
from hireflow.llm.client import MissingAPIKeyError


def use_settings(monkeypatch, **values):
    """Make get_chat_model() see exactly these settings.
    _env_file=None ignores your real .env, so tests never use your real API key."""
    monkeypatch.setattr(client, "get_settings", lambda: Settings(_env_file=None, **values))
    client.get_chat_model.cache_clear()  # it has @lru_cache, so forget the last model


@pytest.mark.parametrize(
    ("provider", "key_field", "class_name"),
    [
        ("anthropic", "anthropic_api_key", "ChatAnthropic"),
        ("openai", "openai_api_key", "ChatOpenAI"),
    ],
)
def test_builds_the_right_client(monkeypatch, provider, key_field, class_name):
    use_settings(monkeypatch, llm_provider=provider, llm_model="some-model", **{key_field: "k"})

    model = client.get_chat_model()  # builds the client object; no API call happens

    assert type(model).__name__ == class_name
    assert getattr(model, "temperature") == 0.0


def test_temperature_none_means_model_default(monkeypatch):
    use_settings(
        monkeypatch, llm_provider="openai", llm_model="m", openai_api_key="k", llm_temperature=None
    )
    assert getattr(client.get_chat_model(), "temperature") is None


def test_missing_key_gives_clear_error(monkeypatch):
    use_settings(monkeypatch, llm_provider="anthropic", llm_model="m", anthropic_api_key=None)

    with pytest.raises(MissingAPIKeyError, match="ANTHROPIC_API_KEY"):
        client.get_chat_model()