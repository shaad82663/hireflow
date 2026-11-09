from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from hireflow.core.config import get_settings


class MissingAPIKeyError(RuntimeError):
    """Raised at startup of the first LLM call when the chosen provider has no key."""


@lru_cache
def get_chat_model() -> BaseChatModel:
    s = get_settings()
    key = s.anthropic_api_key if s.llm_provider == "anthropic" else s.openai_api_key
    if key is None:
        raise MissingAPIKeyError(f"Set {s.llm_provider.upper()}_API_KEY in your .env file")

    # Settings reads .env into this object only, not into os.environ,
    # so we pass the key explicitly instead of relying on LangChain finding it.
    kwargs: dict = {"api_key": key.get_secret_value()}
    if s.llm_temperature is not None:
        kwargs["temperature"] = s.llm_temperature
    return init_chat_model(s.llm_model, model_provider=s.llm_provider, **kwargs)