import math

import fastembed
import pytest

from hireflow.core.config import Settings
from hireflow.llm.client import MissingAPIKeyError
from hireflow.rag import embeddings as emb


def use_settings(monkeypatch, **values):
    monkeypatch.setattr(emb, "get_settings", lambda: Settings(_env_file=None, **values))
    emb.get_embeddings.cache_clear()


def test_keyword_hash_embeddings_are_normalised_and_similar():
    e = emb.KeywordHashEmbeddings(dim=64)
    a, b, c = e.embed_documents(["python fastapi", "fastapi python", "excel tableau"])
    assert math.isclose(sum(x * x for x in a), 1.0)  # unit length
    assert a == b  # same words, same vector
    assert a != c
    assert e.embed_query("")[0] == 1.0  # empty text still gets a valid direction


def test_provider_switch(monkeypatch):
    use_settings(monkeypatch, embedding_provider="fake", embedding_dim=8)
    assert isinstance(emb.get_embeddings(), emb.KeywordHashEmbeddings)

    use_settings(monkeypatch, embedding_provider="openai", openai_api_key=None)
    with pytest.raises(MissingAPIKeyError):
        emb.get_embeddings()


def test_local_embeddings_wrapper(monkeypatch):
    class FakeFastEmbed:  # stands in for the real model, so the test downloads nothing
        def __init__(self, model_name):
            self.model_name = model_name

        def embed(self, texts):
            return ([0.5, 0.25] for _ in texts)  # fastembed returns a generator

    monkeypatch.setattr(fastembed, "TextEmbedding", FakeFastEmbed)
    local = emb.LocalEmbeddings("BAAI/bge-small-en-v1.5")

    assert local.embed_documents(["a", "b"]) == [[0.5, 0.25], [0.5, 0.25]]
    assert local.embed_query("a") == [0.5, 0.25]