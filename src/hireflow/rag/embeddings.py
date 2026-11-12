import math
import re
import zlib
from functools import lru_cache


from langchain_core.embeddings import Embeddings
from hireflow.core.config import get_settings

class LocalEmbeddings(Embeddings):

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vec] for vec in self._load().embed(texts)] 

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class KeywordHashEmbeddings(Embeddings):
    """A tiny fake embedder for tests: each word adds 1 to a bucket chosen by hashing it.
    Texts that share words get similar vectors. Deterministic, instant, no download.
    Not smart (no synonyms), so never use it for real screening."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def embed_query(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9+#]+", text.lower()):
            vec[zlib.crc32(word.encode()) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:  # no words at all: an all-zero vector has no direction for cosine distance
            vec[0], norm = 1.0, 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

@lru_cache
def get_embeddings() -> Embeddings:
    s = get_settings()
    if s.embedding_provider == 'fake':
        return KeywordHashEmbeddings(s.embedding_dim)
    if s.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        from hireflow.llm.client import MissingAPIKeyError
        if s.openai_api_key is None:
            raise MissingAPIKeyError("Set OPENAI_API_KEY in .env to use OpenAI embeddings")

        return OpenAIEmbeddings(
            model=s.embedding_model,
            dimensions=s.embedding_dim,
            api_key=s.openai_api_key.get_secret_value(),
        )

    return LocalEmbeddings(s.embedding_model)