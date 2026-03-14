"""Embedding engine for CLAW.

Wraps sentence-transformers for encode/cosine_similarity and provides
sqlite-vec compatible storage via binary serialization.
"""

from __future__ import annotations

import logging
import os
import struct
from typing import Optional

import numpy as np

from claw.core.config import EmbeddingsConfig
from claw.core.exceptions import ConfigError

logger = logging.getLogger("claw.embeddings")

# Lazy import — sentence-transformers is heavy
_SentenceTransformer = None


def _get_sentence_transformer():
    global _SentenceTransformer
    if _SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer
        _SentenceTransformer = SentenceTransformer
    return _SentenceTransformer


class EmbeddingEngine:
    """Encodes text to vectors and provides similarity search utilities.

    Uses all-MiniLM-L6-v2 (384 dimensions) by default.
    Model is loaded lazily on first encode() call.
    """

    def __init__(self, config: Optional[EmbeddingsConfig] = None):
        self.config = config or EmbeddingsConfig()
        self.model_name = self.config.model
        self.dimension = self.config.dimension
        if self.config.required_model and self.model_name != self.config.required_model:
            raise ConfigError(
                f"Embeddings model '{self.model_name}' rejected; "
                f"required model is '{self.config.required_model}'"
            )
        self._model = None
        self._genai_client = None
        self._uses_gemini_api = self.model_name.startswith("gemini-embedding") or self.model_name.startswith("models/gemini-embedding")

    @property
    def model(self):
        if self._uses_gemini_api:
            # Gemini path does not use sentence-transformers.
            return None
        if self._model is None:
            SentenceTransformer = _get_sentence_transformer()
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded (%dD)", self.dimension)
        return self._model

    def _get_genai_client(self):
        if self._genai_client is None:
            api_key = os.getenv(self.config.api_key_env, "")
            if not api_key:
                raise RuntimeError(
                    f"{self.config.api_key_env} is required for Gemini embeddings model '{self.model_name}'"
                )
            from google import genai

            self._genai_client = genai.Client(api_key=api_key)
            logger.info(
                "Gemini embedding client initialized (model=%s, dimension=%d)",
                self.model_name,
                self.dimension,
            )
        return self._genai_client

    def _normalize_dimension(self, vec: list[float]) -> list[float]:
        if len(vec) == self.dimension:
            return vec
        if len(vec) > self.dimension:
            return vec[:self.dimension]
        if len(vec) < self.dimension:
            return vec + [0.0] * (self.dimension - len(vec))
        return vec

    @staticmethod
    def _extract_values_from_genai_response(resp: object) -> list[list[float]]:
        # SDK response shapes vary by endpoint/version:
        # - resp.embedding.values (single)
        # - resp.embeddings[i].values (batch)
        if hasattr(resp, "embeddings") and getattr(resp, "embeddings"):
            vectors = []
            for emb in getattr(resp, "embeddings"):
                values = getattr(emb, "values", None)
                if values is not None:
                    vectors.append([float(v) for v in values])
            if vectors:
                return vectors

        if hasattr(resp, "embedding") and getattr(resp, "embedding") is not None:
            emb = getattr(resp, "embedding")
            values = getattr(emb, "values", None)
            if values is not None:
                return [[float(v) for v in values]]

        return []

    def _embed_with_gemini(self, texts: list[str]) -> list[list[float]]:
        client = self._get_genai_client()
        from google.genai import types

        embed_cfg = {
            "output_dimensionality": self.dimension,
        }
        if self.config.task_type:
            embed_cfg["task_type"] = self.config.task_type

        try:
            resp = client.models.embed_content(
                model=self.model_name,
                contents=texts if len(texts) > 1 else texts[0],
                config=types.EmbedContentConfig(**embed_cfg),
            )
            vectors = self._extract_values_from_genai_response(resp)
            if not vectors:
                raise RuntimeError("Gemini embeddings response contained no vectors")
            return [self._normalize_dimension(v) for v in vectors]
        except Exception as e:
            raise RuntimeError(f"Gemini embeddings call failed: {e}") from e

    def encode(self, text: str) -> list[float]:
        """Encode a single text string to a vector."""
        if self._uses_gemini_api:
            clipped = text[:12000]
            return self._embed_with_gemini([clipped])[0]
        vec = self.model.encode(text, show_progress_bar=False)
        return vec.tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts to vectors."""
        if self._uses_gemini_api:
            if not texts:
                return []
            clipped = [t[:12000] for t in texts]
            return self._embed_with_gemini(clipped)
        vecs = self.model.encode(texts, show_progress_bar=False, batch_size=32)
        return [v.tolist() for v in vecs]

    @staticmethod
    def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Returns a value between -1 and 1 (1 = identical).
        """
        a = np.array(vec1)
        b = np.array(vec2)
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(dot / norm)

    @staticmethod
    def to_sqlite_vec(vec: list[float]) -> bytes:
        """Convert a float vector to sqlite-vec binary format (little-endian float32 array)."""
        return struct.pack(f"<{len(vec)}f", *vec)

    @staticmethod
    def from_sqlite_vec(data: bytes) -> list[float]:
        """Convert sqlite-vec binary format back to a float vector."""
        count = len(data) // 4
        return list(struct.unpack(f"<{count}f", data))
