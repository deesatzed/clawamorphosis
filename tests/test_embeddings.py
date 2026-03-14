"""Tests for CLAW embedding engine."""

import struct

from claw.core.config import EmbeddingsConfig
from claw.core.exceptions import ConfigError
from claw.db.embeddings import EmbeddingEngine


class TestEmbeddingEngine:
    def test_cosine_similarity_identical(self):
        sim = EmbeddingEngine.cosine_similarity([1, 0, 0], [1, 0, 0])
        assert abs(sim - 1.0) < 0.001

    def test_cosine_similarity_orthogonal(self):
        sim = EmbeddingEngine.cosine_similarity([1, 0, 0], [0, 1, 0])
        assert abs(sim) < 0.001

    def test_cosine_similarity_opposite(self):
        sim = EmbeddingEngine.cosine_similarity([1, 0], [-1, 0])
        assert abs(sim - (-1.0)) < 0.001

    def test_cosine_similarity_zero_vector(self):
        sim = EmbeddingEngine.cosine_similarity([0, 0, 0], [1, 0, 0])
        assert sim == 0.0


class TestSqliteVecConversion:
    def test_roundtrip(self):
        vec = [0.1, 0.2, 0.3, 0.4, 0.5]
        packed = EmbeddingEngine.to_sqlite_vec(vec)
        unpacked = EmbeddingEngine.from_sqlite_vec(packed)
        assert len(unpacked) == 5
        for a, b in zip(vec, unpacked):
            assert abs(a - b) < 0.0001

    def test_384_dim_vector(self):
        vec = [float(i) / 384 for i in range(384)]
        packed = EmbeddingEngine.to_sqlite_vec(vec)
        assert len(packed) == 384 * 4  # 4 bytes per float32
        unpacked = EmbeddingEngine.from_sqlite_vec(packed)
        assert len(unpacked) == 384

    def test_empty_vector(self):
        packed = EmbeddingEngine.to_sqlite_vec([])
        assert len(packed) == 0
        unpacked = EmbeddingEngine.from_sqlite_vec(packed)
        assert unpacked == []


class TestGeminiEmbeddingPath:
    def test_gemini_encode_uses_api_path(self, monkeypatch):
        cfg = EmbeddingsConfig(model="gemini-embedding-001", dimension=4)
        engine = EmbeddingEngine(cfg)

        monkeypatch.setattr(
            engine,
            "_embed_with_gemini",
            lambda texts: [[1.0, 2.0, 3.0, 4.0]],
        )

        vec = engine.encode("hello")
        assert vec == [1.0, 2.0, 3.0, 4.0]

    def test_gemini_encode_batch_uses_api_path(self, monkeypatch):
        cfg = EmbeddingsConfig(model="gemini-embedding-001", dimension=3)
        engine = EmbeddingEngine(cfg)

        monkeypatch.setattr(
            engine,
            "_embed_with_gemini",
            lambda texts: [[float(i), 0.0, 1.0] for i, _ in enumerate(texts)],
        )

        vecs = engine.encode_batch(["a", "b", "c"])
        assert len(vecs) == 3
        assert vecs[2][0] == 2.0

    def test_normalize_dimension_pads_or_truncates(self):
        cfg = EmbeddingsConfig(model="gemini-embedding-001", dimension=4)
        engine = EmbeddingEngine(cfg)
        assert engine._normalize_dimension([1.0, 2.0]) == [1.0, 2.0, 0.0, 0.0]
        assert engine._normalize_dimension([1.0, 2.0, 3.0, 4.0, 5.0]) == [1.0, 2.0, 3.0, 4.0]

    def test_extract_values_from_response_shapes(self):
        class Emb:
            def __init__(self, values):
                self.values = values

        class SingleResp:
            def __init__(self):
                self.embedding = Emb([0.1, 0.2])

        class BatchResp:
            def __init__(self):
                self.embeddings = [Emb([1.0, 2.0]), Emb([3.0, 4.0])]

        single = EmbeddingEngine._extract_values_from_genai_response(SingleResp())
        batch = EmbeddingEngine._extract_values_from_genai_response(BatchResp())

        assert single == [[0.1, 0.2]]
        assert batch == [[1.0, 2.0], [3.0, 4.0]]

    def test_required_model_rejects_non_matching_model(self):
        cfg = EmbeddingsConfig(
            model="gemini-embedding-001",
            required_model="gemini-embedding-2-preview",
            dimension=4,
        )
        try:
            EmbeddingEngine(cfg)
            assert False, "expected ConfigError for model mismatch"
        except ConfigError:
            pass
