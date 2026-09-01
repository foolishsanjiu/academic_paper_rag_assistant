"""Tests for runtime index-manifest consistency checks."""

import unittest

from index_manifest import validate_index_manifest


class IndexManifestTests(unittest.TestCase):
    def test_accepts_matching_count_and_embedding_model(self) -> None:
        validate_index_manifest(
            vector_count=10,
            manifest={
                "chunk_count": 10,
                "embedding_model": "BAAI/bge-m3",
            },
            embedding_model="BAAI/bge-m3",
        )

    def test_rejects_count_and_embedding_model_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "Chroma=9"):
            validate_index_manifest(
                vector_count=9,
                manifest={"chunk_count": 10},
            )

        with self.assertRaisesRegex(ValueError, "Embedding"):
            validate_index_manifest(
                vector_count=10,
                manifest={
                    "chunk_count": 10,
                    "embedding_model": "old-model",
                },
                embedding_model="BAAI/bge-m3",
            )

    def test_rejects_invalid_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "vector_count"):
            validate_index_manifest(
                vector_count=True,
                manifest={"chunk_count": 1},
            )
        with self.assertRaisesRegex(ValueError, "chunk_count"):
            validate_index_manifest(
                vector_count=1,
                manifest={"chunk_count": "1"},
            )


if __name__ == "__main__":
    unittest.main()
