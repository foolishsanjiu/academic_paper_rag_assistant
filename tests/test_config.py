"""Tests for centralized project defaults."""

import unittest

import config


class ProjectConfigurationTests(unittest.TestCase):
    def test_index_paths_share_the_project_root(self) -> None:
        self.assertEqual(
            config.INDEX_MANIFEST_PATH.parent,
            config.CHROMA_DIRECTORY,
        )
        self.assertEqual(
            config.PAPER_DIRECTORY,
            config.PROJECT_ROOT / "data" / "papers",
        )

    def test_chunk_defaults_are_valid(self) -> None:
        self.assertGreater(config.DEFAULT_CHUNK_SIZE, 0)
        self.assertGreaterEqual(config.DEFAULT_CHUNK_OVERLAP, 0)
        self.assertLess(
            config.DEFAULT_CHUNK_OVERLAP,
            config.DEFAULT_CHUNK_SIZE,
        )

    def test_multi_document_defaults_match_experiment(self) -> None:
        self.assertEqual(config.MULTI_DOCUMENT_TOP_K, 8)
        self.assertEqual(config.MULTI_DOCUMENT_CANDIDATE_K, 32)
        self.assertEqual(config.MULTI_DOCUMENT_MAX_CHUNKS_PER_FILE, 3)


if __name__ == "__main__":
    unittest.main()
