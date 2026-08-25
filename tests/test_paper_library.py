"""Tests for PaperLibraryTool."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pymupdf

from tools.paper_library import query_paper_library


class PaperLibraryToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.paper_directory = Path(self.temporary_directory.name) / "papers"
        self.paper_directory.mkdir()
        self.manifest_path = (
            Path(self.temporary_directory.name) / "index_manifest.json"
        )

        document = pymupdf.open()
        document.new_page()
        document.save(self.paper_directory / "Example.pdf")
        document.close()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def call(self, action: str, paper_name: str | None = None):
        return query_paper_library(
            action,
            paper_name,
            paper_directory=self.paper_directory,
            manifest_path=self.manifest_path,
        )

    def test_count_and_list_papers(self) -> None:
        count = self.call("paper_count")
        listing = self.call("paper_list")

        self.assertEqual(count["paper_count"], 1)
        self.assertEqual(listing["papers"], ["Example.pdf"])

    def test_exists_is_case_insensitive(self) -> None:
        result = self.call("paper_exists", "example.PDF")

        self.assertTrue(result["ok"])
        self.assertTrue(result["exists"])
        self.assertEqual(result["paper_name"], "Example.pdf")

    def test_missing_paper_name_returns_structured_error(self) -> None:
        result = self.call("paper_info")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "missing_paper_name")

    def test_path_traversal_is_rejected(self) -> None:
        result = self.call("paper_exists", "../Example.pdf")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "invalid_paper_name")

    def test_paper_info_uses_real_pdf_metadata(self) -> None:
        result = self.call("paper_info", "Example.pdf")

        self.assertTrue(result["ok"])
        self.assertEqual(result["paper"]["page_count"], 1)
        self.assertEqual(len(result["paper"]["document_id"]), 16)
        self.assertGreater(result["paper"]["size_bytes"], 0)

    def test_index_status_reports_manifest_and_drift(self) -> None:
        self.manifest_path.write_text(
            json.dumps(
                {
                    "paper_count": 2,
                    "page_count": 10,
                    "chunk_count": 50,
                    "chunk_size": 600,
                    "chunk_overlap": 100,
                    "embedding_model": "test-model",
                    "built_at": "2026-08-25T00:00:00+08:00",
                }
            ),
            encoding="utf-8",
        )

        result = self.call("index_status")

        self.assertTrue(result["indexed"])
        self.assertEqual(result["chunk_count"], 50)
        self.assertFalse(result["paper_set_matches_index"])

    def test_invalid_action_returns_allowed_values(self) -> None:
        result = self.call("delete_everything")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "invalid_action")
        self.assertIn("paper_count", result["error"]["message"])


if __name__ == "__main__":
    unittest.main()
