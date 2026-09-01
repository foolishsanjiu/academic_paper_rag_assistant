"""Tests for PDF discovery and upload-size validation."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

from document_loader import load_pdf_directory
from knowledge_base import list_pdf_files, validate_pdf_bytes


class KnowledgeBaseTests(unittest.TestCase):
    def test_discovers_uppercase_pdf_extensions_cross_platform(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paper_path = Path(directory) / "Paper.PDF"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "searchable text")
            document.save(paper_path)
            document.close()

            self.assertEqual(list_pdf_files(Path(directory)), [paper_path])
            loaded = load_pdf_directory(Path(directory))

        self.assertEqual(len(loaded), 1)
        self.assertIn("searchable text", loaded[0].page_content)

    def test_rejects_files_over_the_configured_upload_limit(self) -> None:
        with patch("knowledge_base.MAX_UPLOAD_SIZE_MB", 0):
            with self.assertRaisesRegex(ValueError, "不能超过"):
                validate_pdf_bytes("paper.pdf", b"%PDF-")


if __name__ == "__main__":
    unittest.main()
