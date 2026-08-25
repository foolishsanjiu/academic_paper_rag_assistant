"""Tests for SourceLookupTool."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pymupdf

from tools.source_lookup import lookup_source


class FakeVectorStore:
    def __init__(self, rows):
        self.rows = rows

    def get(self, ids=None, where=None, include=None):
        selected = self.rows
        if ids is not None:
            selected = [row for row in selected if row["id"] in ids]
        if where is not None:
            selected = [
                row
                for row in selected
                if all(row["metadata"].get(key) == value for key, value in where.items())
            ]
        return {
            "ids": [row["id"] for row in selected],
            "documents": [row["text"] for row in selected],
            "metadatas": [row["metadata"] for row in selected],
        }


class SourceLookupToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.paper_directory = Path(self.temporary_directory.name) / "papers"
        self.paper_directory.mkdir()
        self.paper_path = self.paper_directory / "Example.pdf"

        document = pymupdf.open()
        document.new_page()
        document.new_page()
        document.save(self.paper_path)
        document.close()

        self.store = FakeVectorStore(
            [
                {
                    "id": "example::page_1::chunk_1",
                    "text": "second chunk",
                    "metadata": {
                        "file_name": "Example.pdf",
                        "page_number": 1,
                        "chunk_index": 1,
                    },
                },
                {
                    "id": "example::page_1::chunk_0",
                    "text": "first chunk",
                    "metadata": {
                        "file_name": "Example.pdf",
                        "page_number": 1,
                        "chunk_index": 0,
                    },
                },
                {
                    "id": "example::page_2::chunk_0",
                    "text": "page two",
                    "metadata": {
                        "file_name": "Example.pdf",
                        "page_number": 2,
                        "chunk_index": 0,
                    },
                },
            ]
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def call(self, **kwargs):
        return lookup_source(
            paper_directory=self.paper_directory,
            vector_store=self.store,
            **kwargs,
        )

    def test_page_lookup_returns_chunks_in_index_order(self) -> None:
        result = self.call(paper_name="Example.pdf", page_number=1)

        self.assertTrue(result["ok"])
        self.assertEqual(result["match_count"], 2)
        self.assertEqual(
            [chunk["text"] for chunk in result["chunks"]],
            ["first chunk", "second chunk"],
        )

    def test_chunk_lookup_returns_exact_chunk(self) -> None:
        result = self.call(
            paper_name="example.PDF",
            chunk_id="example::page_2::chunk_0",
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["chunks"][0]["text"], "page two")

    def test_page_and_chunk_must_match(self) -> None:
        result = self.call(
            paper_name="Example.pdf",
            page_number=1,
            chunk_id="example::page_2::chunk_0",
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "chunk_not_found")

    def test_missing_locator_is_rejected(self) -> None:
        result = self.call(paper_name="Example.pdf")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "missing_locator")

    def test_invalid_and_out_of_range_pages_are_rejected(self) -> None:
        invalid = self.call(paper_name="Example.pdf", page_number=0)
        out_of_range = self.call(
            paper_name="Example.pdf",
            page_number=3,
        )

        self.assertEqual(invalid["error"]["code"], "invalid_page_number")
        self.assertEqual(
            out_of_range["error"]["code"],
            "page_out_of_range",
        )

    def test_missing_paper_and_chunk_are_structured_errors(self) -> None:
        missing_paper = self.call(
            paper_name="Missing.pdf",
            page_number=1,
        )
        missing_chunk = self.call(
            paper_name="Example.pdf",
            chunk_id="missing-chunk",
        )

        self.assertEqual(
            missing_paper["error"]["code"],
            "paper_not_found",
        )
        self.assertEqual(
            missing_chunk["error"]["code"],
            "chunk_not_found",
        )

    def test_path_traversal_is_rejected(self) -> None:
        result = self.call(
            paper_name="../Example.pdf",
            page_number=1,
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "invalid_paper_name")


if __name__ == "__main__":
    unittest.main()
