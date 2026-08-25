import unittest

from agent_response import format_tool_response, source_lookup_sources


class AgentResponseTests(unittest.TestCase):
    def test_formats_paper_count(self):
        text = format_tool_response(
            {
                "ok": True,
                "tool": "paper_library",
                "action": "paper_count",
                "paper_count": 15,
            }
        )
        self.assertIn("15", text)

    def test_formats_structured_error_without_internal_details(self):
        text = format_tool_response(
            {
                "ok": False,
                "error": {"code": "paper_not_found", "message": "论文不存在。"},
            }
        )
        self.assertEqual(text, "论文不存在。")

    def test_adapts_source_chunks_for_existing_renderer(self):
        sources = source_lookup_sources(
            {
                "ok": True,
                "tool": "source_lookup",
                "paper_name": "Example.pdf",
                "chunks": [
                    {
                        "chunk_id": "example::page_2::chunk_0",
                        "page_number": 2,
                        "text": "source text",
                    }
                ],
            }
        )
        self.assertEqual(sources[0]["rank"], 1)
        self.assertEqual(sources[0]["file_name"], "Example.pdf")
        self.assertEqual(sources[0]["page_number"], 2)


if __name__ == "__main__":
    unittest.main()
