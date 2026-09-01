"""Offline tests for LLM request parameter safety."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from config import Settings
from llm_client import LLMClient


class LLMClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = LLMClient(
            Settings(api_key="test-key", model="test-model")
        )

    def test_rejects_invalid_max_tokens_before_api_call(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            self.client.chat("question", max_tokens=0)

    def test_passes_optional_max_tokens_to_request(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="answer"))
            ]
        )
        create = Mock(return_value=response)
        self.client.client.chat.completions.create = create

        answer = self.client.chat(
            "question",
            temperature=0.2,
            max_tokens=800,
            thinking=False,
        )

        self.assertEqual(answer, "answer")
        self.assertEqual(create.call_args.kwargs["max_tokens"], 800)
        self.assertEqual(
            create.call_args.kwargs["extra_body"],
            {"thinking": {"type": "disabled"}},
        )


if __name__ == "__main__":
    unittest.main()
