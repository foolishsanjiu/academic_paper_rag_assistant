import json
from pathlib import Path
import unittest

from agent_router import Intent, dispatch_route, route_query
from retriever import RetrievalStrategy


CASES_PATH = Path(__file__).with_name("router_cases.json")


class FakeLLM:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def chat_json(self, _message):
        if self.error:
            raise self.error
        return self.response


class AgentRouterTests(unittest.TestCase):
    @staticmethod
    def paper_names():
        return ["Example.pdf"]

    def test_rule_router_dataset(self):
        cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(cases), 40)

        for case in cases:
            with self.subTest(query=case["query"]):
                decision = route_query(
                    case["query"],
                    paper_names_provider=self.paper_names,
                )
                self.assertEqual(decision.intent.value, case["intent"])
                if "strategy" in case:
                    self.assertEqual(
                        decision.retrieval_strategy.value,
                        case["strategy"],
                    )
                if "action" in case:
                    self.assertEqual(decision.tool_args["action"], case["action"])

    def test_llm_json_is_validated_and_used(self):
        llm = FakeLLM(
            '```json\n{"intent":"paper_qa",'
            '"retrieval_strategy":"multi_document"}\n```'
        )
        decision = route_query(
            "summarize the papers",
            llm,
            paper_names_provider=self.paper_names,
        )
        self.assertEqual(decision.intent, Intent.PAPER_QA)
        self.assertEqual(decision.retrieval_strategy.value, "multi_document")
        self.assertEqual(decision.route_source, "llm_json")

    def test_arbitrary_model_tool_name_is_ignored(self):
        llm = FakeLLM(
            '{"intent":"knowledge_base_query",'
            '"action":"paper_count","tool_name":"delete_files"}'
        )
        decision = route_query(
            "count papers",
            llm,
            paper_names_provider=self.paper_names,
        )
        self.assertEqual(decision.tool_name, "paper_library")
        self.assertEqual(decision.tool_args, {"action": "paper_count"})

    def test_invalid_json_falls_back_to_rules(self):
        decision = route_query(
            "比较两篇论文",
            FakeLLM("not json"),
            paper_names_provider=self.paper_names,
        )
        self.assertEqual(decision.intent, Intent.PAPER_QA)
        self.assertEqual(decision.retrieval_strategy.value, "multi_document")
        self.assertEqual(decision.route_source, "rule_fallback")

    def test_api_error_falls_back_to_rules(self):
        decision = route_query(
            "当前知识库有多少篇论文？",
            FakeLLM(error=RuntimeError("offline")),
            paper_names_provider=self.paper_names,
        )
        self.assertEqual(decision.intent, Intent.KNOWLEDGE_BASE_QUERY)
        self.assertEqual(decision.tool_args["action"], "paper_count")

    def test_empty_query_is_rejected(self):
        with self.assertRaises(ValueError):
            route_query("   ", paper_names_provider=self.paper_names)

    def test_dispatch_uses_selected_rag_strategy(self):
        decision = route_query(
            "比较两篇论文",
            paper_names_provider=self.paper_names,
        )
        result = dispatch_route(
            decision,
            "比较两篇论文",
            rag_handlers={
                RetrievalStrategy.FOCUSED: lambda query: f"focused:{query}",
                RetrievalStrategy.MULTI_DOCUMENT: lambda query: f"multi:{query}",
            },
        )
        self.assertEqual(result, "multi:比较两篇论文")

    def test_dispatch_calls_only_validated_library_tool(self):
        decision = route_query(
            "有多少篇论文",
            paper_names_provider=self.paper_names,
        )
        result = dispatch_route(
            decision,
            "有多少篇论文",
            paper_library_tool=lambda **kwargs: kwargs,
        )
        self.assertEqual(result, {"action": "paper_count"})

    def test_source_dispatch_requires_paper_name(self):
        decision = route_query(
            "显示第 2 页原文",
            paper_names_provider=self.paper_names,
        )
        result = dispatch_route(decision, "显示第 2 页原文")
        self.assertEqual(result["error"]["code"], "missing_paper_name")


if __name__ == "__main__":
    unittest.main()
