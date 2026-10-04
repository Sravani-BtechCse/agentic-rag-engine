import os
import sys
import unittest
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agentic_rag.schema import QueryRequest, IntentCategory
from src.agentic_rag.vectorstore import LightweightVectorStore
from src.agentic_rag.workflow import AgenticWorkflowEngine

class TestWorkflowEngine(unittest.TestCase):
    def test_workflow_execution(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            temp_path = tmp.name

        try:
            vs = LightweightVectorStore(storage_path=temp_path)
            vs.add_documents(
                doc_id="ai_doc",
                content="Decision Optimization combines mathematical programming and artificial intelligence to solve operational scheduling."
            )

            engine = AgenticWorkflowEngine(vectorstore=vs)
            req = QueryRequest(query="What is Decision Optimization in operational scheduling?")
            resp = engine.execute(req)

            self.assertEqual(resp.query, req.query)
            self.assertEqual(resp.intent, IntentCategory.KNOWLEDGE_RAG)
            self.assertGreater(len(resp.retrieved_chunks), 0)
            self.assertIsNotNone(resp.relevance_grade)
            self.assertTrue(resp.relevance_grade.is_relevant)
            self.assertIsNotNone(resp.answer)
            self.assertGreater(len(resp.traces), 0)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_calculator_intent_routing(self):
        engine = AgenticWorkflowEngine()
        req = QueryRequest(query="calculate: 45 * 12 + 100")
        resp = engine.execute(req)

        self.assertEqual(resp.intent, IntentCategory.CALCULATION)
        self.assertIn("640", resp.answer)

if __name__ == "__main__":
    unittest.main()
