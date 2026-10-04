import pytest
import tempfile
import os
from src.agentic_rag.schema import QueryRequest, IntentCategory
from src.agentic_rag.vectorstore import LightweightVectorStore
from src.agentic_rag.workflow import AgenticWorkflowEngine

def test_workflow_execution():
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

        assert resp.query == req.query
        assert resp.intent == IntentCategory.KNOWLEDGE_RAG
        assert len(resp.retrieved_chunks) > 0
        assert resp.relevance_grade is not None
        assert resp.relevance_grade.is_relevant is True
        assert resp.answer is not None
        assert len(resp.traces) > 0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_calculator_intent_routing():
    engine = AgenticWorkflowEngine()
    req = QueryRequest(query="calculate: 45 * 12 + 100")
    resp = engine.execute(req)

    assert resp.intent == IntentCategory.CALCULATION
    assert "640" in resp.answer
