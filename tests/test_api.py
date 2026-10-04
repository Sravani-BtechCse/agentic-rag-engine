from fastapi.testclient import TestClient
from src.agentic_rag.api import app

client = TestClient(app)

def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"

def test_ingest_and_query_endpoints():
    # Ingest document
    ingest_payload = {
        "doc_id": "doc_api_test",
        "content": "Self-corrective RAG engines rewrite queries when document relevance scores fall below thresholds.",
        "metadata": {"source": "unit_test"}
    }
    ingest_res = client.post("/ingest", json=ingest_payload)
    assert ingest_res.status_code == 200
    assert ingest_res.json()["chunks_created"] > 0

    # Query document
    query_payload = {
        "query": "Why do RAG engines rewrite queries?",
        "top_k": 2
    }
    query_res = client.post("/query", json=query_payload)
    assert query_res.status_code == 200
    data = query_res.json()
    assert data["query"] == query_payload["query"]
    assert len(data["retrieved_chunks"]) > 0
