import os
import pytest
import tempfile
from src.agentic_rag.vectorstore import LightweightVectorStore

def test_vectorstore_indexing_and_query():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        temp_path = tmp.name

    try:
        vs = LightweightVectorStore(storage_path=temp_path)
        doc_text = "Agentic RAG systems use dynamic intent routing and vector chunking to answer queries."
        chunks_added = vs.add_documents(doc_id="test_doc", content=doc_text)

        assert chunks_added > 0
        assert len(vs.chunks) == chunks_added

        # Query matching document
        results = vs.query("agentic rag dynamic intent routing")
        assert len(results) > 0
        assert results[0].score > 0.0
        assert "Agentic RAG" in results[0].chunk.content

        # Clear store
        vs.clear()
        assert len(vs.chunks) == 0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
