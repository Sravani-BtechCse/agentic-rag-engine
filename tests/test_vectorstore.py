import os
import unittest
import tempfile
import sys

# Ensure src in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agentic_rag.vectorstore import LightweightVectorStore

class TestVectorStore(unittest.TestCase):
    def test_indexing_and_query(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            temp_path = tmp.name

        try:
            vs = LightweightVectorStore(storage_path=temp_path)
            doc_text = "Agentic RAG systems use dynamic intent routing and vector chunking to answer queries."
            chunks_added = vs.add_documents(doc_id="test_doc", content=doc_text)

            self.assertGreater(chunks_added, 0)
            self.assertEqual(len(vs.chunks), chunks_added)

            # Query matching document
            results = vs.query("agentic rag dynamic intent routing")
            self.assertGreater(len(results), 0)
            self.assertGreater(results[0].score, 0.0)
            self.assertIn("Agentic RAG", results[0].chunk.content)

            # Clear store
            vs.clear()
            self.assertEqual(len(vs.chunks), 0)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    unittest.main()
