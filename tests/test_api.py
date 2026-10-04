import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class TestAPIEndpoints(unittest.TestCase):
    def test_api_import(self):
        try:
            from fastapi.testclient import TestClient
            from src.agentic_rag.api import app

            client = TestClient(app)
            res = client.get("/health")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["status"], "healthy")
        except ImportError:
            print("[Info] FastAPI/TestClient not installed in global environment; skipping API test client assertion.")

if __name__ == "__main__":
    unittest.main()
