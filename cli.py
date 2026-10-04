import sys
import os
import argparse
import json
import uvicorn

# Ensure UTF-8 output encoding for console
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.agentic_rag.schema import QueryRequest, IngestRequest
from src.agentic_rag.vectorstore import LightweightVectorStore
from src.agentic_rag.workflow import AgenticWorkflowEngine

def seed_sample_data(vectorstore: LightweightVectorStore):
    """Seed sample docs from data/sample_docs into vector store."""
    docs_dir = os.path.join(os.path.dirname(__file__), "data", "sample_docs")
    if not os.path.exists(docs_dir):
        print(f"Sample docs directory not found at {docs_dir}")
        return

    print("[*] Seeding sample knowledge base into vector database...")
    total_chunks = 0
    for fname in os.listdir(docs_dir):
        if fname.endswith(".md") or fname.endswith(".txt"):
            fpath = os.path.join(docs_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            chunks = vectorstore.add_documents(doc_id=fname, content=content, metadata={"source": fname})
            total_chunks += chunks
            print(f"  |- Indexed '{fname}' -> {chunks} chunks")
    print(f"[OK] Seeding complete. Total indexed chunks: {len(vectorstore.chunks)}\n")

def main():
    parser = argparse.ArgumentParser(description="Agentic RAG Engine CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Seed command
    parser_seed = subparsers.add_parser("seed", help="Seed sample markdown documents into vector store")

    # Ingest command
    parser_ingest = subparsers.add_parser("ingest", help="Ingest a text file or text string into vector store")
    parser_ingest.add_argument("--doc-id", required=True, help="Unique document ID")
    parser_ingest.add_argument("--file", help="Path to text file")
    parser_ingest.add_argument("--text", help="Raw text string")

    # Query command
    parser_query = subparsers.add_parser("query", help="Run an Agentic query through the pipeline")
    parser_query.add_argument("query_text", help="The user query to run")
    parser_query.add_argument("--top-k", type=int, default=4, help="Top-K vector results")
    parser_query.add_argument("--json", action="store_true", help="Output full JSON response object")

    # Serve command
    parser_serve = subparsers.add_parser("serve", help="Launch FastAPI REST server")
    parser_serve.add_argument("--host", default="127.0.0.1", help="Host address")
    parser_serve.add_argument("--port", type=int, default=8000, help="Port number")

    args = parser.parse_args()

    vectorstore = LightweightVectorStore()
    engine = AgenticWorkflowEngine(vectorstore)

    if args.command == "seed":
        seed_sample_data(vectorstore)

    elif args.command == "ingest":
        content = ""
        if args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                content = f.read()
        elif args.text:
            content = args.text
        else:
            print("Error: Must specify --file or --text")
            sys.exit(1)

        chunks = vectorstore.add_documents(args.doc_id, content)
        print(f"[OK] Successfully ingested '{args.doc_id}'. Created {chunks} chunks.")

    elif args.command == "query":
        # Auto-seed if empty
        if not vectorstore.chunks:
            seed_sample_data(vectorstore)

        request = QueryRequest(query=args.query_text, top_k=args.top_k)
        response = engine.execute(request)

        if args.json:
            print(json.dumps(response.model_dump(), indent=2))
        else:
            print("=" * 60)
            print(" AGENTIC RAG ENGINE RESULT")
            print("=" * 60)
            print(f"Query           : {response.query}")
            print(f"Detected Intent : {response.intent.value}")
            print(f"Execution Time  : {response.execution_time_sec}s")
            print(f"Reflections     : {response.reflection_attempts}")
            
            if response.relevance_grade:
                print(f"Relevance Score : {response.relevance_grade.score:.4f} (Relevant: {response.relevance_grade.is_relevant})")
            
            if response.hallucination_grade:
                print(f"Groundedness    : {response.hallucination_grade.score:.2f} (Grounded: {response.hallucination_grade.is_grounded})")

            print("\n--- SYNTHESIZED ANSWER ---")
            print(response.answer)
            print("\n--- STEP TRACES ---")
            for t in response.traces:
                print(f"  * {t.step_name}: {json.dumps(t.details)}")
            print("=" * 60)

    elif args.command == "serve":
        print(f"Starting Agentic RAG Engine API server at http://{args.host}:{args.port}")
        uvicorn.run("src.agentic_rag.api:app", host=args.host, port=args.port, reload=True)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
