from fastapi import FastAPI, HTTPException, status
from src.agentic_rag.schema import (
    QueryRequest, WorkflowResponse, IngestRequest, IngestResponse
)
from src.agentic_rag.vectorstore import LightweightVectorStore
from src.agentic_rag.workflow import AgenticWorkflowEngine
from src.agentic_rag.config import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Enterprise Multi-Agent RAG Engine with Self-Correction, Query Rewriting, Hallucination Verification, and Web Search Fallback."
)

# Singletons
vectorstore = LightweightVectorStore()
engine = AgenticWorkflowEngine(vectorstore)

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "service": settings.app_name,
        "version": settings.version,
        "indexed_chunks": len(vectorstore.chunks),
        "docs": "/docs"
    }

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "storage": settings.vector_store_path, "chunks": len(vectorstore.chunks)}

@app.post("/ingest", response_model=IngestResponse, tags=["Document Management"])
def ingest_document(req: IngestRequest):
    """Ingest and chunk a document into the vector store."""
    try:
        chunks_count = vectorstore.add_documents(req.doc_id, req.content, req.metadata)
        return IngestResponse(
            doc_id=req.doc_id,
            chunks_created=chunks_count,
            message=f"Successfully indexed document '{req.doc_id}' into {chunks_count} vector chunks."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(e)}"
        )

@app.post("/query", response_model=WorkflowResponse, tags=["Agentic Workflow"])
def run_query(req: QueryRequest):
    """Execute the Agentic Self-Corrective RAG pipeline."""
    try:
        response = engine.execute(req)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow execution failed: {str(e)}"
        )
