import os
from pydantic import BaseModel, Field

class Settings(BaseModel):
    """System configuration settings for Agentic RAG Framework."""
    app_name: str = "Agentic RAG Engine"
    version: str = "1.0.0"
    debug: bool = Field(default=False)
    
    # Vector Search settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4
    relevance_threshold: float = 0.35
    
    # Workflow Loop settings
    max_reflection_retries: int = 2
    enable_web_fallback: bool = True
    enable_hallucination_check: bool = True
    
    # LLM Settings
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "mock"))
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model_name: str = Field(default_factory=lambda: os.getenv("MODEL_NAME", "gpt-4o-mini"))
    
    # Paths
    vector_store_path: str = os.path.join(os.path.dirname(__file__), "..", "..", "data", "vector_store.json")

settings = Settings()
