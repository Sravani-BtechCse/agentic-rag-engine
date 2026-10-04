from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import time

class IntentCategory(str, Enum):
    KNOWLEDGE_RAG = "knowledge_rag"
    CALCULATION = "calculation"
    WEB_SEARCH = "web_search"
    DIRECT_ANSWER = "direct_answer"

class DocumentChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

class SearchResult(BaseModel):
    chunk: DocumentChunk
    score: float

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 4
    enable_web_fallback: Optional[bool] = True
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))

class RelevanceGrade(BaseModel):
    score: float
    is_relevant: bool
    reasoning: str

class HallucinationGrade(BaseModel):
    is_grounded: bool
    score: float
    hallucination_detected: bool
    reasoning: str

class AgentStepTrace(BaseModel):
    step_name: str
    timestamp: float = Field(default_factory=time.time)
    details: Dict[str, Any] = Field(default_factory=dict)

class WorkflowResponse(BaseModel):
    query: str
    intent: IntentCategory
    answer: str
    retrieved_chunks: List[SearchResult] = Field(default_factory=list)
    relevance_grade: Optional[RelevanceGrade] = None
    hallucination_grade: Optional[HallucinationGrade] = None
    reflection_attempts: int = 0
    traces: List[AgentStepTrace] = Field(default_factory=list)
    execution_time_sec: float = 0.0

class IngestRequest(BaseModel):
    doc_id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class IngestResponse(BaseModel):
    doc_id: str
    chunks_created: int
    message: str
