import time
from typing import Optional, List
from src.agentic_rag.schema import (
    QueryRequest, WorkflowResponse, IntentCategory, SearchResult, AgentStepTrace
)
from src.agentic_rag.vectorstore import LightweightVectorStore
from src.agentic_rag.llm import LLMProvider
from src.agentic_rag.tools import AgentTools
from src.agentic_rag.agent import (
    IntentRouterNode, RelevanceGraderNode, AnswerSynthesizerNode, HallucinationCheckerNode
)
from src.agentic_rag.config import settings

class AgenticWorkflowEngine:
    r"""
    State-Graph Agentic Orchestrator implementing Self-Corrective RAG.
    
    Graph Topology:
    [Query] ---> (Intent Router)
                     ├──> [Calculation] ---> (Calculator Tool) --------------┐
                     ├──> [Direct Answer] ---> (Direct Synthesizer) ----------┼--> [Final Answer]
                     └──> [Knowledge RAG] ---> (Vector Retrieval)            │
                                                      │                      │
                                             (Relevance Grader)              │
                                               /            \                │
                                       [Relevant]         [Not Relevant]     │
                                           │                    │            │
                                           │             (Query Rewriter)    │
                                           │                    │            │
                                           │           (Web Fallback Search) │
                                           │                    │            │
                                           └─────────┬──────────┘            │
                                                     ▼                       │
                                           (Answer Synthesizer)              │
                                                     │                       │
                                           (Hallucination Check)             │
                                                     │                       │
                                                     └───────────────────────┘
    """
    def __init__(self, vectorstore: Optional[LightweightVectorStore] = None):
        self.vectorstore = vectorstore or LightweightVectorStore()
        self.llm = LLMProvider()
        self.tools = AgentTools(self.vectorstore)
        
        # Initialize nodes
        self.router = IntentRouterNode(self.llm)
        self.relevance_grader = RelevanceGraderNode(self.llm)
        self.synthesizer = AnswerSynthesizerNode(self.llm)
        self.hallucination_checker = HallucinationCheckerNode(self.llm)

    def execute(self, request: QueryRequest) -> WorkflowResponse:
        start_time = time.time()
        traces: List[AgentStepTrace] = []
        
        def add_trace(step: str, details: dict):
            traces.append(AgentStepTrace(step_name=step, details=details))

        query = request.query
        add_trace("start_query", {"query": query})

        # Step 1: Intent Routing
        intent = self.router.route(query)
        add_trace("route_intent", {"intent": intent.value})

        # Step 2: Handle Non-RAG Paths
        if intent == IntentCategory.CALCULATION:
            calc_result = self.tools.execute_calculator(query)
            add_trace("execute_tool", {"tool": "calculator", "output": calc_result})
            
            return WorkflowResponse(
                query=query,
                intent=intent,
                answer=f"Calculation result for '{query}': {calc_result}",
                traces=traces,
                execution_time_sec=round(time.time() - start_time, 4)
            )

        if intent == IntentCategory.DIRECT_ANSWER:
            answer = self.llm.generate(query)
            add_trace("direct_answer", {"answer_preview": answer[:100]})
            return WorkflowResponse(
                query=query,
                intent=intent,
                answer=answer,
                traces=traces,
                execution_time_sec=round(time.time() - start_time, 4)
            )

        # Step 3: Vector Knowledge Retrieval
        top_k = request.top_k or settings.top_k
        results = self.tools.retrieve_documents(query, top_k=top_k)
        add_trace("retrieve_vector_chunks", {"retrieved_count": len(results)})

        # Step 4: Relevance Grading & Self-Correction Reflection Loop
        rel_grade = self.relevance_grader.grade(query, results)
        add_trace("grade_relevance", rel_grade.model_dump())

        reflection_attempts = 0
        if not rel_grade.is_relevant and request.enable_web_fallback and settings.enable_web_fallback:
            reflection_attempts += 1
            # Rewrite query & execute fallback search
            rewritten_query = self.tools.rewrite_query(query)
            add_trace("rewrite_query", {"original": query, "rewritten": rewritten_query})

            fallback_results = self.tools.execute_web_search(rewritten_query)
            add_trace("fallback_web_search", {"results_count": len(fallback_results)})

            results = fallback_results
            rel_grade = self.relevance_grader.grade(query, results)

        # Step 5: Answer Synthesis
        synthesized_answer = self.synthesizer.synthesize(query, results, intent)
        add_trace("synthesize_answer", {"answer_length": len(synthesized_answer)})

        # Step 6: Hallucination & Groundedness Verification
        hal_grade = None
        if settings.enable_hallucination_check and results:
            hal_grade = self.hallucination_checker.grade(synthesized_answer, results)
            add_trace("check_hallucination", hal_grade.model_dump())

        execution_time = round(time.time() - start_time, 4)
        add_trace("completed", {"execution_time_sec": execution_time})

        return WorkflowResponse(
            query=query,
            intent=intent,
            answer=synthesized_answer,
            retrieved_chunks=results,
            relevance_grade=rel_grade,
            hallucination_grade=hal_grade,
            reflection_attempts=reflection_attempts,
            traces=traces,
            execution_time_sec=execution_time
        )
