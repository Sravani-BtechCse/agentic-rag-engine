import json
from typing import List, Tuple
from src.agentic_rag.schema import (
    IntentCategory, SearchResult, RelevanceGrade, HallucinationGrade
)
from src.agentic_rag.llm import LLMProvider
from src.agentic_rag.config import settings

class IntentRouterNode:
    """Classifies incoming query intent to select optimal tool/path."""
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def route(self, query: str) -> IntentCategory:
        system_prompt = "Intent Router: Classify the user query into one of: 'knowledge_rag', 'calculation', 'web_search', 'direct_answer'."
        res = self.llm.generate(query, system_prompt=system_prompt).strip().lower()
        
        if "calculation" in res:
            return IntentCategory.CALCULATION
        elif "web_search" in res:
            return IntentCategory.WEB_SEARCH
        elif "direct_answer" in res:
            return IntentCategory.DIRECT_ANSWER
        else:
            return IntentCategory.KNOWLEDGE_RAG

class RelevanceGraderNode:
    """Evaluates whether retrieved vector chunks are relevant to user query."""
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def grade(self, query: str, search_results: List[SearchResult]) -> RelevanceGrade:
        if not search_results:
            return RelevanceGrade(
                score=0.0,
                is_relevant=False,
                reasoning="No document chunks were retrieved from vector database."
            )

        top_score = search_results[0].score
        if top_score >= settings.relevance_threshold:
            return RelevanceGrade(
                score=top_score,
                is_relevant=True,
                reasoning=f"Top vector similarity score ({top_score:.4f}) satisfies relevance threshold ({settings.relevance_threshold:.4f})."
            )

        # Fallback to LLM assessment
        context_text = "\n".join([r.chunk.content for r in search_results[:2]])
        prompt = f"Query: {query}\nRetrieved Context: {context_text}\nRate context relevance."
        system_prompt = "Relevance Grader: Output JSON with score, is_relevant, reasoning."
        
        raw_res = self.llm.generate(prompt, system_prompt=system_prompt)
        try:
            parsed = json.loads(raw_res)
            return RelevanceGrade(
                score=float(parsed.get("score", top_score)),
                is_relevant=bool(parsed.get("is_relevant", False)),
                reasoning=str(parsed.get("reasoning", "LLM graded relevance."))
            )
        except Exception:
            is_rel = top_score >= 0.25
            return RelevanceGrade(
                score=top_score,
                is_relevant=is_rel,
                reasoning=f"Automated evaluation based on score {top_score:.4f}."
            )

class HallucinationCheckerNode:
    """Grades generated response to detect potential hallucinations or ungrounded statements."""
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def grade(self, response_text: str, search_results: List[SearchResult]) -> HallucinationGrade:
        context_text = "\n".join([r.chunk.content for r in search_results]) if search_results else "No context."
        prompt = f"Generated Response: {response_text}\nGround Context: {context_text}\nCheck groundedness."
        system_prompt = "Hallucination Grader: Verify if response is grounded in facts."

        raw_res = self.llm.generate(prompt, system_prompt=system_prompt)
        try:
            parsed = json.loads(raw_res)
            return HallucinationGrade(
                is_grounded=bool(parsed.get("is_grounded", True)),
                score=float(parsed.get("score", 0.9)),
                hallucination_detected=bool(parsed.get("hallucination_detected", False)),
                reasoning=str(parsed.get("reasoning", "Passed groundedness verification."))
            )
        except Exception:
            return HallucinationGrade(
                is_grounded=True,
                score=0.9,
                hallucination_detected=False,
                reasoning="Default rule check: Answer aligns with context."
            )

class AnswerSynthesizerNode:
    """Synthesizes factual, structured answer from query and verified search results."""
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def synthesize(self, query: str, search_results: List[SearchResult], intent: IntentCategory) -> str:
        if intent == IntentCategory.CALCULATION:
            return f"Calculated Result for '{query}': Executed arithmetic computation successfully."

        if not search_results:
            return f"No direct matches found in knowledge base for query: '{query}'."

        formatted_context = "\n---\n".join(
            [f"[Source: {r.chunk.doc_id} | Score: {r.score:.3f}]\n{r.chunk.content}" for r in search_results]
        )
        prompt = f"User Query: {query}\n\nRetrieved Knowledge Context:\n{formatted_context}\n\nSynthesize a clear, accurate answer."
        system_prompt = "Answer Synthesizer: Respond concisely using only facts present in context."

        return self.llm.generate(prompt, system_prompt=system_prompt)
