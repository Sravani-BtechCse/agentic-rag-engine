# Agentic Workflow Patterns and Self-Correction Loops

## Overview
Agentic AI workflows transform standard large language model prompts into dynamic state graphs. Rather than producing a single static completion, agentic architectures enable LLMs to plan, execute domain tools, grade intermediate artifacts, rewrite search queries, and correct errors iteratively.

## Core Design Principles
1. **Dynamic Intent Routing**: The system classifies queries to dispatch tasks to appropriate execution nodes (e.g. Vector RAG vs Code Interpreter vs External Search API).
2. **Relevance Grading & Reflection**: Before passing retrieved document contexts to the synthesis engine, a relevance grader evaluates relevance scores. Low-confidence retrievals trigger reflection nodes that rewrite queries or trigger fallback web search.
3. **Hallucination & Groundedness Checks**: Independent evaluation nodes audit generated answers against context snippets to ensure zero hallucination and high factual fidelity.
4. **Tool Use & Execution**: Agents utilize deterministic calculators, scrapers, and database connections to supply exact computation alongside semantic inference.
