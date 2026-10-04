import os
import json
import math
import re
from typing import List, Dict, Tuple, Optional
import numpy as np
from src.agentic_rag.schema import DocumentChunk, SearchResult
from src.agentic_rag.config import settings

class LightweightVectorStore:
    """
    Self-contained, zero-dependency Vector Store utilizing TF-IDF and Cosine Similarity.
    Allows local persistence, sub-millisecond retrieval, chunking, and similarity scoring.
    """
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or settings.vector_store_path
        self.chunks: List[DocumentChunk] = []
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.matrix: Optional[np.ndarray] = None
        self._load_if_exists()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize and normalize input text into words."""
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Filter stop words (minimal set for clean TF-IDF)
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on", "at", "to", "for", "of", "with", "by"}
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def chunk_text(self, doc_id: str, content: str, metadata: dict = None) -> List[DocumentChunk]:
        """Split text content into overlapping semantic chunks."""
        words = content.split()
        chunk_size = settings.chunk_size // 5  # approximate words per chunk
        overlap = settings.chunk_overlap // 5
        
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            if chunk_text.strip():
                chunk = DocumentChunk(
                    doc_id=doc_id,
                    content=chunk_text,
                    metadata=metadata or {"start_idx": start, "end_idx": end}
                )
                chunks.append(chunk)
            if end >= len(words):
                break
            start += (chunk_size - overlap)
        return chunks

    def add_documents(self, doc_id: str, content: str, metadata: dict = None) -> int:
        """Chunk, vectorize, and store new document content."""
        new_chunks = self.chunk_text(doc_id, content, metadata)
        self.chunks.extend(new_chunks)
        self._rebuild_index()
        self._save()
        return len(new_chunks)

    def _rebuild_index(self):
        """Recompute TF-IDF vocabulary and matrix across all stored chunks."""
        if not self.chunks:
            self.vocab = {}
            self.idf = np.array([])
            self.matrix = None
            return

        # 1. Build Vocabulary
        doc_tokens = [self.tokenize(c.content) for c in self.chunks]
        all_words = set(w for tokens in doc_tokens for w in tokens)
        self.vocab = {w: i for i, w in enumerate(sorted(all_words))}
        vocab_size = len(self.vocab)

        if vocab_size == 0:
            return

        num_docs = len(self.chunks)
        tf_matrix = np.zeros((num_docs, vocab_size), dtype=np.float32)
        doc_counts = np.zeros(vocab_size, dtype=np.int32)

        for i, tokens in enumerate(doc_tokens):
            if not tokens:
                continue
            counts: Dict[int, int] = {}
            for t in tokens:
                if t in self.vocab:
                    idx = self.vocab[t]
                    counts[idx] = counts.get(idx, 0) + 1
            
            total = len(tokens)
            for idx, count in counts.items():
                tf_matrix[i, idx] = count / total
                doc_counts[idx] += 1

        # IDF formula: log((1 + N) / (1 + df)) + 1
        self.idf = np.log((1 + num_docs) / (1 + doc_counts)) + 1.0
        
        # TF-IDF matrix
        tfidf = tf_matrix * self.idf
        # L2 normalization for vector cosine similarity
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = tfidf / norms

    def query(self, query_text: str, top_k: int = 4) -> List[SearchResult]:
        """Vector search using cosine similarity against query vector."""
        if not self.chunks or self.matrix is None or len(self.vocab) == 0:
            return []

        tokens = self.tokenize(query_text)
        if not tokens:
            return []

        vocab_size = len(self.vocab)
        q_tf = np.zeros(vocab_size, dtype=np.float32)
        
        counts: Dict[int, int] = {}
        for t in tokens:
            if t in self.vocab:
                idx = self.vocab[t]
                counts[idx] = counts.get(idx, 0) + 1

        total = len(tokens)
        for idx, count in counts.items():
            q_tf[idx] = count / total

        q_tfidf = q_tf * self.idf
        norm = np.linalg.norm(q_tfidf)
        if norm > 0:
            q_vector = q_tfidf / norm
        else:
            q_vector = q_tfidf

        # Dot product of normalized vectors yields cosine similarity
        scores = np.dot(self.matrix, q_vector)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.0:
                results.append(SearchResult(chunk=self.chunks[idx], score=round(score, 4)))

        return results

    def _save(self):
        """Save vector store state to file."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        data = {
            "chunks": [c.model_dump() for c in self.chunks]
        }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load_if_exists(self):
        """Load vector store state if existing store file found."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.chunks = [DocumentChunk(**c) for c in data.get("chunks", [])]
                self._rebuild_index()
            except Exception as e:
                print(f"Warning: Failed to load vector store from {self.storage_path}: {e}")
                self.chunks = []

    def clear(self):
        """Wipe store."""
        self.chunks = []
        self._rebuild_index()
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)
