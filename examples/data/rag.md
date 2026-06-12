# Retrieval-Augmented Generation

RAG combines an information-retrieval step with text generation: the system
first retrieves passages relevant to the user's question, then a language
model writes an answer grounded in those passages.

## Why hybrid retrieval

Dense embeddings capture meaning but miss exact identifiers; keyword search
(BM25) catches exact terms but misses paraphrases. Fusing both with Reciprocal
Rank Fusion is a robust default.

## Why citations matter

Grounded answers with source citations let users verify claims, which is the
main defense against hallucination in production systems.
