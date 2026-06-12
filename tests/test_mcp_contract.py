"""The MCP ``recall`` tool returns this JSON contract — agents depend on it."""

import json

from ragnite.memory.types import ConfidenceSignals, MemoryAnswer
from ragnite.server.mcp import recall_payload

EXPECTED_KEYS = {"mode", "confidence", "suggestion", "context", "tokens", "cached", "signals"}


def test_recall_payload_contract():
    answer = MemoryAnswer(
        query="which db do we use?",
        mode="direct",
        confidence=0.86,
        suggestion="Answer directly.",
        context="- [decision|sim 0.81|3mo] database: Postgres 16",
        tokens=18,
        signals=ConfidenceSignals(top_similarity=0.81, conflict=False),
        cached=True,
    )
    payload = recall_payload(answer)
    assert set(payload) == EXPECTED_KEYS
    assert payload["mode"] == "direct"
    assert payload["cached"] is True
    assert payload["signals"]["top_similarity"] == 0.81
    json.dumps(payload)  # must be JSON-serializable as-is


def test_recall_payload_for_empty_memory():
    answer = MemoryAnswer(
        query="anything", mode="refuse_guess", confidence=0.0, suggestion="Say you don't know."
    )
    payload = recall_payload(answer)
    assert set(payload) == EXPECTED_KEYS
    assert payload["context"] == ""
    assert payload["confidence"] == 0.0
