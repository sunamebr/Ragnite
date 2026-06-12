# GitHub repository metadata (apply in Settings, or via `gh`)

GitHub repo description/topics can't live in the repo itself — apply these
once in **Settings → General** (description/website) and the ⚙️ next to
"About" (topics), or run the `gh` commands below.

## Description (≤ 350 chars)

```
Confidence-Aware RAG Memory Engine for LLMs and coding agents — typed memory (facts, decisions, episodes, code), confidence scoring with answer modes, token-budgeted context packing, semantic caching, hybrid retrieval, and an MCP server. Stop agents from re-analyzing the same project every session.
```

## Topics

```
rag retrieval-augmented-generation llm agent-memory vector-search mcp
mcp-server claude anthropic semantic-cache embeddings coding-agents
ai-agents python knowledge-base bm25 hybrid-search
```

## Website

`https://github.com/sunamebr/Ragnite#readme` (until a docs site exists)

## Social preview text (1280×640 card)

> **Ragnite**
> Memory, context and confidence for LLM agents.
> recall → `direct (0.86)` → answer without re-reading the repo.

## Apply via GitHub CLI

```sh
winget install GitHub.cli   # if gh is missing (Windows)
gh auth login

gh repo edit sunamebr/Ragnite \
  --description "Confidence-Aware RAG Memory Engine for LLMs and coding agents — typed memory (facts, decisions, episodes, code), confidence scoring with answer modes, context packing, semantic caching, hybrid retrieval, and an MCP server." \
  --add-topic rag --add-topic retrieval-augmented-generation --add-topic llm \
  --add-topic agent-memory --add-topic vector-search --add-topic mcp \
  --add-topic mcp-server --add-topic claude --add-topic anthropic \
  --add-topic semantic-cache --add-topic embeddings --add-topic coding-agents \
  --add-topic ai-agents --add-topic python --add-topic hybrid-search
```
