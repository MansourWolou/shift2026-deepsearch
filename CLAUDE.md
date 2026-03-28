# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fact-checking engine that deploys multiple LLM research agents in parallel (different models, same query), then synthesizes results via a thinking model to produce a structured verdict. Built on LangChain/LangGraph + Tavily web search.

## Commands

```bash
# Install dependencies (uses uv, not pip)
uv sync

# Run fact-check (default: 5 agents + Claude thinking synthesis)
python factcheck.py "L'info a verifier"

# Choose specific agents
python factcheck.py --agents openai,gemini,perplexity "..."

# Choose synthesis model
python factcheck.py --synthesis-provider openai --synthesis-model o3 "..."

# Export JSON
python factcheck.py --output result.json --verbose "..."
```

No test suite or linter configured.

## Architecture

### Pipeline (`factcheck.py`)

```
Query -> N research agents (ThreadPoolExecutor, parallel) -> Thinking model -> Verdict JSON
```

1. **Research Phase**: N agents run in parallel via `ThreadPoolExecutor`. Each agent does ReAct loops with Tavily web search. Different LLMs provide diverse perspectives.
2. **Synthesis Phase**: A thinking model (Claude Sonnet 4.5 extended thinking by default) cross-references all research outputs and produces a fact-check verdict with confidence score.

### Research Agents (`deepsearch_*.py`)

Each file implements one LLM provider using LangChain + LangGraph ReAct pattern with Tavily Search. All expose `build_agent()` returning a LangGraph runnable. Exception: Perplexity also exposes `deep_research()` for native search (no Tavily).

Default 5 agents: Claude, OpenAI GPT-4o, Gemini Flash, Perplexity (native), Mistral Large. Additional available: Grok, Groq.

### Config Resolution

CLI args > env vars (`FACTCHECK_*`) > hardcoded defaults in `factcheck.py`.

### Output Schema

Top-level: `query`, `timestamp`, `config`, `timing`, `verdict`. The verdict contains: `claim`, `verdict` (TRUE/FALSE/PARTIALLY TRUE/UNVERIFIABLE/MISLEADING), `confidence`, `summary`, `evidence_for`, `evidence_against`, `consensus`, `disagreements`, `blind_spots`, `nuances`.

## Key Patterns

- **Agent registry**: dict in `factcheck.py` maps agent names to invoke functions. Adding a provider = adding a registry entry.
- **Dynamic imports**: `deepsearch_*.py` modules loaded on demand to avoid importing all providers at startup.
- **JSON extraction**: multi-strategy fallback (direct parse, ```json block, regex `{...}`).
- **Mistral edge case**: content can be `list[dict]` instead of `str` — handled in `_invoke_langgraph()`.
- **French language**: agent prompts and comments are in French.
- **Environment**: all API keys via `.env` file (see `.env.example`).
