# MANIFEST (Compatible Codex + Claude Code)

This file is a CLAUDE.md-style project manifest intended for both Codex and Claude Code.
Use it as the shared source of truth for agent behavior in this repository.

## 1) Project Scope

- Repository: `shift2026`
- Primary goal: build and iterate fast DeepSearch orchestrators and provider-specific agents.
- Main stack: Python 3.12, CrewAI, LangChain, OpenAI-compatible providers, Tavily.

## 2) Core Principles

- Prefer pragmatic, minimal changes over large refactors.
- Keep outputs structured (`JSON`) when possible.
- Optimize for speed/cost first, then depth when needed.
- Fail clearly with actionable error messages.

## 3) Working Rules

- Read existing files before editing.
- Preserve current conventions and naming patterns.
- Avoid destructive git commands unless explicitly requested.
- Do not revert unrelated local changes.
- Add lightweight validation (`py_compile`, `--help`, targeted run) after edits.

## 4) Runtime Conventions

- Environment variables are loaded from `.env`.
- Add any new required variables to `.env.example`.
- For OpenAI-compatible APIs, use explicit `base_url` and `api_key`.
- Keep model defaults configurable via environment variables.

## 5) Orchestrator Design Contract

- Normalize input query.
- Run at least two analysis paths (analytic + alternative) when orchestrating.
- Add a comparison/synthesis step.
- Return:
  - `query`
  - `timing`
  - `model_config`
  - `agent_outputs`
  - `comparative_analysis`

## 6) Performance Contract (Blazing Fast Mode)

- Use fast/default-small models by default.
- Keep prompts concise and bounded.
- Limit iterative depth/tool recursion.
- Add fallbacks for token/rate-limit errors where feasible.

## 7) Quality Contract

- Prefer deterministic, parseable outputs.
- Parse/validate JSON responses defensively.
- Include uncertainties and sources in research outputs.
- Surface assumptions explicitly.

## 8) Useful Commands

```bash
.venv/bin/python -m py_compile agent-orchestrator.py
.venv/bin/python -m py_compile agent-orchestrator-telemetry.py
.venv/bin/python -m py_compile agent-orchestrator-openrouteur.py
.venv/bin/python agent-orchestrator-openrouteur.py --help
```

## 9) Sync Note

If needed, mirror this file into `CLAUDE.md` and/or `AGENTS.md` to keep a single policy across tools.
