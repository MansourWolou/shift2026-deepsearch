"""
agent-orchestrator-claude-openrouter.py
========================================
Multi-Agent DeepSearch — LangChain + OpenRouter (sans CrewAI).

Stack :
  • LangChain  : ChatOpenAI, TavilySearch, create_agent (LangGraph ReAct)
  • OpenRouter : routing provider, HTTP-Referer / X-Title headers
  • Python     : ThreadPoolExecutor — Phase 1 concurrente

Agents de recherche (Phase 1 — CONCURRENT, 5 threads) :
  • Cerebras 8B  — meta-llama/llama-3.1-8b-instruct → provider Cerebras (>2 000 t/s)
  • Groq    8B   — meta-llama/llama-3.1-8b-instruct → provider Groq    (>750  t/s)
  • Qwen  2.5 7B — qwen/qwen-2.5-7b-instruct        → sort: throughput
  • Llama 3.2 3B — meta-llama/llama-3.2-3b-instruct → sort: throughput
  • Mistral NeMo — mistralai/mistral-nemo            → sort: throughput

Chaque agent dispose de TavilySearch pour effectuer de vraies recherches web.

Comparateur (Phase 2 — séquentiel) :
  • Claude Sonnet 4.5 via OpenRouter — synthèse des 5 outputs (appel direct LLM)

Usage:
    python agent-orchestrator-claude-openrouter.py
    python agent-orchestrator-claude-openrouter.py --query "..." --output result.json
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents import create_agent

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY",  "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "https://github.com/shift2026")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "DeepSearch Multi-Agent")
MAX_TOKENS          = int(os.getenv("OPENROUTER_MAX_TOKENS", "1200"))
TAVILY_API_KEY      = os.getenv("TAVILY_API_KEY", "")

# ── Model IDs OpenRouter ───────────────────────────────────────────────────────
MODEL_LLAMA_8B = "meta-llama/llama-3.1-8b-instruct"   # Cerebras + Groq
MODEL_QWEN_7B  = os.getenv("QWEN_MODEL",    "qwen/qwen-2.5-7b-instruct")
MODEL_LLAMA_3B = os.getenv("LLAMA32_MODEL", "meta-llama/llama-3.2-3b-instruct")
MODEL_NEMO     = "mistralai/mistral-nemo"
MODEL_CLAUDE   = "anthropic/claude-sonnet-4-5"


# ──────────────────────────────────────────────────────────────────────────────
# Agent descriptor
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentSpec:
    key:            str
    label:          str
    model:          str
    provider_order: list[str]   # [] = sort:throughput automatique
    temperature:    float
    directive:      str         # angle méthodologique
    backstory:      str


AGENT_SPECS: list[AgentSpec] = [
    AgentSpec(
        key="cerebras_8b",
        label="Cerebras 8B (Llama 3.1)",
        model=MODEL_LLAMA_8B,
        provider_order=["Cerebras"],
        temperature=0.15,
        directive="Approche factuelle et structurée : hiérarchise les données, distingue faits / inférences / incertitudes, densité informationnelle maximale.",
        backstory="Expert en analyse factuelle rapide. Tu vas à l'essentiel, structures l'information de façon hiérarchique et valorises la précision sur la rhétorique.",
    ),
    AgentSpec(
        key="groq_8b",
        label="Groq 8B (Llama 3.1)",
        model=MODEL_LLAMA_8B,
        provider_order=["Groq"],
        temperature=0.35,
        directive="Approche exploratoire : identifie les contre-arguments, angles non conventionnels et remise en question des présupposés dominants.",
        backstory="Chercheur critique spécialisé en détection des biais cognitifs. Tu explores les implications systémiques cachées et valorises la diversité des perspectives.",
    ),
    AgentSpec(
        key="qwen_7b",
        label="Qwen 2.5 7B",
        model=MODEL_QWEN_7B,
        provider_order=[],
        temperature=0.2,
        directive="Approche systématique et exhaustive : couvre tous les sous-thèmes, identifie les interdépendances, produit une analyse structurée et complète.",
        backstory="Expert en analyse systémique. Tu cartographies les problèmes de façon exhaustive et identifies les relations causales et effets de second ordre.",
    ),
    AgentSpec(
        key="llama_3b",
        label="Llama 3.2 3B",
        model=MODEL_LLAMA_3B,
        provider_order=[],
        temperature=0.1,
        directive="Approche ultra-concise : chaque point clé en une phrase dense. Pas de superflu. Priorité à la clarté et à l'actionabilité.",
        backstory="Spécialiste en communication d'élite. Tu condenses l'information complexe en insights denses et actionnables. Aucun mot inutile.",
    ),
    AgentSpec(
        key="mistral_nemo",
        label="Mistral NeMo 12B",
        model=MODEL_NEMO,
        provider_order=[],
        temperature=0.4,
        directive="Approche adversariale : joue l'avocat du diable, challenge chaque affirmation, identifie les failles logiques et les scénarios alternatifs.",
        backstory="Expert en pensée critique et raisonnement adversarial. Tu appliques un scepticisme méthodique à chaque affirmation et construis des scénarios alternatifs.",
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# LLM Factory — ChatOpenAI + OpenRouter
# Réf. : https://openrouter.ai/docs/quickstart
# ──────────────────────────────────────────────────────────────────────────────

def _or_headers() -> dict:
    """Headers recommandés par OpenRouter (leaderboard + identification app)."""
    return {
        "HTTP-Referer": OPENROUTER_SITE_URL,
        "X-Title":      OPENROUTER_APP_NAME,
    }


def _provider_body(provider_order: list[str]) -> dict:
    """
    Objet provider OpenRouter pour le routing.
    Réf. : https://openrouter.ai/docs/provider-routing
    """
    cfg: dict = {"allow_fallbacks": True}
    if provider_order:
        cfg["order"] = provider_order
    else:
        cfg["sort"] = "throughput"
    return {"provider": cfg}


def make_research_llm(spec: AgentSpec) -> ChatOpenAI:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY manquant dans .env")
    return ChatOpenAI(
        model=spec.model,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers=_or_headers(),
        # OpenRouter provider routing must be sent in extra_body.
        extra_body=_provider_body(spec.provider_order),
        temperature=spec.temperature,
        max_tokens=MAX_TOKENS,
    )


def make_claude_llm() -> ChatOpenAI:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY manquant dans .env")
    return ChatOpenAI(
        model=MODEL_CLAUDE,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers=_or_headers(),
        # OpenRouter provider routing must be sent in extra_body.
        extra_body={"provider": {"allow_fallbacks": True, "sort": "latency"}},
        temperature=0.15,
        max_tokens=2500,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tavily Tool
# ──────────────────────────────────────────────────────────────────────────────

def make_tavily_tool() -> TavilySearch:
    """Instance TavilySearch par thread (thread-safe)."""
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY manquant dans .env")
    return TavilySearch(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        include_images=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

_RESEARCH_SYSTEM = textwrap.dedent("""
    {backstory}

    DIRECTIVE MÉTHODOLOGIQUE : {directive}

    Tu as accès à l'outil TavilySearch pour effectuer des recherches web en temps réel.
    Effectue entre 2 et 5 recherches ciblées pour collecter des informations récentes et fiables.

    Après tes recherches, réponds UNIQUEMENT avec un objet JSON valide :
    {{
      "executive_summary": "Résumé dense en 2-3 phrases basé sur les résultats de recherche.",
      "key_points":    ["Point factuel issu des recherches 1", "Point 2", "Point 3"],
      "assumptions":   ["Hypothèse ou limite des sources trouvées"],
      "uncertainties": ["Zone d'incertitude ou contradiction entre sources"],
      "sources":       ["URL ou titre de source réelle trouvée"]
    }}
    Aucun texte avant ou après le JSON.
""").strip()

_COMPARISON_SYSTEM = textwrap.dedent("""
    Tu es un méta-analyste senior expert en synthèse comparative multi-sources.
    Tu reçois 5 analyses de recherche aux approches délibérément différentes.
    Tu analyses, tu compares, tu identifies convergences et divergences, tu enrichis.
    Tu ne résumes pas : tu produis une valeur ajoutée par rapport à chaque analyse individuelle.
""").strip()

_COMPARISON_PROMPT = textwrap.dedent("""
    Analyse comparative de 5 réponses de recherche sur la même requête.

    REQUÊTE : {query}

    {context}

    Réponds UNIQUEMENT avec un objet JSON valide :
    {{
      "consensus": ["Point sur lequel la majorité des agents convergent"],
      "disagreements": ["Désaccord : description + agents concernés + origine"],
      "unique_insights": {{
        "cerebras_8b":  ["Insight propre à Cerebras"],
        "groq_8b":      ["Insight propre à Groq"],
        "qwen_7b":      ["Insight propre à Qwen"],
        "llama_3b":     ["Insight propre à Llama"],
        "mistral_nemo": ["Insight propre à Mistral NeMo"]
      }},
      "blind_spots": ["Dimension non traitée par aucun des 5 agents"],
      "quality_ranking": ["cerebras_8b", "groq_8b", "qwen_7b", "llama_3b", "mistral_nemo"],
      "quality_assessment": {{
        "cerebras_8b":  "Forces et faiblesses.",
        "groq_8b":      "Forces et faiblesses.",
        "qwen_7b":      "Forces et faiblesses.",
        "llama_3b":     "Forces et faiblesses.",
        "mistral_nemo": "Forces et faiblesses."
      }},
      "final_synthesis": "Synthèse finale (4-6 phrases) exploitant la complémentarité des 5 approches."
    }}
    Aucun texte avant ou après le JSON.
""").strip()


# ──────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ──────────────────────────────────────────────────────────────────────────────

def normalize_query(raw: str) -> str:
    return " ".join(raw.strip().split())


def clean_text(text: str) -> str:
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_json(text: str) -> Optional[dict]:
    candidates = [text.strip()]
    for pat in [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]:
        m = re.search(pat, text, re.DOTALL)
        if m:
            candidates.append(m.group(1))
    m = re.search(r"\{[\s\S]*\}", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            p = json.loads(c.strip())
            if isinstance(p, dict):
                return p
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def parse_research(raw: str) -> dict:
    p = extract_json(raw)
    if p:
        return {
            "executive_summary": str(p.get("executive_summary", "")),
            "key_points":    list(p.get("key_points",    [])),
            "assumptions":   list(p.get("assumptions",   [])),
            "uncertainties": list(p.get("uncertainties", [])),
            "sources":       list(p.get("sources",       [])),
        }
    return {
        "executive_summary": raw[:500] if raw else "Parsing échoué.",
        "key_points": [], "assumptions": [],
        "uncertainties": ["Réponse non parseable."],
        "sources": [],
    }


def parse_comparison(raw: str, agent_keys: list[str]) -> dict:
    p = extract_json(raw)
    if p:
        ui = p.get("unique_insights", {})
        qa = p.get("quality_assessment", {})
        return {
            "consensus":     list(p.get("consensus",     [])),
            "disagreements": list(p.get("disagreements", [])),
            "unique_insights": {k: list(ui.get(k, [])) for k in agent_keys},
            "blind_spots":   list(p.get("blind_spots",   [])),
            "quality_ranking": list(p.get("quality_ranking", agent_keys)),
            "quality_assessment": {k: str(qa.get(k, "")) for k in agent_keys},
            "final_synthesis": str(p.get("final_synthesis", "")),
        }
    return {
        "consensus": [], "disagreements": [],
        "unique_insights": {k: [] for k in agent_keys},
        "blind_spots": [],
        "quality_ranking": agent_keys,
        "quality_assessment": {k: "" for k in agent_keys},
        "final_synthesis": raw[:2000] if raw else "Parsing échoué.",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Exécution d'un agent de recherche (thread-safe)
# ──────────────────────────────────────────────────────────────────────────────

def _run_one_agent(spec: AgentSpec, query: str) -> tuple[str, str, float]:
    """
    Crée un agent LangChain ReAct + TavilySearch et exécute la requête.
    Retourne (key, raw_output, duration_s).
    Appelé depuis ThreadPoolExecutor — chaque thread a ses propres instances LLM + tool.
    """
    t0 = time.perf_counter()

    llm    = make_research_llm(spec)
    tavily = make_tavily_tool()

    system_prompt = SystemMessage(content=_RESEARCH_SYSTEM.format(
        backstory=spec.backstory,
        directive=spec.directive,
    ))

    # Agent ReAct LangChain avec TavilySearch comme outil
    graph  = create_agent(model=llm, tools=[tavily], system_prompt=system_prompt)
    result = graph.invoke({"messages": [("user", query)]})
    raw    = clean_text(str(result["messages"][-1].content))

    return spec.key, raw, time.perf_counter() - t0


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrateur
# ──────────────────────────────────────────────────────────────────────────────

class OpenRouterOrchestrator:
    """
    Phase 1 — 5 agents LangChain ReAct + Tavily en parallèle (ThreadPoolExecutor).
    Phase 2 — Claude Sonnet synthétise les 5 outputs (appel LLM direct).
    """

    def __init__(self, max_workers: int = 5) -> None:
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY manquant dans .env")
        if not TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY manquant dans .env")
        self.max_workers = max_workers
        self.agent_keys  = [s.key for s in AGENT_SPECS]

    # ── Phase 1 ───────────────────────────────────────────────────────────────

    def _research_phase(self, query: str) -> tuple[dict[str, str], dict[str, float]]:
        raw_outputs: dict[str, str]   = {}
        durations:   dict[str, float] = {}

        print(
            f"\n[OpenRouter] {len(AGENT_SPECS)} agents LangChain+Tavily en parallèle "
            f"(workers={self.max_workers})...",
            file=sys.stderr,
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures: dict[Future, AgentSpec] = {
                executor.submit(_run_one_agent, spec, query): spec
                for spec in AGENT_SPECS
            }
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    key, raw, dur = future.result()
                    raw_outputs[key] = raw
                    durations[key]   = dur
                    print(f"  ✓ [{spec.label}] {dur:.2f}s — {len(raw)} chars", file=sys.stderr)
                except Exception as exc:
                    print(f"  ✗ [{spec.label}] ERREUR : {exc}", file=sys.stderr)
                    raw_outputs[spec.key] = f"Erreur : {exc}"
                    durations[spec.key]   = 0.0

        return raw_outputs, durations

    # ── Phase 2 ───────────────────────────────────────────────────────────────

    def _comparison_phase(self, query: str, raw_outputs: dict[str, str]) -> tuple[str, float]:
        t0 = time.perf_counter()

        context = "\n\n".join(
            f"── {spec.label} ({spec.key}) ──\n{raw_outputs.get(spec.key, 'N/A')[:2500]}"
            for spec in AGENT_SPECS
        )

        llm = make_claude_llm()
        response = llm.invoke([
            SystemMessage(content=_COMPARISON_SYSTEM),
            HumanMessage(content=_COMPARISON_PROMPT.format(query=query, context=context)),
        ])

        return clean_text(str(response.content)), time.perf_counter() - t0

    # ── Entrée principale ─────────────────────────────────────────────────────

    def run(self, query: str) -> dict:
        query   = normalize_query(query)
        t_start = time.perf_counter()

        # Phase 1
        t1 = time.perf_counter()
        raw_outputs, agent_durations = self._research_phase(query)
        t_research = time.perf_counter() - t1

        structured = {k: parse_research(raw_outputs.get(k, "")) for k in self.agent_keys}

        # Phase 2
        print("\n[OpenRouter] Claude synthétise les 5 outputs...", file=sys.stderr)
        comparison_raw, t_comparison = self._comparison_phase(query, raw_outputs)
        print(f"  ✓ [Claude Sonnet] {t_comparison:.2f}s", file=sys.stderr)

        comparative = parse_comparison(comparison_raw, self.agent_keys)
        t_total     = time.perf_counter() - t_start

        return {
            "query": query,
            "meta": {
                "stack": "LangChain + OpenRouter (sans CrewAI)",
                "openrouter": {
                    "base_url":   OPENROUTER_BASE_URL,
                    "site_url":   OPENROUTER_SITE_URL,
                    "app_name":   OPENROUTER_APP_NAME,
                    "max_tokens": MAX_TOKENS,
                },
                "tools": {
                    "tavily_search": {
                        "search_depth": "advanced",
                        "max_results":  5,
                        "agents":       "tous les 5 agents de recherche",
                    }
                },
                "models": {
                    spec.key: {
                        "model":    spec.model,
                        "provider": spec.provider_order or ["auto:throughput"],
                        "temp":     spec.temperature,
                    }
                    for spec in AGENT_SPECS
                } | {"claude": {"model": MODEL_CLAUDE, "provider": ["auto:latency"]}},
                "timing": {
                    "total_s":      round(t_total,      2),
                    "research_s":   round(t_research,   2),
                    "comparison_s": round(t_comparison, 2),
                    "per_agent":    {k: round(v, 2) for k, v in agent_durations.items()},
                    "concurrency_gain_x": round(
                        sum(agent_durations.values()) / max(t_research, 0.001), 2
                    ),
                },
            },
            "agent_outputs": {
                k: {"raw": raw_outputs.get(k, ""), "structured": structured[k]}
                for k in self.agent_keys
            },
            "comparative_analysis": comparative,
        }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DeepSearch — LangChain + OpenRouter + TavilySearch (sans CrewAI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Agents : Cerebras 8B · Groq 8B · Qwen 2.5 7B · Llama 3.2 3B · Mistral NeMo 12B
            Outil  : TavilySearch (recherche web temps réel)
            Synth. : Claude Sonnet 4.5 via OpenRouter

            Exemple :
              python agent-orchestrator-claude-openrouter.py --query "Le soleil est une étoile"
              python agent-orchestrator-claude-openrouter.py --output result.json --workers 5
        """),
    )
    parser.add_argument(
        "--query",
        default="il y a entre 2 et 4 milliards d'étoiles dans le système solaire",
        help="Requête DeepSearch",
    )
    parser.add_argument("--output",    default=None,        help="Fichier de sortie JSON")
    parser.add_argument("--workers",   type=int, default=5, help="Threads parallèles (défaut: 5)")
    parser.add_argument("--no-pretty", action="store_true", help="JSON compact")
    args = parser.parse_args()

    indent = None if args.no_pretty else 2

    try:
        orchestrator = OpenRouterOrchestrator(max_workers=args.workers)
        result       = orchestrator.run(args.query)
        output_json  = json.dumps(result, ensure_ascii=False, indent=indent)
        print(output_json)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(output_json)
            print(f"\n[OK] Sauvegardé : {args.output}", file=sys.stderr)

    except ValueError as exc:
        print(json.dumps({"error": f"Configuration : {exc}"}, ensure_ascii=False, indent=2))
        sys.exit(1)
    except KeyboardInterrupt:
        print(json.dumps({"error": "Interrompu."}, ensure_ascii=False, indent=2))
        sys.exit(130)
    except Exception as exc:
        print(json.dumps({"error": f"Erreur : {exc}"}, ensure_ascii=False, indent=2))
        sys.exit(1)
