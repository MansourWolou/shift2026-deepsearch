"""
workflow_agent.py
=================
Workflow LangGraph multi-agent DeepSearch.

Architecture :
  ┌─────────────────────────────────────────────────────┐
  │                    START                            │
  │                      │                             │
  │              [router] fan-out                       │
  │         ╔═════╦═════╦═════╦═════╗                  │
  │         ▼     ▼     ▼     ▼     ▼                  │
  │      cerebras groq  qwen llama linkup_agent         │
  │      (Tavily) (Tav) (Tav) (Tav) (Linkup)           │
  │         ╚═════╩═════╩═════╩═════╝                  │
  │                      │  fan-in (operator.add)       │
  │               [synthesizer]                         │
  │                      │                             │
  │                     END                            │
  └─────────────────────────────────────────────────────┘

Modèles via OpenRouter :
  • Cerebras 8B  — meta-llama/llama-3.1-8b-instruct  → provider Cerebras
  • Groq    8B   — meta-llama/llama-3.1-8b-instruct  → provider Groq
  • Qwen  2.5 7B — qwen/qwen-2.5-7b-instruct
  • Llama 3.2 3B — meta-llama/llama-3.2-3b-instruct

Outils de recherche :
  • TavilySearch — agents Cerebras, Groq, Qwen, Llama
  • LinkupSearch — agent dédié (source alternative)

Synthèse :
  • Claude Sonnet 4.5 via OpenRouter (appel direct LLM)

Usage :
    python workflow_agent.py
    python workflow_agent.py --query "..." --output result.json
"""

from __future__ import annotations

import json
import operator
import os
import re
import sys
import textwrap
import time
import argparse
import threading
from typing import Annotated, Any, Optional, Union
from typing_extensions import TypedDict
from uuid import UUID

from dotenv import load_dotenv

# ── LangChain / LangGraph ──────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool as lc_tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY",  "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "https://github.com/shift2026")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "DeepSearch Workflow")
MAX_TOKENS          = int(os.getenv("OPENROUTER_MAX_TOKENS", "1200"))

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY", "")
LINKUP_API_KEY    = os.getenv("LINKUP_API_KEY", "")

# OpenRouter model IDs (agents de recherche)
MODEL_LLAMA_8B      = "meta-llama/llama-3.1-8b-instruct"
MODEL_QWEN_7B       = os.getenv("QWEN_MODEL",         "qwen/qwen-2.5-7b-instruct")
MODEL_MISTRAL_NEMO  = os.getenv("MISTRAL_NEMO_MODEL",  "mistralai/mistral-nemo")

# Claude direct via Anthropic API (synthétiseur)
MODEL_CLAUDE   = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")


# ──────────────────────────────────────────────────────────────────────────────
# Définition des agents
# ──────────────────────────────────────────────────────────────────────────────

AGENT_CONFIGS: list[dict] = [
    {
        "key":            "cerebras_8b",
        "label":          "Cerebras 8B",
        "model":          MODEL_LLAMA_8B,
        "provider_order": ["Cerebras"],
        "temperature":    0.15,
        "search_tool":    "tavily",
        "directive":      "Approche factuelle et structurée : hiérarchise les données, distingue faits établis / inférences / incertitudes.",
        "backstory":      "Expert en analyse factuelle rapide. Tu valorises la précision et la densité informationnelle.",
    },
    {
        "key":            "groq_8b",
        "label":          "Groq 8B",
        "model":          MODEL_LLAMA_8B,
        "provider_order": ["Groq"],
        "temperature":    0.3,
        "search_tool":    "tavily",
        "directive":      "Approche exploratoire : identifie les contre-arguments et angles non conventionnels.",
        "backstory":      "Chercheur critique spécialisé en détection des biais cognitifs et angles morts.",
    },
    {
        "key":            "qwen_7b",
        "label":          "Qwen 2.5 7B",
        "model":          MODEL_QWEN_7B,
        "provider_order": [],
        "temperature":    0.2,
        "search_tool":    "tavily",
        "directive":      "Approche systématique : couvre tous les sous-thèmes, identifie les interdépendances.",
        "backstory":      "Expert en analyse systémique. Tu cartographies les problèmes de façon exhaustive.",
    },
    {
        "key":            "mistral_nemo",
        "label":          "Mistral NeMo 12B",
        "model":          MODEL_MISTRAL_NEMO,
        "provider_order": [],
        "temperature":    0.2,
        "search_tool":    "tavily",
        "directive":      "Approche nuancée et multilingue : contextualise chaque fait, relie les implications pratiques et théoriques.",
        "backstory":      "Analyste polyvalent avec forte capacité de raisonnement. Tu articules les faits avec leurs implications et leur contexte élargi.",
    },
    {
        "key":            "linkup_agent",
        "label":          "Linkup Agent (Qwen 2.5 7B)",
        "model":          MODEL_QWEN_7B,
        "provider_order": [],
        "temperature":    0.25,
        "search_tool":    "linkup",
        "directive":      "Approche adversariale : challenge chaque affirmation, identifie failles et scénarios alternatifs.",
        "backstory":      "Expert en pensée critique. Tu appliques un scepticisme méthodique et construis des contre-narratifs argumentés.",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Timing Callback — mesure thinking vs search séparément
# ──────────────────────────────────────────────────────────────────────────────

class TimingCallback(BaseCallbackHandler):
    """
    Callback LangChain mesurant séparément :
      • thinking_s  — temps cumulé des appels LLM (raisonnement pur)
      • search_s    — temps cumulé des appels outils (recherche web)
      • llm_calls   — nombre d'appels LLM
      • tool_calls  — nombre d'appels outils

    Thread-safe via un verrou interne (un agent = une instance).
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock       = threading.Lock()
        self.thinking_s  = 0.0
        self.search_s    = 0.0
        self.llm_calls   = 0
        self.tool_calls  = 0
        # stacks de timestamps (plusieurs appels peuvent se chevaucher dans un agent)
        self._llm_starts:  dict[str, float] = {}
        self._tool_starts: dict[str, float] = {}

    # ── LLM hooks ────────────────────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict,
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            self._llm_starts[str(run_id)] = time.perf_counter()

    def on_chat_model_start(
        self,
        serialized: dict,
        messages: list,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            self._llm_starts[str(run_id)] = time.perf_counter()

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        with self._lock:
            t0 = self._llm_starts.pop(key, None)
            if t0 is not None:
                self.thinking_s += time.perf_counter() - t0
                self.llm_calls  += 1

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        with self._lock:
            t0 = self._llm_starts.pop(key, None)
            if t0 is not None:
                self.thinking_s += time.perf_counter() - t0
                self.llm_calls  += 1

    # ── Tool hooks ───────────────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: dict,
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            self._tool_starts[str(run_id)] = time.perf_counter()

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        with self._lock:
            t0 = self._tool_starts.pop(key, None)
            if t0 is not None:
                self.search_s  += time.perf_counter() - t0
                self.tool_calls += 1

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        with self._lock:
            t0 = self._tool_starts.pop(key, None)
            if t0 is not None:
                self.search_s  += time.perf_counter() - t0
                self.tool_calls += 1

    def snapshot(self, total_s: float) -> dict:
        """Retourne un snapshot des métriques de timing."""
        with self._lock:
            return {
                "total_s":    round(total_s, 3),
                "thinking_s": round(self.thinking_s, 3),
                "search_s":   round(self.search_s, 3),
                "other_s":    round(max(0.0, total_s - self.thinking_s - self.search_s), 3),
                "llm_calls":  self.llm_calls,
                "tool_calls": self.tool_calls,
            }


# ──────────────────────────────────────────────────────────────────────────────
# LLM Factory — OpenRouter
# ──────────────────────────────────────────────────────────────────────────────

def _or_headers() -> dict:
    return {
        "HTTP-Referer": OPENROUTER_SITE_URL,
        "X-Title":      OPENROUTER_APP_NAME,
    }


def _provider_body(provider_order: list[str]) -> dict:
    cfg: dict = {"allow_fallbacks": True}
    if provider_order:
        cfg["order"] = provider_order
    else:
        cfg["sort"] = "throughput"
    return {"provider": cfg}


def make_llm(
    model: str,
    provider_order: list[str],
    temperature: float = 0.2,
    max_tokens: int = MAX_TOKENS,
) -> ChatOpenAI:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY manquant dans .env")
    return ChatOpenAI(
        model=model,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers=_or_headers(),
        extra_body=_provider_body(provider_order),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def make_claude_llm(
    model: str = MODEL_CLAUDE,
    temperature: float = 0.15,
    max_tokens: int = 3500,
) -> ChatAnthropic:
    """LLM Claude via l'API Anthropic directe (pas OpenRouter)."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY manquant dans .env")
    return ChatAnthropic(
        model=model,
        api_key=ANTHROPIC_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Outils de recherche
# ──────────────────────────────────────────────────────────────────────────────

def make_tavily_tool() -> TavilySearch:
    """Outil TavilySearch — instance par appel (thread-safe)."""
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY manquant dans .env")
    return TavilySearch(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        include_images=False,
    )


def make_linkup_tool():
    """
    Outil LinkupSearch — wrappé comme LangChain BaseTool via @lc_tool.
    Utilise l'API Linkup (linkup.so) pour une source de recherche alternative.
    SDK : pip install linkup-sdk
    Doc : https://docs.linkup.so
    """
    if not LINKUP_API_KEY:
        raise ValueError("LINKUP_API_KEY manquant dans .env")

    try:
        from linkup import LinkupClient
        _client = LinkupClient(api_key=LINKUP_API_KEY)

        @lc_tool
        def linkup_search(query: str) -> str:
            """
            Effectue une recherche web via Linkup (source alternative à Tavily).
            Retourne des résultats récents avec titres, URLs et extraits de contenu.
            Utilise cet outil pour obtenir des informations actualisées et fiables.
            """
            try:
                output = _client.search(
                    q=query,
                    depth="standard",
                    output_type="searchResults",
                )
                results = getattr(output, "results", []) or []
                if not results:
                    return f"Aucun résultat Linkup pour : {query}"
                return "\n\n".join(
                    f"[{r.name}]({r.url})\n{r.content[:500]}"
                    for r in results[:5]
                )
            except Exception as exc:
                return f"Erreur Linkup : {exc}"

        return linkup_search

    except ImportError:
        # Fallback : appel HTTP direct si linkup-sdk non installé
        import urllib.request

        @lc_tool
        def linkup_search(query: str) -> str:
            """
            Effectue une recherche web via Linkup API (HTTP direct).
            Retourne des résultats récents avec titres, URLs et extraits.
            Utilise cet outil pour obtenir des informations actualisées.
            """
            import json as _json
            url     = "https://api.linkup.so/v0/search"
            payload = _json.dumps({
                "q":          query,
                "depth":      "standard",
                "outputType": "searchResults",
            }).encode()
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Authorization": f"Bearer {LINKUP_API_KEY}",
                    "Content-Type":  "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data    = _json.loads(resp.read())
                    results = data.get("results", [])
                    if not results:
                        return f"Aucun résultat Linkup pour : {query}"
                    return "\n\n".join(
                        f"[{r.get('name','?')}]({r.get('url','')})\n{r.get('content','')[:500]}"
                        for r in results[:5]
                    )
            except Exception as exc:
                return f"Erreur Linkup HTTP : {exc}"

        return linkup_search


def get_search_tool(tool_name: str):
    """Factory — retourne l'outil de recherche selon le nom."""
    if tool_name == "linkup":
        return make_linkup_tool()
    return make_tavily_tool()


# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

_AGENT_SYSTEM = """{backstory}

DIRECTIVE : {directive}

Tu disposes d'un outil de recherche web. Effectue 2 à 4 recherches ciblées
pour collecter des informations récentes et fiables avant de répondre.

Réponds UNIQUEMENT avec un objet JSON valide :
{{
  "executive_summary": "Résumé dense en 2-3 phrases basé sur les recherches.",
  "key_points":    ["Point factuel 1", "Point 2", "Point 3"],
  "assumptions":   ["Hypothèse ou limite des sources"],
  "uncertainties": ["Zone d'incertitude ou contradiction entre sources"],
  "sources":       ["URL ou titre de source réelle trouvée"]
}}
Aucun texte avant ou après le JSON."""

_SYNTHESIS_SYSTEM = """Tu es un méta-analyste senior expert en synthèse comparative multi-sources.
Tu reçois les outputs de plusieurs agents de recherche aux approches différentes (factuelle, \
exploratoire, systémique, synthétique, adversariale).
Tu identifies consensus, divergences, angles morts et produis une méta-analyse à forte valeur ajoutée.
Tu ne résumes pas : tu analyses, recroises et enrichis avec un regard critique et structuré."""

_SYNTHESIS_HUMAN = """Méta-analyse comparative de {n} outputs de recherche sur la même requête.

═══════════════════════════════════════════════
REQUÊTE : {query}
═══════════════════════════════════════════════

{context}

═══════════════════════════════════════════════
INSTRUCTIONS D'ANALYSE
═══════════════════════════════════════════════

1. CONSENSUS — Points convergents validés par ≥2 agents (avec haute confiance).
2. CONTRADICTIONS — Désaccords explicites entre agents : cite les agents et l'origine du désaccord.
3. INSIGHTS UNIQUES — Contributions propres à chaque agent, non couvertes par les autres.
4. ANGLES MORTS — Dimensions importantes non traitées par aucun agent.
5. QUALITÉ DES SOURCES — Évalue la fiabilité, fraîcheur et diversité des sources de chaque agent.
6. RÉSULTATS CLÉS — Les 3-5 conclusions factuelles les plus importantes et les mieux étayées.
7. COMPARAISON MÉTHODOLOGIQUE — En quoi les approches divergent et comment elles se complètent.
8. NIVEAU DE CONFIANCE GLOBAL — Score 0-100 avec justification (basé sur convergence + qualité sources).
9. RECHERCHES RECOMMANDÉES — 2-3 axes de recherche complémentaires pour approfondir le sujet.
10. SYNTHÈSE FINALE — Paragraphe de 5-8 phrases dense, exploitant la complémentarité des approches, \
avec nuances et limites épistémiques.

Réponds UNIQUEMENT avec un objet JSON valide :
{{
  "consensus":     ["Point validé par plusieurs agents avec source"],
  "contradictions": ["Agent A affirme X ; Agent B affirme Y — origine probable : ..."],
  "disagreements": ["Désaccord + agents concernés + origine (rétro-compat)"],
  "unique_insights": {{ {unique_keys} }},
  "blind_spots":   ["Dimension non traitée"],
  "key_findings": [
    {{"finding": "Conclusion factuelle 1", "confidence": "high|medium|low", "supported_by": ["agent_key1"]}},
    {{"finding": "Conclusion factuelle 2", "confidence": "high|medium|low", "supported_by": ["agent_key2"]}}
  ],
  "methodology_comparison": {{
    "strengths": {{ {mcomp_keys} }},
    "weaknesses": {{ {mcomp_keys2} }},
    "complementarity": "Comment les approches se complètent mutuellement."
  }},
  "source_quality_assessment": {{ {qa_keys} }},
  "quality_assessment": {{ {qa_keys2} }},
  "confidence_level": {{
    "score": 75,
    "label": "medium|high|low",
    "rationale": "Justification du niveau de confiance global."
  }},
  "recommended_further_research": [
    {{"topic": "Axe de recherche 1", "rationale": "Pourquoi explorer cet axe"}},
    {{"topic": "Axe de recherche 2", "rationale": "Pourquoi explorer cet axe"}}
  ],
  "quality_ranking": {ranking_placeholder},
  "final_synthesis": "Synthèse finale dense (5-8 phrases) avec nuances et limites épistémiques."
}}
Aucun texte avant ou après le JSON."""


# ──────────────────────────────────────────────────────────────────────────────
# LangGraph — State
# ──────────────────────────────────────────────────────────────────────────────

class WorkflowState(TypedDict):
    query:         str
    agent_results: Annotated[list[dict], operator.add]   # fan-in accumulation
    synthesis:     dict


class AgentNodeState(TypedDict):
    query:     str
    agent_key: str


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


def parse_synthesis(raw: str, agent_keys: list[str]) -> dict:
    p = extract_json(raw)
    if p:
        ui   = p.get("unique_insights", {})
        qa   = p.get("quality_assessment", {})
        sqa  = p.get("source_quality_assessment", {})
        mc   = p.get("methodology_comparison", {})
        cl   = p.get("confidence_level", {})
        kf   = p.get("key_findings", [])
        rfr  = p.get("recommended_further_research", [])

        # Normalise key_findings (liste de dicts ou de strings)
        key_findings = []
        for item in kf:
            if isinstance(item, dict):
                key_findings.append({
                    "finding":      str(item.get("finding", "")),
                    "confidence":   str(item.get("confidence", "medium")),
                    "supported_by": list(item.get("supported_by", [])),
                })
            elif isinstance(item, str):
                key_findings.append({"finding": item, "confidence": "medium", "supported_by": []})

        # Normalise recommended_further_research
        recommendations = []
        for item in rfr:
            if isinstance(item, dict):
                recommendations.append({
                    "topic":     str(item.get("topic", "")),
                    "rationale": str(item.get("rationale", "")),
                })
            elif isinstance(item, str):
                recommendations.append({"topic": item, "rationale": ""})

        # Normalise methodology_comparison
        mc_strengths  = mc.get("strengths", {})
        mc_weaknesses = mc.get("weaknesses", {})
        methodology_comparison = {
            "strengths":       {k: str(mc_strengths.get(k, ""))  for k in agent_keys},
            "weaknesses":      {k: str(mc_weaknesses.get(k, "")) for k in agent_keys},
            "complementarity": str(mc.get("complementarity", "")),
        }

        # Normalise confidence_level
        if isinstance(cl, dict):
            confidence_level = {
                "score":    int(cl.get("score", 50)) if str(cl.get("score", "50")).isdigit() else 50,
                "label":    str(cl.get("label", "medium")),
                "rationale": str(cl.get("rationale", "")),
            }
        else:
            confidence_level = {"score": 50, "label": "medium", "rationale": str(cl)}

        return {
            "consensus":              list(p.get("consensus",     [])),
            "contradictions":         list(p.get("contradictions", p.get("disagreements", []))),
            "disagreements":          list(p.get("disagreements", [])),
            "unique_insights":        {k: list(ui.get(k, [])) for k in agent_keys},
            "blind_spots":            list(p.get("blind_spots", [])),
            "key_findings":           key_findings,
            "methodology_comparison": methodology_comparison,
            "source_quality_assessment": {k: str(sqa.get(k, qa.get(k, ""))) for k in agent_keys},
            "quality_assessment":     {k: str(qa.get(k, "")) for k in agent_keys},
            "confidence_level":       confidence_level,
            "recommended_further_research": recommendations,
            "quality_ranking":        list(p.get("quality_ranking", agent_keys)),
            "final_synthesis":        str(p.get("final_synthesis", "")),
        }
    return {
        "consensus": [], "contradictions": [], "disagreements": [],
        "unique_insights":              {k: [] for k in agent_keys},
        "blind_spots":                  [],
        "key_findings":                 [],
        "methodology_comparison":       {
            "strengths": {k: "" for k in agent_keys},
            "weaknesses": {k: "" for k in agent_keys},
            "complementarity": "",
        },
        "source_quality_assessment":    {k: "" for k in agent_keys},
        "quality_assessment":           {k: "" for k in agent_keys},
        "confidence_level":             {"score": 0, "label": "low", "rationale": "Parsing échoué."},
        "recommended_further_research": [],
        "quality_ranking":              agent_keys,
        "final_synthesis":              raw[:2000] if raw else "Parsing échoué.",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Nœuds du workflow LangGraph
# ──────────────────────────────────────────────────────────────────────────────

def router(state: WorkflowState) -> list[Send]:
    """
    Fan-out : envoie la requête à chaque agent en parallèle via Send.
    LangGraph exécute tous les Send simultanément.
    """
    return [
        Send("research_agent", {"query": state["query"], "agent_key": cfg["key"]})
        for cfg in AGENT_CONFIGS
    ]


def research_agent(state: AgentNodeState) -> dict:
    """
    Nœud de recherche : instancie LLM + outil de recherche, exécute l'agent ReAct.
    Appelé en parallèle pour chaque agent par LangGraph.
    Mesure le thinking (LLM) et le search (outil) séparément via TimingCallback.
    """
    cfg      = next(c for c in AGENT_CONFIGS if c["key"] == state["agent_key"])
    callback = TimingCallback()
    t0       = time.perf_counter()

    print(f"  → [{cfg['label']}] démarrage ({cfg['search_tool']})...", file=sys.stderr)

    try:
        llm         = make_llm(cfg["model"], cfg["provider_order"], cfg["temperature"])
        search_tool = get_search_tool(cfg["search_tool"])
        sys_msg     = SystemMessage(content=_AGENT_SYSTEM.format(
            backstory=cfg["backstory"],
            directive=cfg["directive"],
        ))

        agent  = create_agent(model=llm, tools=[search_tool], system_prompt=sys_msg)
        result = agent.invoke(
            {"messages": [("user", state["query"])]},
            config={"callbacks": [callback]},
        )
        raw    = clean_text(str(result["messages"][-1].content))
        status = "✓"

    except Exception as exc:
        raw    = f"Erreur agent : {exc}"
        status = "✗"

    total_s = time.perf_counter() - t0
    timing  = callback.snapshot(total_s)

    print(
        f"  {status} [{cfg['label']}] {total_s:.2f}s total"
        f" | thinking {timing['thinking_s']:.2f}s ({timing['llm_calls']} LLM calls)"
        f" | search {timing['search_s']:.2f}s ({timing['tool_calls']} tool calls)"
        f" — {len(raw)} chars",
        file=sys.stderr,
    )

    return {
        "agent_results": [{
            "key":        cfg["key"],
            "label":      cfg["label"],
            "model":      cfg["model"],
            "tool":       cfg["search_tool"],
            "raw":        raw,
            "structured": parse_research(raw),
            "duration_s": round(total_s, 2),
            "timing":     timing,
        }]
    }


def synthesizer(state: WorkflowState) -> dict:
    """
    Fan-in : reçoit tous les résultats accumulés et produit la synthèse Claude.
    """
    results   = state["agent_results"]
    query     = state["query"]
    agent_keys = [r["key"] for r in results]

    print(f"\n[Workflow] Synthèse Claude ({MODEL_CLAUDE} via Anthropic) sur {len(results)} outputs...", file=sys.stderr)
    t0 = time.perf_counter()

    # Contexte pour Claude
    context = "\n\n".join(
        f"── {r['label']} [{r['tool']}] ──\n{r['raw'][:2500]}"
        for r in results
    )

    # Clés JSON dynamiques selon les agents présents
    unique_keys  = ", ".join(f'"{k}": ["insight propre à {k}"]' for k in agent_keys)
    qa_keys      = ", ".join(f'"{k}": "évaluation qualité sources {k}"' for k in agent_keys)
    qa_keys2     = ", ".join(f'"{k}": "évaluation globale {k}"' for k in agent_keys)
    mcomp_keys   = ", ".join(f'"{k}": "force approche {k}"' for k in agent_keys)
    mcomp_keys2  = ", ".join(f'"{k}": "faiblesse approche {k}"' for k in agent_keys)
    ranking      = json.dumps(agent_keys)

    try:
        llm = make_claude_llm()
        response = llm.invoke([
            SystemMessage(content=_SYNTHESIS_SYSTEM),
            HumanMessage(content=_SYNTHESIS_HUMAN.format(
                n=len(results),
                query=query,
                context=context,
                unique_keys=unique_keys,
                qa_keys=qa_keys,
                qa_keys2=qa_keys2,
                mcomp_keys=mcomp_keys,
                mcomp_keys2=mcomp_keys2,
                ranking_placeholder=ranking,
            )),
        ])
        raw_synthesis = clean_text(str(response.content))
    except Exception as exc:
        raw_synthesis = f"Erreur synthèse : {exc}"

    duration = time.perf_counter() - t0
    print(f"  ✓ [Claude Sonnet] {duration:.2f}s", file=sys.stderr)

    return {
        "synthesis": {
            **parse_synthesis(raw_synthesis, agent_keys),
            "_duration_s": round(duration, 2),
        }
    }


# ──────────────────────────────────────────────────────────────────────────────
# Construction du graphe LangGraph
# ──────────────────────────────────────────────────────────────────────────────

def build_workflow() -> Any:
    """
    Construit et compile le StateGraph LangGraph.

    Topologie :
      START ──[router fan-out]──► research_agent (×N, parallèle)
                                        │ fan-in via operator.add
                                  [synthesizer]
                                        │
                                       END
    """
    graph = StateGraph(WorkflowState)

    graph.add_node("research_agent", research_agent)
    graph.add_node("synthesizer",    synthesizer)

    # Fan-out conditionnel depuis START via router → Send
    graph.add_conditional_edges(START, router, ["research_agent"])

    # Fan-in : tous les research_agent convergent vers synthesizer
    graph.add_edge("research_agent", "synthesizer")
    graph.add_edge("synthesizer",    END)

    return graph.compile()


# ──────────────────────────────────────────────────────────────────────────────
# Interface publique
# ──────────────────────────────────────────────────────────────────────────────

def run_workflow(query: str) -> dict:
    """
    Exécute le workflow complet et retourne le résultat structuré.
    """
    query   = normalize_query(query)
    t_start = time.perf_counter()

    agent_labels = {c["key"]: c["label"] for c in AGENT_CONFIGS}

    print(f"\n[Workflow] Requête : {query}", file=sys.stderr)
    print(f"[Workflow] {len(AGENT_CONFIGS)} agents → fan-out parallèle...\n", file=sys.stderr)

    workflow = build_workflow()
    final_state = workflow.invoke({"query": query, "agent_results": [], "synthesis": {}})

    t_total  = time.perf_counter() - t_start
    results  = final_state["agent_results"]
    agent_keys = [r["key"] for r in results]

    return {
        "query": query,
        "meta": {
            "stack":    "LangGraph + LangChain + OpenRouter",
            "workflow": "fan-out → research_agent×N → synthesizer",
            "tools":    {
                "tavily": [c["key"] for c in AGENT_CONFIGS if c["search_tool"] == "tavily"],
                "linkup": [c["key"] for c in AGENT_CONFIGS if c["search_tool"] == "linkup"],
            },
            "models": {
                c["key"]: {
                    "model":    c["model"],
                    "provider": c["provider_order"] or ["auto:throughput"],
                    "tool":     c["search_tool"],
                }
                for c in AGENT_CONFIGS
            },
            "timing": {
                "total_s":     round(t_total, 2),
                "synthesis_s": final_state["synthesis"].get("_duration_s", 0),
                "per_agent": {
                    r["key"]: r.get("timing", {"total_s": r["duration_s"]})
                    for r in results
                },
            },
        },
        "agent_outputs": {
            r["key"]: {
                "label":      r["label"],
                "tool":       r["tool"],
                "raw":        r["raw"],
                "structured": r["structured"],
                "timing":     r.get("timing", {"total_s": r["duration_s"]}),
            }
            for r in results
        },
        "synthesis": {
            k: v for k, v in final_state["synthesis"].items()
            if not k.startswith("_")
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Générateur HTML
# ──────────────────────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DeepSearch — {query_short}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
:root{{
  --bg:#0d0d14;--surface:#15151f;--card:#1c1c2a;--border:#2a2a3d;
  --text:#e2e2f0;--muted:#6b6b90;--accent:#7c3aed;--accent2:#a855f7;
  --thinking:#7c3aed;--search:#06b6d4;--other:#2e2e42;
  --good:#10b981;--warn:#f59e0b;--danger:#f43f5e;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);font-size:14px;line-height:1.6}}
a{{color:var(--accent2);text-decoration:none}}
/* Header */
header{{background:var(--surface);border-bottom:1px solid var(--border);padding:18px 32px}}
.hrow{{display:flex;align-items:flex-start;gap:24px;flex-wrap:wrap;max-width:1200px;margin:0 auto}}
.hlogo{{font-size:1rem;font-weight:800;color:var(--accent2);white-space:nowrap;padding-top:4px}}
.hquery{{flex:1;min-width:180px}}
.hquery label{{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}}
.hquery p{{font-size:.95rem;font-weight:600;margin-top:2px}}
.hstats{{display:flex;gap:20px;flex-wrap:wrap}}
.hs{{text-align:center}}
.hs .v{{font-size:1.4rem;font-weight:800;color:var(--accent2)}}
.hs .l{{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}}
/* Layout */
main{{max-width:1200px;margin:0 auto;padding:28px 32px;display:flex;flex-direction:column;gap:28px}}
.stitle{{font-size:.82rem;font-weight:700;color:var(--accent2);text-transform:uppercase;letter-spacing:.1em;
  margin-bottom:14px;display:flex;align-items:center;gap:10px}}
.stitle::after{{content:'';flex:1;height:1px;background:var(--border)}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:18px}}
.ctitle{{font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin-bottom:12px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
.grid3{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}}
/* Timing bars */
.trow{{display:grid;grid-template-columns:152px 1fr 72px;align-items:center;gap:10px;margin-bottom:8px}}
.alabel{{display:flex;align-items:center;gap:7px;font-size:.84rem;font-weight:600}}
.dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
.track{{height:22px;border-radius:4px;background:var(--bg);display:flex;overflow:hidden}}
.seg{{height:100%}}
.seg:first-child{{border-radius:4px 0 0 4px}}
.seg:last-child{{border-radius:0 4px 4px 0}}
.ttotal{{font-size:.84rem;font-weight:700;text-align:right}}
.legend{{display:flex;gap:14px;font-size:.76rem;color:var(--muted);margin-top:8px;flex-wrap:wrap}}
.legend span{{display:flex;align-items:center;gap:5px}}
.legend i{{width:10px;height:10px;border-radius:2px;display:inline-block}}
/* ReAct steps */
.srow{{display:grid;grid-template-columns:152px 1fr;align-items:center;gap:10px;margin-bottom:10px}}
.steps{{display:flex;align-items:center;gap:3px;flex-wrap:nowrap;overflow:hidden}}
.sblk{{border-radius:4px;display:flex;align-items:center;justify-content:center;
  font-size:.68rem;font-weight:700;color:#fff;padding:0 6px;height:26px;white-space:nowrap;flex-shrink:0}}
.sllm{{background:var(--thinking)}}
.stool{{background:var(--search)}}
.sarrow{{color:var(--muted);font-size:.8rem;flex-shrink:0;padding:0 2px}}
.smeta{{font-size:.73rem;color:var(--muted);margin-left:8px;white-space:nowrap}}
/* Search table */
table{{width:100%;border-collapse:collapse}}
th{{text-align:left;padding:8px 10px;font-size:.74rem;text-transform:uppercase;
  letter-spacing:.06em;color:var(--muted);border-bottom:1px solid var(--border)}}
td{{padding:9px 10px;border-bottom:1px solid var(--border);font-size:.83rem;vertical-align:middle}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:#1e1e2e}}
.badge{{padding:2px 7px;border-radius:4px;font-size:.7rem;font-weight:700;text-transform:uppercase}}
.bt{{background:#0e3344;color:#06b6d4}}
.bl{{background:#2d1b5e;color:#8b5cf6}}
.mbar{{display:flex;align-items:center;gap:6px}}
.mbtrack{{flex:1;height:5px;background:var(--bg);border-radius:3px;overflow:hidden;min-width:40px}}
.mbfill{{height:100%;border-radius:3px}}
.mbval{{font-size:.8rem;font-weight:700;min-width:34px;text-align:right}}
/* Thinking cards */
.tcard{{background:var(--card);border:1px solid var(--border);border-radius:10px;overflow:hidden}}
.tchead{{padding:12px 16px;display:flex;align-items:center;gap:10px;border-bottom:1px solid var(--border)}}
.tchead h3{{font-size:.9rem;font-weight:700;flex:1}}
.tcbody{{padding:14px 16px}}
.summary{{font-size:.82rem;color:var(--muted);font-style:italic;margin-bottom:10px;
  padding-bottom:10px;border-bottom:1px solid var(--border)}}
.kps{{list-style:none;display:flex;flex-direction:column;gap:4px}}
.kps li{{font-size:.81rem;padding-left:12px;position:relative}}
.kps li::before{{content:'›';position:absolute;left:0;color:var(--accent2)}}
.tmini{{display:flex;gap:10px;margin-top:12px;padding-top:10px;border-top:1px solid var(--border)}}
.tm{{flex:1;display:flex;flex-direction:column;align-items:center}}
.tm .tv{{font-size:.92rem;font-weight:700}}
.tm .tl{{font-size:.68rem;color:var(--muted)}}
/* Synthesis */
.scard{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:20px}}
.conf-row{{display:flex;align-items:center;gap:16px;padding:14px;background:var(--bg);border-radius:8px;margin-bottom:18px}}
.cring{{position:relative;width:68px;height:68px;flex-shrink:0}}
.cring svg{{transform:rotate(-90deg)}}
.cring .cv{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:1rem;font-weight:800}}
.cinfo .cl{{font-size:.95rem;font-weight:700;margin-bottom:3px}}
.cinfo .cr{{font-size:.8rem;color:var(--muted)}}
.slist{{list-style:none;display:flex;flex-direction:column;gap:5px}}
.slist li{{font-size:.82rem;padding:7px 10px;background:var(--bg);border-radius:6px;border-left:3px solid var(--border)}}
.slist li.good{{border-color:var(--good)}}
.slist li.warn{{border-color:var(--warn)}}
.slist li.danger{{border-color:var(--danger)}}
.kfc{{padding:9px 12px;background:var(--bg);border-radius:7px;display:flex;gap:9px;align-items:flex-start;margin-bottom:5px}}
.kfbadge{{padding:2px 6px;border-radius:3px;font-size:.68rem;font-weight:700;text-transform:uppercase;flex-shrink:0;margin-top:1px}}
.kfbadge.high{{background:#0d3321;color:var(--good)}}
.kfbadge.medium{{background:#2d2000;color:var(--warn)}}
.kfbadge.low{{background:#2d0f0f;color:var(--danger)}}
.kftxt{{font-size:.82rem;flex:1}}
.kfby{{font-size:.7rem;color:var(--muted);margin-top:2px}}
.final{{padding:16px;background:var(--bg);border-radius:8px;border-left:4px solid var(--accent);
  font-size:.88rem;line-height:1.8;margin-top:16px}}
.mcomp-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:8px;margin-top:8px}}
.mcc{{background:var(--bg);border-radius:7px;padding:10px 12px}}
.mcl{{font-size:.73rem;font-weight:700;color:var(--muted);margin-bottom:6px;display:flex;align-items:center;gap:5px}}
.mcs{{font-size:.79rem;color:var(--good);margin-bottom:3px}}
.mcw{{font-size:.79rem;color:var(--warn)}}
.rec{{padding:9px 12px;background:var(--bg);border-radius:7px;border-left:3px solid var(--accent2);margin-bottom:6px}}
.rtopic{{font-size:.84rem;font-weight:700;margin-bottom:2px}}
.rratio{{font-size:.78rem;color:var(--muted)}}
footer{{text-align:center;padding:20px;font-size:.76rem;color:var(--muted);border-top:1px solid var(--border)}}
@media(max-width:800px){{.grid2,.grid3{{grid-template-columns:1fr}}.trow{{grid-template-columns:120px 1fr 60px}}}}
</style>
</head>
<body>
<header>
  <div class="hrow">
    <div class="hlogo">⚡ DeepSearch</div>
    <div class="hquery">
      <label>Requête analysée</label>
      <p id="hq"></p>
    </div>
    <div class="hstats">
      <div class="hs"><div class="v" id="ht"></div><div class="l">Durée totale</div></div>
      <div class="hs"><div class="v" id="ha"></div><div class="l">Agents</div></div>
      <div class="hs"><div class="v" id="hs2"></div><div class="l">Synthèse</div></div>
      <div class="hs"><div class="v" id="hc"></div><div class="l">Confiance</div></div>
    </div>
  </div>
</header>
<main>

<!-- 1. Temps de traitement -->
<section>
  <div class="stitle">① Temps de traitement</div>
  <div class="card">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px">
      <div class="ctitle" style="margin:0">Thinking · Search · Overhead par agent</div>
      <div class="legend">
        <span><i style="background:var(--thinking)"></i>Thinking LLM</span>
        <span><i style="background:var(--search)"></i>Search outil</span>
        <span><i style="background:var(--other)"></i>Overhead</span>
      </div>
    </div>
    <div id="tbars"></div>
  </div>
</section>

<!-- 2. Pertinence du thinking -->
<section>
  <div class="stitle">② Pertinence du thinking</div>
  <div class="grid2" style="margin-bottom:18px">
    <div class="card" style="height:280px">
      <div class="ctitle">% thinking vs total par agent</div>
      <canvas id="chart-ratio"></canvas>
    </div>
    <div class="card" style="height:280px">
      <div class="ctitle">Temps moyen par appel LLM (s)</div>
      <canvas id="chart-avg"></canvas>
    </div>
  </div>
  <div id="agent-cards" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px"></div>
</section>

<!-- 3. Comparaison des searches -->
<section>
  <div class="stitle">③ Comparaison des recherches</div>
  <div class="grid2" style="margin-bottom:18px">
    <div class="card" style="height:280px">
      <div class="ctitle">Appels LLM vs outil</div>
      <canvas id="chart-calls"></canvas>
    </div>
    <div class="card" style="height:280px">
      <div class="ctitle">Séquence ReAct par agent</div>
      <div id="react-steps" style="margin-top:8px"></div>
    </div>
  </div>
  <div class="card">
    <table>
      <thead><tr>
        <th>Agent</th><th>Outil</th><th>Appels</th>
        <th>Search total</th><th>Moy/appel</th><th>Sources</th><th>% search/total</th>
      </tr></thead>
      <tbody id="stbody"></tbody>
    </table>
  </div>
</section>

<!-- 4. Synthèse Claude -->
<section>
  <div class="stitle">④ Synthèse Claude — {claude_model}</div>
  <div class="scard">
    <div class="conf-row" id="conf"></div>
    <div class="grid2">
      <div>
        <div class="ctitle">Résultats clés</div>
        <div id="kfindings"></div>
        <div class="ctitle" style="margin-top:16px">Consensus</div>
        <ul class="slist" id="consensus"></ul>
        <div class="ctitle" style="margin-top:16px">Contradictions</div>
        <ul class="slist" id="contradictions"></ul>
      </div>
      <div>
        <div class="ctitle">Angles morts</div>
        <ul class="slist" id="blindspots"></ul>
        <div class="ctitle" style="margin-top:16px">Comparaison méthodologique</div>
        <div id="mcomp"></div>
        <div class="ctitle" style="margin-top:16px">Recherches recommandées</div>
        <div id="recs"></div>
      </div>
    </div>
    <div class="ctitle" style="margin-top:18px">Synthèse finale</div>
    <div class="final" id="final"></div>
  </div>
</section>

</main>
<footer id="ft"></footer>

<script>
const D = {data_json};
const COLORS = {{
  cerebras_8b:'#06b6d4',groq_8b:'#f59e0b',qwen_7b:'#10b981',
  mistral_nemo:'#f43f5e',linkup_agent:'#8b5cf6'
}};
const col = k => COLORS[k]||'#7c3aed';
const fmt = n => typeof n==='number'?n.toFixed(2)+'s':n;

const T = D.meta.timing, PA = T.per_agent, AO = D.agent_outputs, SY = D.synthesis;
const keys = Object.keys(PA);
const maxT  = Math.max(...keys.map(k=>PA[k].total_s||0),1);
const maxSS = Math.max(...keys.map(k=>PA[k].search_s||0),1);

// Header
document.getElementById('hq').textContent  = D.query;
document.getElementById('ht').textContent  = T.total_s+'s';
document.getElementById('ha').textContent  = keys.length;
document.getElementById('hs2').textContent = T.synthesis_s+'s';
const cs = SY.confidence_level?.score??'?';
document.getElementById('hc').textContent  = cs+'%';
document.getElementById('ft').textContent  = D.meta.stack+' · '+D.meta.workflow;

// ① Timing bars
const tb = document.getElementById('tbars');
keys.forEach(k => {{
  const t=PA[k], lbl=AO[k]?.label||k, c=col(k);
  const pct = v => ((v/maxT)*100).toFixed(1)+'%';
  tb.innerHTML += `<div class="trow">
    <div class="alabel"><span class="dot" style="background:${{c}}"></span>${{lbl}}</div>
    <div class="track" title="thinking:${{t.thinking_s}}s  search:${{t.search_s}}s  overhead:${{t.other_s||0}}s">
      <div class="seg" style="width:${{pct(t.thinking_s)}};background:var(--thinking)"></div>
      <div class="seg" style="width:${{pct(t.search_s)}};background:var(--search)"></div>
      <div class="seg" style="width:${{pct(t.other_s||0)}};background:var(--other)"></div>
    </div>
    <div class="ttotal">${{t.total_s}}s</div>
  </div>`;
}});

// ② Charts
const charts = [];
const mkChart = (id,type,data,opts) => {{
  const c=new Chart(document.getElementById(id),{{type,data,options:{{
    responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{labels:{{color:'#6b6b90',font:{{size:10}}}}}}}},
    scales:type!=='doughnut'?{{
      x:{{ticks:{{color:'#6b6b90',font:{{size:10}}}},grid:{{color:'#1e1e2e'}}}},
      y:{{ticks:{{color:'#6b6b90'}},grid:{{color:'#1e1e2e'}},beginAtZero:true}}
    }}:undefined,
    ...opts
  }}}});
  charts.push(c);
}};

// ratio thinking/total
mkChart('chart-ratio','bar',{{
  labels: keys.map(k=>AO[k]?.label||k),
  datasets:[{{
    label:'% thinking',
    data:keys.map(k=>PA[k].total_s>0?+((PA[k].thinking_s/PA[k].total_s)*100).toFixed(1):0),
    backgroundColor:keys.map(col),borderRadius:4
  }}]
}},{{plugins:{{legend:{{display:false}}}}}});

// avg per LLM call
mkChart('chart-avg','bar',{{
  labels: keys.map(k=>AO[k]?.label||k),
  datasets:[{{
    label:'s/appel LLM',
    data:keys.map(k=>PA[k].llm_calls?+(PA[k].thinking_s/PA[k].llm_calls).toFixed(2):0),
    backgroundColor:keys.map(col),borderRadius:4
  }}]
}},{{plugins:{{legend:{{display:false}}}}}});

// LLM vs tool calls
mkChart('chart-calls','bar',{{
  labels: keys.map(k=>AO[k]?.label||k),
  datasets:[
    {{label:'Appels LLM', data:keys.map(k=>PA[k].llm_calls||0),  backgroundColor:'#7c3aed',borderRadius:4}},
    {{label:'Appels outil',data:keys.map(k=>PA[k].tool_calls||0), backgroundColor:'#06b6d4',borderRadius:4}}
  ]
}},{{}});

// Agent cards
const ac = document.getElementById('agent-cards');
keys.forEach(k => {{
  const t=PA[k], o=AO[k]||{{}}, s=o.structured||{{}}, c=col(k);
  const tr = t.total_s>0?((t.thinking_s/t.total_s)*100).toFixed(0):0;
  const tc = +tr>70?'#f59e0b':'#10b981';
  const tb2= o.tool==='linkup'?'<span class="badge bl">Linkup</span>':'<span class="badge bt">Tavily</span>';
  const kps=(s.key_points||[]).slice(0,4).map(p=>`<li>${{p}}</li>`).join('');
  ac.innerHTML += `<div class="tcard">
    <div class="tchead" style="border-top:3px solid ${{c}}">
      <span class="dot" style="width:12px;height:12px;background:${{c}}"></span>
      <h3>${{o.label||k}}</h3>${{tb2}}
      <span class="badge" style="background:#1e1e2e;color:var(--muted)">${{t.total_s}}s</span>
    </div>
    <div class="tcbody">
      <p class="summary">${{s.executive_summary||'—'}}</p>
      <ul class="kps">${{kps||'<li style="color:var(--muted)">—</li>'}}</ul>
      <div class="tmini">
        <div class="tm"><span class="tv" style="color:#a78bfa">${{t.thinking_s}}s</span><span class="tl">Thinking</span></div>
        <div class="tm"><span class="tv" style="color:#06b6d4">${{t.search_s}}s</span><span class="tl">Search</span></div>
        <div class="tm"><span class="tv" style="color:var(--muted)">${{t.llm_calls}}·${{t.tool_calls}}</span><span class="tl">LLM·Tool</span></div>
        <div class="tm"><span class="tv" style="color:${{tc}}">${{tr}}%</span><span class="tl">% thinking</span></div>
      </div>
    </div>
  </div>`;
}});

// ReAct steps
const rs = document.getElementById('react-steps');
keys.forEach(k => {{
  const t=PA[k], c=col(k), lbl=AO[k]?.label||k;
  const n=t.llm_calls||1, m=t.tool_calls||0;
  const al=t.thinking_s/n, at=m>0?t.search_s/m:0;
  const maxD=t.total_s||1;
  let html=`<div class="srow"><div class="alabel"><span class="dot" style="background:${{c}}"></span>${{lbl}}</div><div class="steps">`;
  const tc2=Math.min(m,n-1);
  const pw=d=>Math.max(38,+(d/maxD*260).toFixed(0))+'px';
  html+=`<div class="sblk sllm" style="width:${{pw(al)}}" title="LLM ~${{al.toFixed(2)}}s">LLM ${{al.toFixed(1)}}s</div>`;
  for(let i=0;i<tc2;i++){{
    html+=`<span class="sarrow">›</span>`;
    html+=`<div class="sblk stool" style="width:${{pw(at)}}" title="Tool ~${{at.toFixed(2)}}s">🔍 ${{at.toFixed(1)}}s</div>`;
    html+=`<span class="sarrow">›</span>`;
    html+=`<div class="sblk sllm" style="width:${{pw(al)}}" title="LLM ~${{al.toFixed(2)}}s">LLM ${{al.toFixed(1)}}s</div>`;
  }}
  html+=`<span class="smeta">${{n}} LLM · ${{m}} outil</span></div></div>`;
  rs.innerHTML += html;
}});

// Search table
const sb = document.getElementById('stbody');
keys.forEach(k => {{
  const t=PA[k], o=AO[k]||{{}}, c=col(k);
  const tc=t.tool_calls||0, ss=t.search_s||0;
  const avg=tc>0?(ss/tc).toFixed(2)+'s':'—';
  const srcs=(o.structured?.sources||[]).length;
  const rat=t.total_s>0?((ss/t.total_s)*100).toFixed(0):0;
  const tb2=o.tool==='linkup'?'<span class="badge bl">Linkup</span>':'<span class="badge bt">Tavily</span>';
  sb.innerHTML += `<tr>
    <td><span class="alabel"><span class="dot" style="background:${{c}}"></span>${{o.label||k}}</span></td>
    <td>${{tb2}}</td>
    <td style="font-weight:700">${{tc}}</td>
    <td><div class="mbar"><div class="mbtrack"><div class="mbfill" style="width:${{(ss/maxSS*100).toFixed(0)}}%;background:var(--search)"></div></div><span class="mbval">${{ss.toFixed(2)}}s</span></div></td>
    <td>${{avg}}</td>
    <td>${{srcs}}</td>
    <td><div class="mbar"><div class="mbtrack"><div class="mbfill" style="width:${{rat}}%;background:var(--thinking)"></div></div><span class="mbval">${{rat}}%</span></div></td>
  </tr>`;
}});

// ④ Synthesis
const cl=SY.confidence_level||{{score:0,label:'?',rationale:''}};
const sc=parseInt(cl.score)||0;
const cc=sc>=70?'#10b981':sc>=40?'#f59e0b':'#f43f5e';
const circ=2*Math.PI*26, dash=(sc/100*circ).toFixed(1);
document.getElementById('conf').innerHTML=`
  <div class="cring"><svg width="68" height="68" viewBox="0 0 68 68">
    <circle cx="34" cy="34" r="26" fill="none" stroke="#1e1e2e" stroke-width="8"/>
    <circle cx="34" cy="34" r="26" fill="none" stroke="${{cc}}" stroke-width="8"
      stroke-dasharray="${{dash}} ${{circ.toFixed(1)}}" stroke-linecap="round"/>
  </svg><div class="cv" style="color:${{cc}}">${{sc}}%</div></div>
  <div class="cinfo"><div class="cl" style="color:${{cc}}">${{cl.label?.toUpperCase()||'—'}}</div>
  <div class="cr">${{cl.rationale||'—'}}</div></div>`;

const kfe=document.getElementById('kfindings');
(SY.key_findings||[]).forEach(kf=>{{
  const by=(kf.supported_by||[]).join(', ');
  kfe.innerHTML+=`<div class="kfc">
    <span class="kfbadge ${{kf.confidence||'medium'}}">${{kf.confidence||'—'}}</span>
    <div><div class="kftxt">${{kf.finding||kf}}</div>${{by?`<div class="kfby">Supporté par : ${{by}}</div>`:''}}</div>
  </div>`;
}});

const ul=(id,items,cls)=>{{
  const el=document.getElementById(id);
  el.innerHTML=items.length?items.map(x=>`<li class="${{cls}}">${{x}}</li>`).join(''):'<li style="color:var(--muted)">—</li>';
}};
ul('consensus', SY.consensus||[], 'good');
const ct=SY.contradictions?.length?SY.contradictions:SY.disagreements||[];
ul('contradictions', ct, 'danger');
ul('blindspots', SY.blind_spots||[], 'warn');

const mc=SY.methodology_comparison||{{}};
const mg=document.getElementById('mcomp');
let mchtml='<div class="mcomp-grid">';
keys.forEach(k=>{{
  const c=col(k), lbl=AO[k]?.label||k;
  mchtml+=`<div class="mcc"><div class="mcl"><span class="dot" style="background:${{c}}"></span>${{lbl}}</div>
    <div class="mcs">+ ${{mc.strengths?.[k]||'—'}}</div>
    <div class="mcw">− ${{mc.weaknesses?.[k]||'—'}}</div></div>`;
}});
mchtml+='</div>';
if(mc.complementarity) mchtml+=`<p style="font-size:.79rem;color:var(--muted);margin-top:8px;font-style:italic">${{mc.complementarity}}</p>`;
mg.innerHTML=mchtml;

const re=document.getElementById('recs');
(SY.recommended_further_research||[]).forEach(r=>{{
  re.innerHTML+=`<div class="rec"><div class="rtopic">🔭 ${{r.topic||r}}</div>${{r.rationale?`<div class="rratio">${{r.rationale}}</div>`:''}}</div>`;
}});

document.getElementById('final').textContent = SY.final_synthesis||'—';
</script>
</body>
</html>"""


def generate_html_report(result: dict) -> str:
    """Génère un rapport HTML auto-contenu à partir du résultat du workflow."""
    query_short = result.get("query", "")[:60]
    claude_model = MODEL_CLAUDE
    data_json = json.dumps(result, ensure_ascii=False)
    return _HTML_TEMPLATE.format(
        query_short=query_short,
        claude_model=claude_model,
        data_json=data_json,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Workflow LangGraph — DeepSearch multi-agent via OpenRouter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Agents : Cerebras 8B · Groq 8B · Qwen 2.5 7B · Llama 3.2 3B · Linkup Agent
            Outils : TavilySearch (×4) + LinkupSearch (×1)
            Synth. : Claude Sonnet 4.5 via OpenRouter

            Variables d'environnement requises :
              OPENROUTER_API_KEY   — clé OpenRouter
              TAVILY_API_KEY       — clé Tavily
              LINKUP_API_KEY       — clé Linkup (https://linkup.so)

            Exemples :
              python workflow_agent.py
              python workflow_agent.py --query "Le système solaire a 8 planètes"
              python workflow_agent.py --query "..." --output result.json --no-pretty
        """),
    )
    parser.add_argument(
        "--query",
        default="l'afganistant à envahi les usa parcequ'il possède la bombe atomique",
        help="Requête DeepSearch",
    )
    parser.add_argument("--output",    default=None,        help="Fichier de sortie JSON")
    parser.add_argument("--no-pretty", action="store_true", help="JSON compact")
    args = parser.parse_args()

    indent = None if args.no_pretty else 2

    try:
        result      = run_workflow(args.query)
        output_json = json.dumps(result, ensure_ascii=False, indent=indent)
        print(output_json)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(output_json)
            print(f"\n[OK] JSON : {args.output}", file=sys.stderr)

        html_content = generate_html_report(result)
        with open("workflow_report.html", "w", encoding="utf-8") as fh:
            fh.write(html_content)
        print(f"[OK] Rapport : workflow_report.html", file=sys.stderr)

    except ValueError as exc:
        print(json.dumps({"error": f"Configuration : {exc}"}, ensure_ascii=False, indent=2))
        sys.exit(1)
    except KeyboardInterrupt:
        print(json.dumps({"error": "Interrompu."}, ensure_ascii=False, indent=2))
        sys.exit(130)
    except Exception as exc:
        print(json.dumps({"error": f"Erreur : {exc}"}, ensure_ascii=False, indent=2))
        sys.exit(1)
