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

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
LINKUP_API_KEY = os.getenv("LINKUP_API_KEY", "")

# OpenRouter model IDs
MODEL_LLAMA_8B = "meta-llama/llama-3.1-8b-instruct"
MODEL_QWEN_7B  = os.getenv("QWEN_MODEL",    "qwen/qwen-2.5-7b-instruct")
MODEL_LLAMA_3B = os.getenv("LLAMA32_MODEL", "meta-llama/llama-3.2-3b-instruct")
MODEL_CLAUDE   = "anthropic/claude-sonnet-4-5"


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
        "key":            "llama_3b",
        "label":          "Llama 3.2 3B",
        "model":          MODEL_LLAMA_3B,
        "provider_order": [],
        "temperature":    0.1,
        "search_tool":    "tavily",
        "directive":      "Approche ultra-concise : chaque point clé en une phrase dense, priorité à l'actionabilité.",
        "backstory":      "Spécialiste en communication d'élite. Aucun mot inutile. Chaque phrase apporte une valeur unique.",
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
        # OpenRouter custom routing fields must go into extra_body with current langchain-openai/openai.
        extra_body=_provider_body(provider_order),
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

    print(f"\n[Workflow] Synthèse Claude sur {len(results)} outputs...", file=sys.stderr)
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
        llm = make_llm(MODEL_CLAUDE, [], temperature=0.15, max_tokens=3500)
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
        default="il y a entre 2 et 4 milliards d'étoiles dans le système solaire",
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
