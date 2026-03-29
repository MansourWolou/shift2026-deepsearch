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
# Tarifs estimatifs (USD / 1M tokens)
# ──────────────────────────────────────────────────────────────────────────────

COST_LLM: dict[str, dict] = {
    MODEL_LLAMA_8B:    {"input": 0.055, "output": 0.055},
    MODEL_QWEN_7B:     {"input": 0.07,  "output": 0.07},
    MODEL_MISTRAL_NEMO:{"input": 0.13,  "output": 0.13},
    MODEL_CLAUDE:      {"input": 3.00,  "output": 15.00},
}
COST_SEARCH: dict[str, float] = {
    "tavily": 0.01,   # USD par appel
    "linkup": 0.005,  # USD par appel
}


def estimate_llm_cost(model: str, input_tok: int, output_tok: int) -> float:
    rates = COST_LLM.get(model, {"input": 0.10, "output": 0.10})
    return round((input_tok * rates["input"] + output_tok * rates["output"]) / 1_000_000, 8)


def estimate_search_cost(tool: str, calls: int) -> float:
    return round(COST_SEARCH.get(tool, 0.0) * calls, 8)


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
        self._lock        = threading.Lock()
        self.thinking_s   = 0.0
        self.search_s     = 0.0
        self.llm_calls    = 0
        self.tool_calls   = 0
        self.input_tokens  = 0
        self.output_tokens = 0
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
            # Collecte les tokens via usage_metadata
            for gen_list in response.generations:
                for gen in gen_list:
                    usage = getattr(getattr(gen, "message", None), "usage_metadata", None)
                    if usage:
                        self.input_tokens  += usage.get("input_tokens",  0)
                        self.output_tokens += usage.get("output_tokens", 0)

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

    def snapshot(self, total_s: float, model: str = "", search_tool: str = "") -> dict:
        """Retourne un snapshot des métriques de timing et de coût."""
        with self._lock:
            llm_cost    = estimate_llm_cost(model, self.input_tokens, self.output_tokens)
            search_cost = estimate_search_cost(search_tool, self.tool_calls)
            return {
                "total_s":       round(total_s, 3),
                "thinking_s":    round(self.thinking_s, 3),
                "search_s":      round(self.search_s, 3),
                "other_s":       round(max(0.0, total_s - self.thinking_s - self.search_s), 3),
                "llm_calls":     self.llm_calls,
                "tool_calls":    self.tool_calls,
                "input_tokens":  self.input_tokens,
                "output_tokens": self.output_tokens,
                "cost": {
                    "llm_usd":    llm_cost,
                    "search_usd": search_cost,
                    "total_usd":  round(llm_cost + search_cost, 8),
                },
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

Tu disposes d'un outil de recherche web. Effectue 2 à 4 recherches ciblées \
pour collecter des informations récentes et fiables avant de répondre.

Rédige ta réponse en texte structuré avec exactement ces sections :

**RÉSUMÉ** : 2 à 3 phrases denses résumant l'essentiel.
**POINTS CLÉS** : liste à puces des faits importants (3 à 5 points).
**INCERTITUDES** : zones d'ombre ou contradictions entre sources.
**SOURCES** : liste des URLs ou titres de sources consultées."""

_SYNTHESIS_SYSTEM = """Tu es un méta-analyste senior. Tu reçois les outputs de plusieurs agents \
de recherche et tu produis une analyse comparative approfondie en texte structuré.
Tu ne résumes pas : tu analyses, croises les sources et révèles ce qu'aucun agent seul ne peut voir."""

_SYNTHESIS_HUMAN = """Méta-analyse de {n} agents de recherche sur la même requête.

═══════════════════════════════════════════════
REQUÊTE : {query}
═══════════════════════════════════════════════

{context}

═══════════════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════════════

Produis une analyse structurée. Utilise exactement ces sections dans cet ordre, \
chaque titre préfixé par "## " :

## RÉSULTATS CLÉS
3 à 5 conclusions factuelles solides, étayées par plusieurs agents. Une par ligne, préfixée par "- ".

## CONSENSUS
Points convergents validés par au moins 2 agents. Une par ligne, préfixée par "- ".

## CONTRADICTIONS
Désaccords entre agents. Format : "Agent X affirme … ; Agent Y affirme …". Une par ligne.

## COMPARAISON DES RAISONNEMENTS
Pour chaque agent, évalue en 1-2 phrases : qualité du raisonnement, originalité, pertinence des sources, \
contribution unique. Format strict :
**[Label exact de l'agent]** : évaluation.

## ANGLES MORTS
Dimensions importantes non couvertes par aucun agent. Une par ligne, préfixée par "- ".

## NIVEAU DE CONFIANCE
Score global : X/100. Justification courte (2-3 phrases) basée sur convergence et qualité des sources.

## RECHERCHES RECOMMANDÉES
2 à 3 axes complémentaires. Format : "- Sujet : raison".

## SYNTHÈSE FINALE
Paragraphe de 5 à 8 phrases dense, exploitant la complémentarité des approches, avec nuances \
et limites épistémiques."""


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
    """Extrait les sections **TITRE** : contenu du texte libre de l'agent."""
    sections: dict[str, str] = {}
    current: Optional[str] = None
    buf: list[str] = []

    for line in raw.split("\n"):
        m = re.match(r"^\*\*([^*]+)\*\*\s*:?\s*(.*)", line)
        if m:
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = m.group(1).strip().upper()
            buf = [m.group(2).strip()] if m.group(2).strip() else []
        else:
            if current is not None:
                buf.append(line)

    if current is not None:
        sections[current] = "\n".join(buf).strip()

    def bullets(text: str) -> list[str]:
        return [l.lstrip("-•* ").strip() for l in text.split("\n") if l.strip().lstrip("-•* ").strip()]

    return {
        "executive_summary": sections.get("RÉSUMÉ", raw[:400] if raw else "—"),
        "key_points":        bullets(sections.get("POINTS CLÉS", "")),
        "uncertainties":     bullets(sections.get("INCERTITUDES", "")),
        "sources":           bullets(sections.get("SOURCES", "")),
    }


def parse_synthesis_markdown(raw: str) -> dict:
    """
    Extrait les sections Markdown (## TITRE) de la réponse de synthèse.
    Retourne un dict {section_name: contenu} + confidence_score.
    """
    sections: dict[str, str] = {}
    current: Optional[str] = None
    buf: list[str] = []

    for line in raw.split("\n"):
        if line.startswith("## "):
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = line[3:].strip().upper()
            buf = []
        else:
            if current is not None:
                buf.append(line)

    if current is not None:
        sections[current] = "\n".join(buf).strip()

    # Extrait le score de confiance depuis la section dédiée
    conf_text = sections.get("NIVEAU DE CONFIANCE", "")
    score = 50
    m = re.search(r"(\d{1,3})\s*/\s*100", conf_text)
    if m:
        score = max(0, min(100, int(m.group(1))))

    return {
        "raw":              raw,
        "sections":         sections,
        "confidence_score": score,
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
    timing  = callback.snapshot(total_s, cfg["model"], cfg["search_tool"])

    cost_usd = timing["cost"]["total_usd"]
    print(
        f"  {status} [{cfg['label']}] {total_s:.2f}s total"
        f" | thinking {timing['thinking_s']:.2f}s ({timing['llm_calls']} LLM)"
        f" | search {timing['search_s']:.2f}s ({timing['tool_calls']} calls)"
        f" | ~${cost_usd:.5f}"
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
    Fan-in : reçoit tous les résultats accumulés et produit la synthèse Claude (texte Markdown structuré).
    """
    results    = state["agent_results"]
    query      = state["query"]

    print(f"\n[Workflow] Synthèse Claude ({MODEL_CLAUDE} via Anthropic) sur {len(results)} outputs...", file=sys.stderr)
    t0 = time.perf_counter()

    # Contexte pour Claude — inclut label, outil, timing et contenu brut
    context_parts = []
    for r in results:
        t = r.get("timing", {})
        header = (
            f"── {r['label']} [{r['tool']}]"
            f" | thinking {t.get('thinking_s', 0):.1f}s ({t.get('llm_calls', 0)} LLM)"
            f" | search {t.get('search_s', 0):.1f}s ({t.get('tool_calls', 0)} appels) ──"
        )
        context_parts.append(f"{header}\n{r['raw'][:2500]}")
    context = "\n\n".join(context_parts)

    try:
        llm = make_claude_llm()
        response = llm.invoke([
            SystemMessage(content=_SYNTHESIS_SYSTEM),
            HumanMessage(content=_SYNTHESIS_HUMAN.format(
                n=len(results),
                query=query,
                context=context,
            )),
        ])
        raw_synthesis = clean_text(str(response.content))
        usage = getattr(response, "usage_metadata", None)
        real_in  = usage.get("input_tokens",  0) if usage else 0
        real_out = usage.get("output_tokens",  0) if usage else 0
    except Exception as exc:
        raw_synthesis = f"## SYNTHÈSE FINALE\nErreur synthèse : {exc}"
        real_in  = 0
        real_out = 0

    duration    = time.perf_counter() - t0
    claude_cost = estimate_llm_cost(MODEL_CLAUDE, real_in, real_out)
    print(
        f"  ✓ [Claude Sonnet] {duration:.2f}s"
        f" | {real_in} in / {real_out} out tokens"
        f" | ~${claude_cost:.5f}",
        file=sys.stderr,
    )

    return {
        "synthesis": {
            **parse_synthesis_markdown(raw_synthesis),
            "_duration_s":    round(duration, 2),
            "_cost_usd":      claude_cost,
            "_input_tokens":  real_in,
            "_output_tokens": real_out,
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

    agents_cost = sum(r.get("timing", {}).get("cost", {}).get("total_usd", 0.0) for r in results)
    claude_cost = final_state["synthesis"].get("_cost_usd", 0.0)
    total_cost  = round(agents_cost + claude_cost, 6)

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
            "cost": {
                "total_usd":         total_cost,
                "agents_usd":        round(agents_cost, 6),
                "claude_usd":        round(claude_cost, 6),
                "claude_in_tokens":  final_state["synthesis"].get("_input_tokens", 0),
                "claude_out_tokens": final_state["synthesis"].get("_output_tokens", 0),
                "per_agent": {
                    r["key"]: r.get("timing", {}).get("cost", {"total_usd": 0.0})
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
  --good:#10b981;--warn:#f59e0b;--danger:#f43f5e;--cost:#f97316;
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
.conf-row{{display:flex;align-items:center;gap:16px;padding:14px;background:var(--bg);border-radius:8px;margin-bottom:20px}}
.cring{{position:relative;width:68px;height:68px;flex-shrink:0}}
.cring svg{{transform:rotate(-90deg)}}
.cring .cv{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:1rem;font-weight:800}}
.cinfo .cl{{font-size:.95rem;font-weight:700;margin-bottom:3px}}
.cinfo .cr{{font-size:.8rem;color:var(--muted)}}
.syn-block{{margin-bottom:18px}}
.syn-block .ctitle{{margin-bottom:8px}}
.syn-lines{{list-style:none;display:flex;flex-direction:column;gap:4px}}
.syn-lines li{{font-size:.83rem;padding:7px 11px;background:var(--bg);border-radius:6px;border-left:3px solid var(--border)}}
.syn-lines li.good{{border-color:var(--good)}}
.syn-lines li.warn{{border-color:var(--warn)}}
.syn-lines li.danger{{border-color:var(--danger)}}
.reason-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px;margin-bottom:20px}}
.rcard{{background:var(--bg);border-radius:8px;padding:12px 14px;border-left:4px solid var(--border)}}
.rcard-head{{display:flex;align-items:center;gap:8px;margin-bottom:6px}}
.rcard-head h4{{font-size:.84rem;font-weight:700;flex:1}}
.rcard-body{{font-size:.8rem;color:var(--muted);line-height:1.55}}
.final{{padding:16px;background:var(--bg);border-radius:8px;border-left:4px solid var(--accent);
  font-size:.88rem;line-height:1.8;margin-top:4px}}
.rec{{padding:9px 12px;background:var(--bg);border-radius:7px;border-left:3px solid var(--accent2);margin-bottom:6px}}
.rtopic{{font-size:.84rem;font-weight:700;margin-bottom:2px}}
.rratio{{font-size:.78rem;color:var(--muted)}}
footer{{text-align:center;padding:20px;font-size:.76rem;color:var(--muted);border-top:1px solid var(--border)}}
/* Cost cards */
.cost-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:14px;margin-bottom:18px}}
.ccard{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 16px}}
.ccard-head{{display:flex;align-items:center;gap:8px;margin-bottom:10px}}
.ccard-head h4{{font-size:.85rem;font-weight:700;flex:1}}
.cost-row{{display:flex;justify-content:space-between;font-size:.78rem;margin-bottom:3px}}
.cost-row .cv2{{font-weight:700}}
.cost-total{{font-size:.95rem;font-weight:800;color:var(--cost);margin-top:6px;padding-top:6px;border-top:1px solid var(--border)}}
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
      <div class="hs"><div class="v" id="hcost" style="color:var(--cost)"></div><div class="l">Coût estimé</div></div>
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

<!-- 4. Estimation des coûts -->
<section>
  <div class="stitle">④ Estimation des coûts</div>
  <div class="cost-grid" id="cost-cards"></div>
  <div class="grid2">
    <div class="card" style="height:280px">
      <div class="ctitle">Coût LLM vs Search par agent (USD)</div>
      <canvas id="chart-cost-bar"></canvas>
    </div>
    <div class="card" style="height:280px">
      <div class="ctitle">Tokens input / output par agent</div>
      <canvas id="chart-tokens"></canvas>
    </div>
  </div>
</section>

<!-- 5. Synthèse Claude -->
<section>
  <div class="stitle">⑤ Synthèse Claude — {claude_model}</div>
  <div class="scard">
    <div class="conf-row" id="conf"></div>

    <div class="ctitle">Comparaison des raisonnements</div>
    <div class="reason-grid" id="reason-grid"></div>

    <div class="grid2">
      <div>
        <div class="syn-block">
          <div class="ctitle">Résultats clés</div>
          <ul class="syn-lines" id="syn-resultats"></ul>
        </div>
        <div class="syn-block">
          <div class="ctitle">Consensus</div>
          <ul class="syn-lines" id="syn-consensus"></ul>
        </div>
        <div class="syn-block">
          <div class="ctitle">Contradictions</div>
          <ul class="syn-lines" id="syn-contradictions"></ul>
        </div>
      </div>
      <div>
        <div class="syn-block">
          <div class="ctitle">Angles morts</div>
          <ul class="syn-lines" id="syn-angles"></ul>
        </div>
        <div class="syn-block">
          <div class="ctitle">Recherches recommandées</div>
          <div id="syn-recs"></div>
        </div>
      </div>
    </div>

    <div class="ctitle" style="margin-top:4px">Synthèse finale</div>
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

const COST = D.meta.cost||{{}};
const fmtUsd = v => v!=null?'$'+Number(v).toFixed(4):'—';

// Header
document.getElementById('hq').textContent    = D.query;
document.getElementById('ht').textContent    = T.total_s+'s';
document.getElementById('ha').textContent    = keys.length;
document.getElementById('hs2').textContent   = T.synthesis_s+'s';
const cs = SY.confidence_score??'?';
document.getElementById('hc').textContent    = cs+'%';
document.getElementById('hcost').textContent = fmtUsd(COST.total_usd);
document.getElementById('ft').textContent    = D.meta.stack+' · '+D.meta.workflow;

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

// ④ Cost cards
const cc2 = document.getElementById('cost-cards');
const perAgent = COST.per_agent||{{}};
keys.forEach(k => {{
  const t=PA[k], o=AO[k]||{{}}, c=col(k), ac=perAgent[k]||{{}};
  const llmC=ac.llm_usd||0, srC=ac.search_usd||0, totC=ac.total_usd||0;
  const tb2=o.tool==='linkup'?'<span class="badge bl">Linkup</span>':'<span class="badge bt">Tavily</span>';
  cc2.innerHTML+=`<div class="ccard" style="border-top:3px solid ${{c}}">
    <div class="ccard-head"><span class="dot" style="background:${{c}}"></span><h4>${{o.label||k}}</h4>${{tb2}}</div>
    <div class="cost-row"><span style="color:var(--muted)">LLM (${{t.input_tokens||0}}in/${{t.output_tokens||0}}out tok)</span><span class="cv2">${{fmtUsd(llmC)}}</span></div>
    <div class="cost-row"><span style="color:var(--muted)">Search (${{t.tool_calls||0}} appels)</span><span class="cv2">${{fmtUsd(srC)}}</span></div>
    <div class="cost-total">Total ≈ ${{fmtUsd(totC)}}</div>
  </div>`;
}});
// Claude cost card
cc2.innerHTML+=`<div class="ccard" style="border-top:3px solid #a855f7">
  <div class="ccard-head"><span class="dot" style="background:#a855f7"></span><h4>Claude Sonnet</h4><span class="badge" style="background:#2d1b5e;color:#c4b5fd">Anthropic</span></div>
  <div class="cost-row"><span style="color:var(--muted)">Tokens utilisés</span><span class="cv2">${{COST.claude_in_tokens||0}} in / ${{COST.claude_out_tokens||0}} out</span></div>
  <div class="cost-total">Total ≈ ${{fmtUsd(COST.claude_usd)}}</div>
</div>`;

// Cost stacked bar
mkChart('chart-cost-bar','bar',{{
  labels: keys.map(k=>AO[k]?.label||k),
  datasets:[
    {{label:'LLM ($)', data:keys.map(k=>(perAgent[k]?.llm_usd||0).toFixed(6)), backgroundColor:'#7c3aed',borderRadius:4}},
    {{label:'Search ($)', data:keys.map(k=>(perAgent[k]?.search_usd||0).toFixed(6)), backgroundColor:'#f97316',borderRadius:4}}
  ]
}},{{scales:{{x:{{stacked:true}},y:{{stacked:true}}}}}});

// Token bar
mkChart('chart-tokens','bar',{{
  labels: keys.map(k=>AO[k]?.label||k),
  datasets:[
    {{label:'Input tokens', data:keys.map(k=>PA[k].input_tokens||0), backgroundColor:'#7c3aed',borderRadius:4}},
    {{label:'Output tokens', data:keys.map(k=>PA[k].output_tokens||0), backgroundColor:'#06b6d4',borderRadius:4}}
  ]
}},{{}});

// ⑤ Synthesis — helper: parse markdown sections from raw text
const SEC = SY.sections||{{}};
const getLines = name => (SEC[name]||'').split('\\n')
  .map(l=>l.replace(/^[-*\u2022]\\s*/,'').trim()).filter(Boolean);

// Confidence ring
const sc = SY.confidence_score||0;
const cc = sc>=70?'#10b981':sc>=40?'#f59e0b':'#f43f5e';
const circ = 2*Math.PI*26, dash=(sc/100*circ).toFixed(1);
const confText = SEC['NIVEAU DE CONFIANCE']||'—';
document.getElementById('conf').innerHTML=`
  <div class="cring"><svg width="68" height="68" viewBox="0 0 68 68">
    <circle cx="34" cy="34" r="26" fill="none" stroke="#1e1e2e" stroke-width="8"/>
    <circle cx="34" cy="34" r="26" fill="none" stroke="${{cc}}" stroke-width="8"
      stroke-dasharray="${{dash}} ${{circ.toFixed(1)}}" stroke-linecap="round"/>
  </svg><div class="cv" style="color:${{cc}}">${{sc}}%</div></div>
  <div class="cinfo"><div class="cl" style="color:${{cc}};font-size:.85rem;font-weight:700">${{sc>=70?'ÉLEVÉ':sc>=40?'MOYEN':'FAIBLE'}}</div>
  <div class="cr" style="max-width:480px">${{confText}}</div></div>`;

// Comparaison des raisonnements — parse "**Label** : texte"
const rg = document.getElementById('reason-grid');
const reasonRaw = SEC['COMPARAISON DES RAISONNEMENTS']||'';
const reasonLines = reasonRaw.split('\\n').filter(Boolean);
keys.forEach(k=>{{
  const lbl = AO[k]?.label||k;
  const c   = col(k);
  // find line mentioning this agent label
  const line = reasonLines.find(l=>l.toLowerCase().includes(lbl.toLowerCase()))||'';
  const body = line.replace(/^[*][*][^*]+[*][*]\\s*:/,'').trim()||'—';
  const t = PA[k]||{{}};
  rg.innerHTML+=`<div class="rcard" style="border-left-color:${{c}}">
    <div class="rcard-head"><span class="dot" style="background:${{c}}"></span>
      <h4>${{lbl}}</h4>
      <span style="font-size:.7rem;color:var(--muted)">${{t.llm_calls||0}} LLM · ${{t.tool_calls||0}} tools</span>
    </div>
    <div class="rcard-body">${{body}}</div>
  </div>`;
}});

// Résultats clés
const synUl=(id,lines,cls)=>{{
  const el=document.getElementById(id);
  el.innerHTML=lines.length?lines.map(x=>`<li class="${{cls}}">${{x}}</li>`).join(''):'<li style="color:var(--muted)">—</li>';
}};
synUl('syn-resultats', getLines('RÉSULTATS CLÉS'), 'good');
synUl('syn-consensus',  getLines('CONSENSUS'),      'good');
synUl('syn-contradictions', getLines('CONTRADICTIONS'), 'danger');
synUl('syn-angles', getLines('ANGLES MORTS'), 'warn');

// Recherches recommandées
const recsEl = document.getElementById('syn-recs');
getLines('RECHERCHES RECOMMANDÉES').forEach(r=>{{
  const [topic,...rest]=r.split(':');
  recsEl.innerHTML+=`<div class="rec"><div class="rtopic">🔭 ${{topic.trim()}}</div>${{rest.length?`<div class="rratio">${{rest.join(':').trim()}}</div>`:''}}</div>`;
}});

document.getElementById('final').textContent = SEC['SYNTHÈSE FINALE']||SY.raw||'—';
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
