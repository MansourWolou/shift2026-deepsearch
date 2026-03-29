"""
factcheck.py
============
Fact-Check Engine — Multi-agent parallel research + thinking model synthesis.

Pipeline :
  1. Linkup search (deep) → récupère les sources web
  2. N modèles LLM analysent les résultats en parallèle (via OpenRouter)
  3. Un thinking model synthétise et produit le verdict

Usage:
    python factcheck.py "L'info à vérifier"
    python factcheck.py --agents openai,gemini,mistral "..."
    python factcheck.py --output result.json --verbose "..."
"""

import argparse
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

load_dotenv()
console = Console()
log = logging.getLogger("factcheck")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LINKUP_API_KEY = os.getenv("LINKUP_API_KEY", "")


# ──────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ──────────────────────────────────────────────────────────────────────────────


def normalize_query(raw: str) -> str:
    return " ".join(raw.strip().split())


def clean_text(text: str) -> str:
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_json(text: str) -> dict | None:
    for candidate in [
        text.strip(),
        *(m.group(1) for m in [re.search(r"```json\s*([\s\S]*?)\s*```", text, re.DOTALL)] if m),
        *(m.group(1) for m in [re.search(r"```\s*([\s\S]*?)\s*```", text, re.DOTALL)] if m),
        *(m.group(0) for m in [re.search(r"\{[\s\S]*\}", text, re.DOTALL)] if m),
    ]:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Pre-screening — la claim est-elle fact-checkable ?
# ──────────────────────────────────────────────────────────────────────────────

SCREENING_PROMPT_TPL = """Tu es un filtre de pré-analyse pour un moteur de fact-checking.

On te donne une phrase. Tu dois déterminer si c'est une AFFIRMATION FACTUELLE VÉRIFIABLE.

Réponds UNIQUEMENT avec un JSON valide, rien d'autre :
{{
  "checkable": true/false,
  "category": "factual" | "opinion" | "idiom" | "question" | "vague" | "subjective" | "future",
  "reason": "Explication courte en 1 phrase"
}}

Règles :
- "factual" = affirmation sur un fait mesurable/daté/vérifiable → checkable=true
- "opinion" = jugement de valeur, goût, préférence → checkable=false
- "idiom" = expression idiomatique, proverbe, métaphore → checkable=false
- "question" = question, pas une affirmation → checkable=false
- "vague" = trop flou pour vérifier (pas de sujet concret) → checkable=false
- "subjective" = dépend du point de vue, pas de réponse objective → checkable=false
- "future" = prédiction sur le futur, pas vérifiable maintenant → checkable=false

Exemples :
- "Macron a été élu en 2017" → factual, checkable=true
- "Le verre est à moitié plein" → idiom, checkable=false
- "Python est le meilleur langage" → opinion, checkable=false
- "Il va pleuvoir demain" → future, checkable=false
- "La tour Eiffel mesure 330m" → factual, checkable=true

PHRASE : {claim}"""


def prescreen_claim(claim: str) -> dict:
    """Quick check: is this claim fact-checkable? Uses a fast/cheap model."""
    llm = ChatOpenAI(
        model="google/gemini-2.5-flash",
        temperature=0.0,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
    response = llm.invoke([HumanMessage(content=SCREENING_PROMPT_TPL.format(claim=claim))])
    content = response.content
    if isinstance(content, list):
        content = "".join(c["text"] if isinstance(c, dict) else str(c) for c in content)

    parsed = extract_json(content)
    if parsed:
        return parsed
    return {"checkable": True, "category": "unknown", "reason": "Parsing failed, proceeding anyway"}


# ──────────────────────────────────────────────────────────────────────────────
# Linkup Search
# ──────────────────────────────────────────────────────────────────────────────


def linkup_search(query: str, depth: str = "deep") -> dict:
    """
    Appelle Linkup pour récupérer des sources web.
    Retourne {"answer": str, "sources": [{"name", "url", "snippet"}]}.
    """
    resp = requests.post(
        "https://api.linkup.so/v1/search",
        headers={"Authorization": f"Bearer {LINKUP_API_KEY}"},
        json={
            "q": query,
            "depth": depth,
            "outputType": "sourcedAnswer",
            "includeInlineCitations": True,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


# ──────────────────────────────────────────────────────────────────────────────
# Agent Registry — chaque agent = un LLM qui analyse les résultats Linkup
# ──────────────────────────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """Tu es un agent de fact-checking. On te donne une affirmation à vérifier
et des résultats de recherche web.

AFFIRMATION : {claim}

RÉSULTATS DE RECHERCHE :
{search_answer}

SOURCES :
{sources_text}

Analyse ces résultats et fournis :
1. Ce que les sources confirment ou infirment
2. La qualité et la fiabilité des sources
3. Les nuances ou contradictions trouvées
4. Ton évaluation préliminaire (vrai, faux, partiellement vrai, invérifiable, trompeur)

Réponds en markdown structuré avec les sources citées."""

# Prompts spécialisés — chaque agent a un angle d'analyse différent
SPECIALIZED_PROMPTS: dict[str, str] = {
    "verificateur": """Tu es un VÉRIFICATEUR DE FAITS rigoureux. Tu te concentres UNIQUEMENT
sur les données factuelles : dates, chiffres, noms, lieux, statistiques.

AFFIRMATION : {claim}

RÉSULTATS DE RECHERCHE :
{search_answer}

SOURCES :
{sources_text}

Ta mission :
1. Extraire chaque fait vérifiable de l'affirmation (dates, chiffres, noms, etc.)
2. Pour chaque fait, indiquer si les sources le confirment ou l'infirment, avec citation
3. Signaler toute donnée factuelle absente des sources (ni confirmée, ni infirmée)
4. Verdict factuel : vrai, faux, partiellement vrai, invérifiable

Sois précis et factuel. Pas d'interprétation, pas d'opinion.""",
    "avocat_diable": """Tu es un AVOCAT DU DIABLE. Ton rôle est de chercher activement
à INFIRMER l'affirmation. Tu dois trouver les failles, les contre-exemples, les biais.

AFFIRMATION : {claim}

RÉSULTATS DE RECHERCHE :
{search_answer}

SOURCES :
{sources_text}

Ta mission :
1. Chercher dans les sources tout ce qui contredit ou nuance l'affirmation
2. Identifier les biais possibles des sources (parti pris, date ancienne, source partiale)
3. Proposer des interprétations alternatives ou des contre-arguments
4. Signaler ce que l'affirmation omet ou simplifie
5. Ton évaluation en partant du principe que l'affirmation est fausse — qu'est-ce qui manque pour la confirmer ?

Sois critique et exigeant. Ton job est de challenger, pas de confirmer.""",
    "analyste_sources": """Tu es un ANALYSTE DE SOURCES spécialisé en fiabilité de l'information.
Tu évalues la QUALITÉ des preuves, pas leur contenu.

AFFIRMATION : {claim}

RÉSULTATS DE RECHERCHE :
{search_answer}

SOURCES :
{sources_text}

Ta mission :
1. Classer chaque source par type : institutionnelle, académique, presse, blog, opinion, wiki
2. Évaluer la fiabilité de chaque source (date, auteur, biais connu, réputation)
3. Identifier les sources primaires vs secondaires vs tertiaires
4. Détecter les circular reporting (sources qui se citent mutuellement)
5. Donner un score de confiance global basé sur la QUALITÉ des sources, pas leur quantité

Sois méthodique. Un fait cité par 10 blogs vaut moins qu'un fait cité par 1 source primaire.""",
    "contextualiste": """Tu es un CONTEXTUALISTE expert. Tu apportes le contexte historique,
géographique, politique et temporel que les autres analyses pourraient manquer.

AFFIRMATION : {claim}

RÉSULTATS DE RECHERCHE :
{search_answer}

SOURCES :
{sources_text}

Ta mission :
1. Replacer l'affirmation dans son contexte historique et temporel
2. Identifier si l'affirmation était vraie à une époque mais plus maintenant (ou l'inverse)
3. Signaler les nuances géographiques ou culturelles
4. Détecter les simplifications abusives ou les généralisations
5. Identifier les "blind spots" — ce que personne ne mentionne mais qui est important

Apporte la profondeur et la nuance. Les faits bruts ne suffisent pas, le contexte change tout.""",
}

# Mapping : quel agent reçoit quel rôle spécialisé (en mode --specialized)
# Les agents sont assignés dans l'ordre de la liste fournie
SPECIALIZED_ROLES = ["verificateur", "avocat_diable", "analyste_sources", "contextualiste"]


def _invoke_agent(model: str, claim: str, search_results: dict, prompt: str | None = None) -> str:
    """Un LLM analyse les résultats de recherche Linkup."""
    llm = ChatOpenAI(
        model=model,
        temperature=0.1,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

    sources_text = "\n".join(
        f"- [{s.get('name', 'N/A')}]({s.get('url', '')}) : {s.get('snippet', '')}"
        for s in search_results.get("sources", [])
    )

    template = prompt or ANALYSIS_PROMPT
    messages = [
        HumanMessage(
            content=template.format(
                claim=claim,
                search_answer=search_results.get("answer", "Aucune réponse"),
                sources_text=sources_text or "Aucune source",
            )
        )
    ]

    response = llm.invoke(messages)
    content = response.content
    if isinstance(content, list):
        return "".join(c["text"] if isinstance(c, dict) else str(c) for c in content)
    return content


AGENT_REGISTRY: dict[str, dict] = {
    # --- Frontier ---
    "claude": {
        "label": "Claude Sonnet 4.6",
        "model": "google/gemini-2.5-flash",
    },
    "openai": {
        "label": "GPT-5.4",
        "model": "openai/gpt-5.4",
    },
    "gemini": {
        "label": "Gemini 2.5 Pro",
        "model": "google/gemini-2.5-pro",
    },
    "grok": {
        "label": "Grok 4",
        "model": "x-ai/grok-4",
    },
    # --- Bon rapport qualité/prix ---
    "mistral": {
        "label": "Mistral Small 3.2",
        "model": "mistralai/mistral-small-3.2-24b-instruct",
    },
    "deepseek": {
        "label": "DeepSeek V3.2",
        "model": "deepseek/deepseek-v3.2",
    },
    "qwen": {
        "label": "Qwen3 Max",
        "model": "qwen/qwen3-max",
    },
    # --- Rapides/pas cher ---
    "gemini-flash": {
        "label": "Gemini 2.5 Flash",
        "model": "google/gemini-2.5-flash",
    },
    "gpt-mini": {
        "label": "GPT-5 Mini",
        "model": "openai/gpt-5-mini",
    },
    "o4-mini": {
        "label": "o4-mini",
        "model": "openai/o4-mini",
    },
}

DEFAULT_AGENTS = ["gemini-flash", "mistral", "o4-mini"]

# Mode presets: fast (prod-like), default, thorough (dev/demo deep analysis)
MODE_PRESETS = {
    "fast": {
        "agents": ["gemini-flash", "o4-mini"],
        "max_workers": 2,
        "search_depth": "standard",
        "description": "Rapide — 2 agents légers, recherche standard (prod-like)",
    },
    "default": {
        "agents": DEFAULT_AGENTS,
        "max_workers": None,  # = len(agents)
        "search_depth": "deep",
        "description": "Équilibré — 3 agents, recherche deep",
    },
    "thorough": {
        "agents": ["claude", "openai", "gemini", "mistral", "grok"],
        "max_workers": None,
        "search_depth": "deep",
        "description": "Complet — 5 agents frontier, recherche deep (démo)",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Recherche + analyse parallèle
# ──────────────────────────────────────────────────────────────────────────────


def run_research_parallel(
    claim: str,
    search_results: dict,
    agents: list[str],
    max_workers: int | None = None,
    specialized: bool = False,
) -> dict[str, dict]:
    """
    Lance N agents LLM en parallèle, chacun analysant les mêmes résultats Linkup.
    Si specialized=True, chaque agent reçoit un prompt spécialisé différent.
    """
    if max_workers is None:
        max_workers = len(agents)

    # Assign roles: cycle through specialized roles if more agents than roles
    agent_prompts: dict[str, str | None] = {}
    agent_roles: dict[str, str] = {}
    if specialized:
        for i, name in enumerate(agents):
            role = SPECIALIZED_ROLES[i % len(SPECIALIZED_ROLES)]
            agent_prompts[name] = SPECIALIZED_PROMPTS[role]
            agent_roles[name] = role
        log.info("Specialized mode — roles: %s", agent_roles)
    else:
        for name in agents:
            agent_prompts[name] = None
            agent_roles[name] = "generic"

    log.info("Parallelism: %d agents, %d workers", len(agents), max_workers)
    for a in agents:
        log.debug(
            "  → %s (%s) [%s]",
            AGENT_REGISTRY[a]["label"],
            AGENT_REGISTRY[a]["model"],
            agent_roles[a],
        )

    results: dict[str, dict] = {}

    def _run_one(agent_name: str) -> tuple[str, str, float, str | None]:
        t0 = time.perf_counter()
        log.debug("[%s] started (%s)", agent_name, agent_roles[agent_name])
        try:
            model = AGENT_REGISTRY[agent_name]["model"]
            raw = _invoke_agent(model, claim, search_results, prompt=agent_prompts[agent_name])
            dt = time.perf_counter() - t0
            log.info("[%s] done in %.1fs (%d chars)", agent_name, dt, len(raw))
            return (agent_name, raw, dt, None)
        except Exception as e:
            dt = time.perf_counter() - t0
            log.error("[%s] failed after %.1fs: %s", agent_name, dt, e)
            return (agent_name, "", dt, str(e))

    status_map: dict[str, str] = {name: "[yellow]en cours…[/yellow]" for name in agents}

    def _build_table() -> Table:
        title = "Analyse en cours (spécialisée)" if specialized else "Analyse en cours"
        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("Agent", style="bold")
        if specialized:
            table.add_column("Rôle", style="dim")
        table.add_column("Statut")
        table.add_column("Durée", justify="right")
        for name in agents:
            duration = results[name]["duration_s"] if name in results else ""
            dur_str = f"{duration:.1f}s" if duration else "…"
            row = [AGENT_REGISTRY[name]["label"]]
            if specialized:
                row.append(agent_roles[name])
            row.extend([status_map[name], dur_str])
            table.add_row(*row)
        return table

    with (
        Live(_build_table(), console=console, refresh_per_second=2) as live,
        ThreadPoolExecutor(max_workers=max_workers) as pool,
    ):
        futures = {pool.submit(_run_one, name): name for name in agents}
        for future in as_completed(futures):
            name, raw, duration, error = future.result()
            results[name] = {
                "label": AGENT_REGISTRY[name]["label"],
                "role": agent_roles[name],
                "raw": raw,
                "duration_s": round(duration, 2),
                "error": error,
            }
            if error:
                status_map[name] = f"[red]erreur: {error[:40]}[/red]"
            else:
                status_map[name] = "[green]terminé[/green]"
            live.update(_build_table())

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Synthèse par thinking model
# ──────────────────────────────────────────────────────────────────────────────

VERDICT_SCHEMA = {
    "claim": "The original claim being fact-checked",
    "verdict": "TRUE | FALSE | PARTIALLY TRUE | UNVERIFIABLE | MISLEADING",
    "confidence": 0.85,
    "summary": "2-3 sentence explanation of the verdict",
    "evidence_for": [
        {"point": "Evidence supporting the claim", "sources": ["url1"], "agents": ["agent_name"]}
    ],
    "evidence_against": [
        {"point": "Evidence contradicting the claim", "sources": ["url1"], "agents": ["agent_name"]}
    ],
    "consensus": ["Points all/most agents agree on"],
    "disagreements": [
        {
            "topic": "What they disagree about",
            "positions": {"agent1": "position A", "agent2": "position B"},
        }
    ],
    "blind_spots": ["Topics or angles no agent covered"],
    "nuances": ["Important caveats or context"],
}

SYNTHESIS_SYSTEM_PROMPT = """You are an expert fact-checker and meta-analyst.

You receive analysis reports from {n_agents} independent AI agents,
each using a different LLM, all analyzing the same web search results about a claim.

Your task is to produce a rigorous fact-check verdict by:
1. CROSS-REFERENCING: Identify points that multiple agents corroborate vs. contradict.
2. SOURCE EVALUATION: Assess the quality and diversity of sources cited.
3. AGREEMENT ANALYSIS: Note where agents agree (high confidence) vs. disagree (flag for user).
4. BIAS DETECTION: Flag if all agents share the same blind spot or framing.
5. VERDICT: Deliver a clear fact-check verdict with calibrated confidence.

You MUST respond with a single valid JSON object matching this exact schema:
{output_schema}

Do not include any text outside the JSON object."""

SYNTHESIS_USER_PROMPT = """CLAIM TO FACT-CHECK: {query}

{agent_reports}

Analyze all {n_agents} reports above. Cross-reference their findings,
identify agreements and contradictions, evaluate source quality,
and produce your fact-check verdict as JSON."""


# Thinking models pour la synthese multi-agent
THINK_REGISTRY: dict[str, dict] = {
    "gemini-flash": {
        "label": "Gemini 2.5 Flash",
        "model": "google/gemini-2.5-flash",
    },
    "o4-mini": {
        "label": "o4-mini",
        "model": "openai/o4-mini",
    },
    "claude": {
        "label": "Claude Sonnet 4.6",
        "model": "anthropic/claude-sonnet-4-6",
    },
    "gemini-pro": {
        "label": "Gemini 2.5 Pro",
        "model": "google/gemini-2.5-pro",
    },
    "gpt": {
        "label": "GPT-5.4",
        "model": "openai/gpt-5.4",
    },
}

THINK_PRESETS: dict[int, list[str]] = {
    1: ["gemini-flash"],
    2: ["gemini-flash", "o4-mini"],
    3: ["gemini-flash", "o4-mini", "claude"],
    5: ["gemini-flash", "o4-mini", "claude", "gemini-pro", "gpt"],
}


def _build_synthesis_messages(query: str, research_results: dict[str, dict]) -> list:
    """Build the messages for a synthesis call (shared by single and multi)."""
    reports = []
    for name, data in research_results.items():
        role = data.get("role", "generic")
        role_tag = f" [role: {role}]" if role != "generic" else ""
        if data["error"]:
            reports.append(
                f"--- REPORT FROM {data['label']} ({name}){role_tag} ---\n"
                f"[FAILED: {data['error']}]\n"
                f"--- END REPORT ---"
            )
        else:
            text = clean_text(data["raw"])[:4000]
            reports.append(
                f"--- REPORT FROM {data['label']} ({name}){role_tag} ---\n"
                f"{text}\n"
                f"--- END REPORT ---"
            )

    agent_reports = "\n\n".join(reports)
    n_agents = len(research_results)

    return [
        SystemMessage(
            content=SYNTHESIS_SYSTEM_PROMPT.format(
                n_agents=n_agents,
                output_schema=json.dumps(VERDICT_SCHEMA, indent=2),
            )
        ),
        HumanMessage(
            content=SYNTHESIS_USER_PROMPT.format(
                query=query,
                agent_reports=agent_reports,
                n_agents=n_agents,
            )
        ),
    ]


def _call_synthesis(model: str, messages: list) -> dict:
    """Call a single synthesis model and return parsed verdict."""
    llm = ChatOpenAI(
        model=model,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
    response = llm.invoke(messages)
    content = response.content
    if isinstance(content, list):
        content = next(
            (
                block["text"]
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ),
            str(content),
        )
    parsed = extract_json(content)
    if parsed:
        return parsed
    return {
        "verdict": "UNVERIFIABLE",
        "confidence": 0.0,
        "summary": "JSON invalide.",
        "raw_response": content[:2000],
    }


def synthesize_verdict(
    query: str,
    research_results: dict[str, dict],
    model: str = "google/gemini-2.5-flash",
) -> dict:
    """Single-model synthesis (--think 1 or default)."""
    messages = _build_synthesis_messages(query, research_results)

    console.print("\n[bold magenta]Synthèse en cours…[/bold magenta] " f"({model})\n")

    verdict = _call_synthesis(model, messages)
    if "claim" not in verdict:
        verdict["claim"] = query
    return verdict


def synthesize_multi(
    query: str,
    research_results: dict[str, dict],
    think_models: list[str],
) -> dict:
    """Multi-model synthesis: N thinking models vote in parallel, then aggregate."""
    messages = _build_synthesis_messages(query, research_results)
    n = len(think_models)

    labels = [THINK_REGISTRY[k]["label"] for k in think_models]
    console.print(
        f"\n[bold magenta]Synthèse multi-think ({n} modèles)…[/bold magenta] "
        f"[dim]{', '.join(labels)}[/dim]\n"
    )

    verdicts: dict[str, dict] = {}
    timings: dict[str, float] = {}

    def _run_think(key: str) -> tuple[str, dict, float]:
        t0 = time.perf_counter()
        model = THINK_REGISTRY[key]["model"]
        result = _call_synthesis(model, messages)
        dt = time.perf_counter() - t0
        log.info("[think:%s] done in %.1fs → %s", key, dt, result.get("verdict", "?"))
        return key, result, dt

    status_map: dict[str, str] = {k: "[yellow]en cours…[/yellow]" for k in think_models}

    def _build_table() -> Table:
        table = Table(title="Synthèse multi-think", show_header=True, header_style="bold magenta")
        table.add_column("Thinker", style="bold")
        table.add_column("Statut")
        table.add_column("Verdict")
        table.add_column("Confiance", justify="right")
        table.add_column("Durée", justify="right")
        for k in think_models:
            if k in verdicts:
                v = verdicts[k]
                vtext = v.get("verdict", "?")
                color = VERDICT_COLORS.get(vtext, "white")
                conf = v.get("confidence", 0)
                conf_str = f"{conf:.0%}" if isinstance(conf, int | float) and conf else ""
                dur_str = f"{timings[k]:.1f}s"
                table.add_row(
                    THINK_REGISTRY[k]["label"],
                    status_map[k],
                    f"[{color}]{vtext}[/{color}]",
                    conf_str,
                    dur_str,
                )
            else:
                table.add_row(THINK_REGISTRY[k]["label"], status_map[k], "", "", "…")
        return table

    with (
        Live(_build_table(), console=console, refresh_per_second=2) as live,
        ThreadPoolExecutor(max_workers=n) as pool,
    ):
        futures = {pool.submit(_run_think, k): k for k in think_models}
        for future in as_completed(futures):
            key, result, dt = future.result()
            verdicts[key] = result
            timings[key] = round(dt, 2)
            status_map[key] = "[green]terminé[/green]"
            live.update(_build_table())

    # Aggregate: majority vote on verdict, average confidence, merge evidence
    verdict_counts: dict[str, int] = {}
    total_conf = 0.0
    conf_count = 0
    all_evidence_for = []
    all_evidence_against = []
    all_nuances = []

    for v in verdicts.values():
        vt = v.get("verdict", "UNVERIFIABLE")
        verdict_counts[vt] = verdict_counts.get(vt, 0) + 1
        c = v.get("confidence", 0)
        if isinstance(c, int | float) and c:
            total_conf += c
            conf_count += 1
        all_evidence_for.extend(v.get("evidence_for", []))
        all_evidence_against.extend(v.get("evidence_against", []))
        all_nuances.extend(v.get("nuances", []))

    # Winner = most votes, tie-break by first in VERDICT order
    verdict_order = ["TRUE", "FALSE", "PARTIALLY TRUE", "MISLEADING", "UNVERIFIABLE"]
    winner = max(
        verdict_counts,
        key=lambda vt: (verdict_counts[vt], -verdict_order.index(vt) if vt in verdict_order else 0),
    )
    avg_conf = total_conf / conf_count if conf_count else 0

    # Deduplicate evidence (by point text)
    def _dedup(items: list) -> list:
        seen = set()
        out = []
        for item in items:
            key = item.get("point", item) if isinstance(item, dict) else item
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out

    # Build think consensus detail
    think_detail = []
    for k in think_models:
        v = verdicts[k]
        think_detail.append(
            {
                "model": THINK_REGISTRY[k]["label"],
                "verdict": v.get("verdict"),
                "confidence": v.get("confidence"),
                "duration_s": timings[k],
            }
        )

    # Check if unanimous
    unanimous = len(verdict_counts) == 1

    summary_parts = []
    if unanimous:
        summary_parts.append(f"Les {n} modèles de synthèse sont unanimes : {winner}.")
    else:
        votes = ", ".join(
            f"{vt} ({ct}x)" for vt, ct in sorted(verdict_counts.items(), key=lambda x: -x[1])
        )
        summary_parts.append(f"Vote des {n} thinkers : {votes}. Verdict majoritaire : {winner}.")

    # Pick best summary from the verdicts that match the winner
    for v in verdicts.values():
        if v.get("verdict") == winner and v.get("summary"):
            summary_parts.append(v["summary"])
            break

    return {
        "claim": query,
        "verdict": winner,
        "confidence": round(avg_conf, 2),
        "summary": " ".join(summary_parts),
        "evidence_for": _dedup(all_evidence_for),
        "evidence_against": _dedup(all_evidence_against),
        "consensus": [],
        "disagreements": [],
        "blind_spots": [],
        "nuances": _dedup(all_nuances),
        "think_detail": think_detail,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Fonction principale
# ──────────────────────────────────────────────────────────────────────────────


def factcheck(
    query: str,
    agents: list[str] | None = None,
    synthesis_model: str = "google/gemini-2.5-flash",
    verbose: bool = False,
    max_workers: int | None = None,
    search_depth: str = "deep",
    skip_screening: bool = False,
    specialized: bool = False,
    think: int = 1,
) -> dict:
    query = normalize_query(query)
    if agents is None:
        agents = list(DEFAULT_AGENTS)

    for name in agents:
        if name not in AGENT_REGISTRY:
            raise ValueError(f"Agent inconnu : {name}. Disponibles : {list(AGENT_REGISTRY.keys())}")

    console.print(
        Panel(
            f"[bold]Fact-Check Engine[/bold]\n"
            f"[white]{query}[/white]\n"
            f"[dim]Agents : {', '.join(agents)} | Synthèse : {synthesis_model}[/dim]",
            border_style="cyan",
        )
    )

    t_total = time.perf_counter()

    # Phase 0 — Pre-screening
    screening = None
    if not skip_screening:
        console.print("\n[bold yellow]Pre-screening…[/bold yellow]")
        t_screen = time.perf_counter()
        screening = prescreen_claim(query)
        screen_duration = time.perf_counter() - t_screen
        log.info("Pre-screening: %s (%.1fs)", screening, screen_duration)

        if not screening.get("checkable", True):
            category = screening.get("category", "unknown")
            reason = screening.get("reason", "")
            console.print(
                f"[yellow]⚠ Claim non fact-checkable ({category})[/yellow]\n"
                f"[dim]{reason}[/dim]\n"
            )
            total_duration = time.perf_counter() - t_total
            return {
                "query": query,
                "timestamp": datetime.now(UTC).isoformat(),
                "config": {
                    "research_agents": agents,
                    "synthesis_model": synthesis_model,
                },
                "screening": screening,
                "timing": {
                    "total_s": round(total_duration, 2),
                    "screening_s": round(screen_duration, 2),
                    "search_s": 0,
                    "analysis_s": 0,
                    "synthesis_s": 0,
                    "per_agent": {},
                },
                "verdict": {
                    "claim": query,
                    "verdict": "NOT_CHECKABLE",
                    "confidence": 0,
                    "summary": reason,
                    "category": category,
                    "evidence_for": [],
                    "evidence_against": [],
                    "consensus": [],
                    "disagreements": [],
                    "blind_spots": [],
                    "nuances": [],
                },
            }

        console.print(
            f"[green]✓ Claim fact-checkable ({screening.get('category', '')})[/green] "
            f"({screen_duration:.1f}s)\n"
        )

    # Phase 1 — Recherche Linkup
    console.print(f"[bold blue]Recherche Linkup ({search_depth})…[/bold blue]")
    t_search = time.perf_counter()
    search_results = linkup_search(query, depth=search_depth)
    search_duration = time.perf_counter() - t_search

    n_sources = len(search_results.get("sources", []))
    console.print(f"[green]{n_sources} sources trouvées[/green] ({search_duration:.1f}s)\n")

    # Phase 2 — Analyse parallèle par N modèles
    t_analysis = time.perf_counter()
    agent_results = run_research_parallel(
        query, search_results, agents, max_workers=max_workers, specialized=specialized
    )
    analysis_duration = time.perf_counter() - t_analysis

    successful = sum(1 for r in agent_results.values() if not r["error"])
    console.print(f"\n[green]{successful}/{len(agents)} agents terminés avec succès[/green]")

    # Phase 3 — Synthèse (1 ou N thinking models)
    t_synthesis = time.perf_counter()
    if think > 1 and think in THINK_PRESETS:
        verdict = synthesize_multi(query, agent_results, THINK_PRESETS[think])
    else:
        verdict = synthesize_verdict(query, agent_results, synthesis_model)
    synthesis_duration = time.perf_counter() - t_synthesis

    total_duration = time.perf_counter() - t_total

    output = {
        "query": query,
        "timestamp": datetime.now(UTC).isoformat(),
        "config": {
            "research_agents": agents,
            "synthesis_model": synthesis_model,
        },
        "timing": {
            "total_s": round(total_duration, 2),
            "search_s": round(search_duration, 2),
            "analysis_s": round(analysis_duration, 2),
            "synthesis_s": round(synthesis_duration, 2),
            "per_agent": {name: data["duration_s"] for name, data in agent_results.items()},
        },
        "verdict": verdict,
    }

    if screening:
        output["screening"] = screening

    if verbose:
        output["search_results"] = search_results
        output["raw_analysis"] = {
            name: {"raw": data["raw"], "duration_s": data["duration_s"], "error": data["error"]}
            for name, data in agent_results.items()
        }

    return output


# ──────────────────────────────────────────────────────────────────────────────
# Affichage du verdict
# ──────────────────────────────────────────────────────────────────────────────

VERDICT_COLORS = {
    "TRUE": "green",
    "FALSE": "red",
    "PARTIALLY TRUE": "yellow",
    "UNVERIFIABLE": "dim",
    "MISLEADING": "red",
    "NOT_CHECKABLE": "yellow",
}


def display_verdict(output: dict) -> None:
    v = output["verdict"]
    verdict_text = v.get("verdict", "UNKNOWN")
    color = VERDICT_COLORS.get(verdict_text, "white")
    confidence = v.get("confidence", 0)

    console.print(
        Panel(
            f"[bold {color}]{verdict_text}[/bold {color}]  "
            f"(confiance : {confidence:.0%})\n\n"
            f"{v.get('summary', '')}",
            title="[bold]Verdict[/bold]",
            border_style=color,
        )
    )

    if v.get("evidence_for"):
        console.print("\n[green]Evidence FOR:[/green]")
        for e in v["evidence_for"]:
            console.print(f"  • {e.get('point', '')}")

    if v.get("evidence_against"):
        console.print("\n[red]Evidence AGAINST:[/red]")
        for e in v["evidence_against"]:
            console.print(f"  • {e.get('point', '')}")

    if v.get("nuances"):
        console.print("\n[yellow]Nuances:[/yellow]")
        for n in v["nuances"]:
            console.print(f"  • {n}")

    # Timeline waterfall
    t = output["timing"]
    total = t["total_s"] or 1
    bar_width = 60

    phases = [
        ("Linkup", t["search_s"], "blue"),
        ("Agents", t["analysis_s"], "cyan"),
        ("Synthèse", t["synthesis_s"], "magenta"),
    ]

    console.print("\n[bold]Pipeline Timeline[/bold]")
    for label, dur, color in phases:
        pct = dur / total
        filled = max(1, round(pct * bar_width))
        bar = "█" * filled + "░" * (bar_width - filled)
        console.print(
            f"  [{color}]{label:>8}[/{color}] [{color}]{bar}[/{color}] " f"{dur:.1f}s ({pct:.0%})"
        )

    # Per-agent detail inside the Agents phase
    per_agent = t.get("per_agent", {})
    if per_agent:
        max_agent = max(per_agent.values()) or 1
        console.print(
            f"\n  [dim]Détail agents (parallèle, wall-clock = {t['analysis_s']:.1f}s) :[/dim]"
        )
        for name, dur in sorted(per_agent.items(), key=lambda x: x[1]):
            label = AGENT_REGISTRY.get(name, {}).get("label", name)
            agent_pct = dur / max_agent
            filled = max(1, round(agent_pct * 40))
            bar = "█" * filled + "░" * (40 - filled)
            console.print(f"    {label:>20} [cyan]{bar}[/cyan] {dur:.1f}s")

    console.print(
        f"\n[dim]Total: {t['total_s']:.1f}s | "
        f"Linkup: {t['search_s']:.1f}s ({t['search_s']/total:.0%}) | "
        f"Agents: {t['analysis_s']:.1f}s ({t['analysis_s']/total:.0%}) | "
        f"Synthèse: {t['synthesis_s']:.1f}s ({t['synthesis_s']/total:.0%})[/dim]"
    )


# ──────────────────────────────────────────────────────────────────────────────
# HTML Report (--html, dev/demo mode)
# ──────────────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fact-Check — {query_escaped}</title>
<style>
  :root {{ --green: #22c55e; --red: #ef4444; --yellow: #eab308; --gray: #6b7280; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem; max-width: 960px; margin: auto; }}
  h1 {{ font-size: 1.4rem; margin-bottom: .5rem; }}
  .claim {{ background: #1e293b; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; font-size: 1.1rem; border-left: 4px solid #3b82f6; }}
  .verdict-box {{ background: #1e293b; padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem; text-align: center; }}
  .verdict {{ font-size: 2rem; font-weight: 800; }}
  .verdict.TRUE {{ color: var(--green); }}
  .verdict.FALSE {{ color: var(--red); }}
  .verdict.PARTIALLY {{ color: var(--yellow); }}
  .verdict.MISLEADING {{ color: var(--red); }}
  .verdict.UNVERIFIABLE {{ color: var(--gray); }}
  .confidence {{ font-size: 1.1rem; color: #94a3b8; margin-top: .3rem; }}
  .summary {{ margin-top: .8rem; line-height: 1.6; }}
  .section {{ background: #1e293b; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }}
  .section h2 {{ font-size: 1rem; margin-bottom: .6rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; }}
  ul {{ padding-left: 1.2rem; }}
  li {{ margin-bottom: .4rem; line-height: 1.5; }}
  .timing {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: .8rem; }}
  .timing-card {{ background: #1e293b; padding: .8rem; border-radius: 8px; text-align: center; }}
  .timing-card .val {{ font-size: 1.4rem; font-weight: 700; color: #3b82f6; }}
  .timing-card .lbl {{ font-size: .75rem; color: #64748b; margin-top: .2rem; }}
  .timeline {{ background: #1e293b; padding: 1.2rem; border-radius: 8px; margin-bottom: 1.5rem; }}
  .timeline h2 {{ font-size: 1rem; margin-bottom: 1rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; }}
  .tl-row {{ display: flex; align-items: center; margin-bottom: .6rem; gap: .8rem; }}
  .tl-label {{ width: 80px; text-align: right; font-weight: 600; font-size: .85rem; flex-shrink: 0; }}
  .tl-track {{ flex: 1; background: #0f172a; border-radius: 4px; height: 28px; position: relative; overflow: hidden; }}
  .tl-bar {{ height: 100%; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: .75rem; font-weight: 600; color: white; min-width: 40px; }}
  .tl-bar.linkup {{ background: linear-gradient(90deg, #3b82f6, #2563eb); }}
  .tl-bar.agents {{ background: linear-gradient(90deg, #06b6d4, #0891b2); }}
  .tl-bar.synthesis {{ background: linear-gradient(90deg, #a855f7, #7c3aed); }}
  .tl-meta {{ width: 90px; text-align: right; font-size: .8rem; color: #94a3b8; font-variant-numeric: tabular-nums; flex-shrink: 0; }}
  .agents {{ margin-top: 1.5rem; }}
  .agent-row {{ display: flex; justify-content: space-between; align-items: center; padding: .5rem .8rem; border-radius: 6px; margin-bottom: .4rem; background: #0f172a; }}
  .agent-name {{ font-weight: 600; }}
  .agent-time {{ color: #3b82f6; font-variant-numeric: tabular-nums; }}
  .agent-bar {{ height: 4px; background: #3b82f6; border-radius: 2px; margin-top: .3rem; transition: width .3s; }}
  .footer {{ text-align: center; color: #475569; font-size: .75rem; margin-top: 2rem; }}
</style>
</head>
<body>
<h1>Fact-Check Report</h1>
<div class="claim">{query_escaped}</div>

<div class="verdict-box">
  <div class="verdict {verdict_class}">{verdict_text}</div>
  <div class="confidence">Confiance : {confidence}%</div>
  <div class="summary">{summary}</div>
</div>

{evidence_for_html}
{evidence_against_html}
{nuances_html}

<div class="timeline">
  <h2>Pipeline Timeline</h2>
  <div class="tl-row">
    <div class="tl-label" style="color:#3b82f6">Linkup</div>
    <div class="tl-track"><div class="tl-bar linkup" style="width:{search_pct}%">{search_s}s</div></div>
    <div class="tl-meta">{search_pct}%</div>
  </div>
  <div class="tl-row">
    <div class="tl-label" style="color:#06b6d4">Agents</div>
    <div class="tl-track"><div class="tl-bar agents" style="width:{analysis_pct}%">{analysis_s}s</div></div>
    <div class="tl-meta">{analysis_pct}%</div>
  </div>
  <div class="tl-row">
    <div class="tl-label" style="color:#a855f7">Synthèse</div>
    <div class="tl-track"><div class="tl-bar synthesis" style="width:{synthesis_pct}%">{synthesis_s}s</div></div>
    <div class="tl-meta">{synthesis_pct}%</div>
  </div>
</div>

<div class="section agents">
  <h2>Agents — Parallélisme</h2>
  {agents_html}
</div>

<div class="timing">
  <div class="timing-card"><div class="val">{total_s}s</div><div class="lbl">Total</div></div>
  <div class="timing-card"><div class="val">{search_s}s</div><div class="lbl">Linkup</div></div>
  <div class="timing-card"><div class="val">{analysis_s}s</div><div class="lbl">Agents</div></div>
  <div class="timing-card"><div class="val">{synthesis_s}s</div><div class="lbl">Synthèse</div></div>
</div>

<div class="footer">Généré le {timestamp} — Fact-Check Engine</div>
</body>
</html>
"""


def generate_html_report(output: dict, path: str) -> str:
    v = output["verdict"]
    t = output["timing"]

    verdict_text = v.get("verdict", "UNKNOWN")
    verdict_class = verdict_text.split()[0]  # "PARTIALLY TRUE" → "PARTIALLY"
    confidence = (
        round(v.get("confidence", 0) * 100)
        if v.get("confidence", 0) <= 1
        else round(v.get("confidence", 0))
    )

    # Agent bars — scaled relative to slowest
    per_agent = t.get("per_agent", {})
    max_dur = max(per_agent.values()) if per_agent else 1
    agents_rows = []
    for name, dur in sorted(per_agent.items(), key=lambda x: x[1]):
        label = AGENT_REGISTRY.get(name, {}).get("label", name)
        pct = (dur / max_dur) * 100
        agents_rows.append(
            f'<div class="agent-row">'
            f'<span class="agent-name">{label}</span>'
            f'<span class="agent-time">{dur:.1f}s</span>'
            f"</div>"
            f'<div class="agent-bar" style="width:{pct:.0f}%"></div>'
        )

    def _list_section(title: str, items: list, color: str) -> str:
        if not items:
            return ""
        lis = "".join(
            f"<li>{_esc(i.get('point', i) if isinstance(i, dict) else i)}</li>" for i in items
        )
        return f'<div class="section"><h2 style="color:{color}">{title}</h2><ul>{lis}</ul></div>'

    def _esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    total = t["total_s"] or 1
    search_pct = round(t["search_s"] / total * 100)
    analysis_pct = round(t["analysis_s"] / total * 100)
    synthesis_pct = round(t["synthesis_s"] / total * 100)

    html = HTML_TEMPLATE.format(
        query_escaped=_esc(output["query"]),
        verdict_text=verdict_text,
        verdict_class=verdict_class,
        confidence=confidence,
        summary=_esc(v.get("summary", "")),
        evidence_for_html=_list_section("Evidence FOR", v.get("evidence_for", []), "var(--green)"),
        evidence_against_html=_list_section(
            "Evidence AGAINST", v.get("evidence_against", []), "var(--red)"
        ),
        nuances_html=_list_section("Nuances", v.get("nuances", []), "var(--yellow)"),
        agents_html="\n".join(agents_rows),
        total_s=f"{t['total_s']:.1f}",
        search_s=f"{t['search_s']:.1f}",
        analysis_s=f"{t['analysis_s']:.1f}",
        synthesis_s=f"{t['synthesis_s']:.1f}",
        search_pct=search_pct,
        analysis_pct=analysis_pct,
        synthesis_pct=synthesis_pct,
        timestamp=output.get("timestamp", ""),
    )

    Path(path).write_text(html, encoding="utf-8")
    return path


REPORTS_DIR = Path("reports")

INDEX_TEMPLATE = """\
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fact-Check — Index des rapports</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem; max-width: 960px; margin: auto; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 1.5rem; }}
  .count {{ color: #64748b; font-size: .9rem; margin-bottom: 1.5rem; }}
  .report {{ background: #1e293b; padding: 1rem 1.2rem; border-radius: 8px; margin-bottom: .6rem; display: flex; justify-content: space-between; align-items: center; text-decoration: none; color: inherit; transition: background .15s; }}
  .report:hover {{ background: #334155; }}
  .report-left {{ flex: 1; }}
  .report-claim {{ font-weight: 600; font-size: 1rem; margin-bottom: .3rem; }}
  .report-meta {{ font-size: .8rem; color: #64748b; }}
  .report-verdict {{ font-size: .9rem; font-weight: 700; padding: .3rem .7rem; border-radius: 6px; flex-shrink: 0; margin-left: 1rem; }}
  .report-verdict.TRUE {{ background: #166534; color: #4ade80; }}
  .report-verdict.FALSE {{ background: #7f1d1d; color: #f87171; }}
  .report-verdict.PARTIALLY {{ background: #713f12; color: #facc15; }}
  .report-verdict.MISLEADING {{ background: #7f1d1d; color: #f87171; }}
  .report-verdict.UNVERIFIABLE {{ background: #374151; color: #9ca3af; }}
  .report-verdict.UNKNOWN {{ background: #374151; color: #9ca3af; }}
  .footer {{ text-align: center; color: #475569; font-size: .75rem; margin-top: 2rem; }}
</style>
</head>
<body>
<h1>Fact-Check Reports</h1>
<div class="count">{count} rapport(s)</div>
{entries}
<div class="footer">Fact-Check Engine</div>
</body>
</html>
"""


def _generate_report_filename(output: dict) -> str:
    """Generate a unique filename from timestamp + claim slug."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    claim = output.get("query", "report")
    slug = re.sub(r"[^a-z0-9]+", "-", claim.lower().strip())[:50].strip("-")
    return f"{ts}_{slug}.html"


def _rebuild_index(reports_dir: Path) -> str:
    """Scan reports dir and rebuild index.html."""
    report_files = sorted(reports_dir.glob("*.html"), reverse=True)
    report_files = [f for f in report_files if f.name != "index.html"]

    entries = []
    for f in report_files:
        content = f.read_text(encoding="utf-8")

        # Extract claim from <div class="claim">...</div>
        claim_match = re.search(r'<div class="claim">(.*?)</div>', content, re.DOTALL)
        claim = claim_match.group(1).strip() if claim_match else f.stem

        # Extract verdict from <div class="verdict ...">...</div>
        verdict_match = re.search(r'<div class="verdict[^"]*">([^<]+)</div>', content)
        verdict = verdict_match.group(1).strip() if verdict_match else "UNKNOWN"
        verdict_class = verdict.split()[0]

        # Extract confidence
        conf_match = re.search(r"Confiance\s*:\s*(\d+)%", content)
        conf = conf_match.group(1) + "%" if conf_match else ""

        # Extract timestamp from footer
        ts_match = re.search(r"Généré le ([^ ]+)", content)
        ts_display = ts_match.group(1)[:19].replace("T", " ") if ts_match else f.stem[:15]

        entries.append(
            f'<a class="report" href="{f.name}">'
            f'<div class="report-left">'
            f'<div class="report-claim">{claim}</div>'
            f'<div class="report-meta">{ts_display} — {conf}</div>'
            f"</div>"
            f'<div class="report-verdict {verdict_class}">{verdict}</div>'
            f"</a>"
        )

    index_html = INDEX_TEMPLATE.format(
        count=len(report_files),
        entries="\n".join(entries),
    )

    index_path = reports_dir / "index.html"
    index_path.write_text(index_html, encoding="utf-8")
    return str(index_path)


def generate_report_with_index(output: dict, reports_dir: Path | None = None) -> tuple[str, str]:
    """Generate a unique report + rebuild the index. Returns (report_path, index_path)."""
    if reports_dir is None:
        reports_dir = REPORTS_DIR
    reports_dir.mkdir(exist_ok=True)

    filename = _generate_report_filename(output)
    report_path = reports_dir / filename
    generate_html_report(output, str(report_path))
    index_path = _rebuild_index(reports_dir)
    return str(report_path), index_path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def _resolve_config(args: argparse.Namespace) -> dict:
    # Mode preset (can be overridden by explicit --agents)
    mode = args.mode or os.getenv("FACTCHECK_MODE", "default")
    preset = MODE_PRESETS.get(mode)
    if not preset:
        raise ValueError(f"Mode inconnu : {mode}. Disponibles : {list(MODE_PRESETS.keys())}")

    # --agents CLI flag > mode preset > FACTCHECK_AGENTS env var > default preset
    if args.agents:
        agents = [a.strip() for a in args.agents.split(",")]
    elif args.mode:
        # Explicit --mode overrides env var
        agents = list(preset["agents"])
    elif os.getenv("FACTCHECK_AGENTS"):
        agents = [a.strip() for a in os.getenv("FACTCHECK_AGENTS").split(",")]
    else:
        agents = list(preset["agents"])

    model = args.synthesis_model or os.getenv(
        "FACTCHECK_SYNTHESIS_MODEL", "google/gemini-2.5-flash"
    )
    max_workers = preset["max_workers"]
    search_depth = preset["search_depth"]

    return {
        "agents": agents,
        "synthesis_model": model,
        "max_workers": max_workers,
        "search_depth": search_depth,
        "mode": mode,
    }


def _setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


def main():
    modes_help = " | ".join(f"{k}: {v['description']}" for k, v in MODE_PRESETS.items())
    parser = argparse.ArgumentParser(
        description="Fact-Check Engine — Linkup search + multi-LLM analysis + thinking synthesis",
    )
    parser.add_argument("claim", nargs="*", help="Claim or query to fact-check")
    parser.add_argument("--mode", choices=MODE_PRESETS.keys(), help=f"Preset mode ({modes_help})")
    parser.add_argument("--agents", help="Comma-separated agent list (overrides mode preset)")
    parser.add_argument(
        "--synthesis-model", help="Model for synthesis (default: google/gemini-2.5-flash)"
    )
    parser.add_argument("--output", help="Save JSON output to file")
    parser.add_argument(
        "--html",
        nargs="?",
        const="reports",
        help="Generate HTML report in dir (default: reports/). Auto-names files, rebuilds index.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Include raw search & analysis in output"
    )
    parser.add_argument("--no-pretty", action="store_true", help="Compact JSON output")
    parser.add_argument(
        "--specialized",
        action="store_true",
        help="Assign specialized roles to agents (verificateur, avocat du diable, analyste sources, contextualiste)",
    )
    parser.add_argument(
        "--think",
        type=int,
        default=1,
        choices=[1, 2, 3, 5],
        help="Number of thinking models for synthesis (1=default, 2/3/5=multi-think parallel vote)",
    )
    parser.add_argument(
        "--no-screen",
        action="store_true",
        help="Skip pre-screening (force full pipeline even for non-factual claims)",
    )
    parser.add_argument(
        "--log-level",
        default="warning",
        choices=["debug", "info", "warning", "error"],
        help="Log level — use 'debug' or 'info' for parallelism details",
    )
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="Run a full test suite: pre-screening, all modes, specialized vs generic, HTML reports",
    )

    args = parser.parse_args()

    _setup_logging(args.log_level)

    if args.test_all:
        _run_test_all(args)
        return

    query = (
        " ".join(args.claim).strip() if args.claim else input("Entrez l'info à vérifier : ").strip()
    )
    if not query:
        console.print("[red]Aucune requête fournie.[/red]")
        return

    config = _resolve_config(args)

    log.info(
        "Mode: %s | Agents: %s | Workers: %s | Search: %s",
        config["mode"],
        config["agents"],
        config["max_workers"] or "auto",
        config["search_depth"],
    )

    output = factcheck(
        query=query,
        agents=config["agents"],
        synthesis_model=config["synthesis_model"],
        verbose=args.verbose,
        max_workers=config["max_workers"],
        search_depth=config["search_depth"],
        skip_screening=args.no_screen,
        specialized=args.specialized,
        think=args.think,
    )

    display_verdict(output)

    if args.output:
        indent = None if args.no_pretty else 2
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=indent)
        console.print(f"\n[dim]JSON sauvegardé → {args.output}[/dim]")

    if args.html:
        report_path, index_path = generate_report_with_index(output, Path(args.html))
        console.print(f"\n[dim]Rapport HTML → {report_path}[/dim]")
        console.print(f"[dim]Index        → {index_path}[/dim]")


# ──────────────────────────────────────────────────────────────────────────────
# Test-all — tests toutes les combinaisons
# ──────────────────────────────────────────────────────────────────────────────

TEST_CLAIMS = [
    # Factual — should pass screening and get a clear verdict
    ("La tour Eiffel mesure 330 metres", "factual"),
    # False — should be caught as FALSE
    ("La capitale de l'Australie est Sydney", "factual-false"),
    # Idiom — should be caught by pre-screening
    ("Le verre est a moitie plein", "idiom"),
    # Opinion — should be caught by pre-screening
    ("Python est le meilleur langage de programmation", "opinion"),
]


def _run_test_all(_args: argparse.Namespace) -> None:
    reports_dir = Path("reports")

    console.print(
        Panel(
            "[bold]Test complet — toutes les combinaisons[/bold]\n"
            f"[dim]{len(TEST_CLAIMS)} claims x (screening + modes + specialized)[/dim]",
            border_style="magenta",
        )
    )

    results_summary = []
    t_global = time.perf_counter()

    # 1. Pre-screening tests
    console.print("\n[bold magenta]═══ Phase 1 : Pre-screening ═══[/bold magenta]\n")
    for claim, _expected_type in TEST_CLAIMS:
        console.print(f"[bold]→ {claim}[/bold]")
        t0 = time.perf_counter()
        screening = prescreen_claim(claim)
        dt = time.perf_counter() - t0
        checkable = screening.get("checkable", True)
        category = screening.get("category", "?")
        icon = "✓" if checkable else "✗"
        color = "green" if checkable else "yellow"
        console.print(
            f"  [{color}]{icon} {category}[/{color}] — "
            f"{screening.get('reason', '')} ({dt:.1f}s)\n"
        )
        results_summary.append(
            {
                "claim": claim,
                "test": "screening",
                "result": category,
                "checkable": checkable,
                "time_s": round(dt, 1),
            }
        )

    # 2. Mode fast (generic) — only factual claims
    console.print("\n[bold magenta]═══ Phase 2 : Mode fast (generic) ═══[/bold magenta]\n")
    for claim, expected_type in TEST_CLAIMS:
        if expected_type.startswith("factual"):
            console.print(f"[bold]→ {claim}[/bold]")
            output = factcheck(
                query=claim,
                agents=MODE_PRESETS["fast"]["agents"],
                max_workers=MODE_PRESETS["fast"]["max_workers"],
                search_depth=MODE_PRESETS["fast"]["search_depth"],
                skip_screening=True,
            )
            display_verdict(output)
            report_path, _ = generate_report_with_index(output, reports_dir)
            console.print(f"[dim]→ {report_path}[/dim]\n")
            results_summary.append(
                {
                    "claim": claim,
                    "test": "fast-generic",
                    "verdict": output["verdict"].get("verdict"),
                    "confidence": output["verdict"].get("confidence"),
                    "time_s": output["timing"]["total_s"],
                }
            )

    # 3. Mode thorough (generic) — factual claims only
    console.print("\n[bold magenta]═══ Phase 3 : Mode thorough (generic) ═══[/bold magenta]\n")
    for claim, expected_type in TEST_CLAIMS:
        if expected_type.startswith("factual"):
            console.print(f"[bold]→ {claim}[/bold]")
            output = factcheck(
                query=claim,
                agents=MODE_PRESETS["thorough"]["agents"],
                search_depth=MODE_PRESETS["thorough"]["search_depth"],
                skip_screening=True,
            )
            display_verdict(output)
            report_path, _ = generate_report_with_index(output, reports_dir)
            console.print(f"[dim]→ {report_path}[/dim]\n")
            results_summary.append(
                {
                    "claim": claim,
                    "test": "thorough-generic",
                    "verdict": output["verdict"].get("verdict"),
                    "confidence": output["verdict"].get("confidence"),
                    "time_s": output["timing"]["total_s"],
                }
            )

    # 4. Mode thorough (specialized) — factual claims only
    console.print("\n[bold magenta]═══ Phase 4 : Mode thorough (spécialisé) ═══[/bold magenta]\n")
    for claim, expected_type in TEST_CLAIMS:
        if expected_type.startswith("factual"):
            console.print(f"[bold]→ {claim}[/bold]")
            output = factcheck(
                query=claim,
                agents=MODE_PRESETS["thorough"]["agents"],
                search_depth=MODE_PRESETS["thorough"]["search_depth"],
                skip_screening=True,
                specialized=True,
            )
            display_verdict(output)
            report_path, _ = generate_report_with_index(output, reports_dir)
            console.print(f"[dim]→ {report_path}[/dim]\n")
            results_summary.append(
                {
                    "claim": claim,
                    "test": "thorough-specialized",
                    "verdict": output["verdict"].get("verdict"),
                    "confidence": output["verdict"].get("confidence"),
                    "time_s": output["timing"]["total_s"],
                }
            )

    # Summary table
    total_time = time.perf_counter() - t_global

    console.print("\n[bold magenta]═══ Résumé ═══[/bold magenta]\n")
    summary_table = Table(show_header=True, header_style="bold cyan")
    summary_table.add_column("Claim", max_width=40)
    summary_table.add_column("Test")
    summary_table.add_column("Résultat")
    summary_table.add_column("Confiance")
    summary_table.add_column("Temps", justify="right")

    for r in results_summary:
        verdict = r.get("verdict", r.get("result", ""))
        conf = r.get("confidence")
        conf_str = f"{conf:.0%}" if isinstance(conf, int | float) and conf else ""
        color = VERDICT_COLORS.get(str(verdict), "white")
        summary_table.add_row(
            r["claim"][:40],
            r["test"],
            f"[{color}]{verdict}[/{color}]",
            conf_str,
            f"{r['time_s']:.1f}s",
        )

    console.print(summary_table)
    console.print(f"\n[bold]Temps total : {total_time:.1f}s[/bold]")
    console.print("[dim]Tous les rapports HTML → reports/index.html[/dim]")

    # Save summary JSON
    summary_path = reports_dir / "test-all-summary.json"
    summary_path.write_text(
        json.dumps(results_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    console.print(f"[dim]Résumé JSON → {summary_path}[/dim]")


if __name__ == "__main__":
    main()
