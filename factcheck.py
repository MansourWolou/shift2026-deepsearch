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

import os
import re
import json
import time
import logging
import argparse
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.logging import RichHandler

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

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


def extract_json(text: str) -> Optional[dict]:
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


def _invoke_agent(model: str, claim: str, search_results: dict) -> str:
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

    messages = [
        HumanMessage(content=ANALYSIS_PROMPT.format(
            claim=claim,
            search_answer=search_results.get("answer", "Aucune réponse"),
            sources_text=sources_text or "Aucune source",
        ))
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
) -> dict[str, dict]:
    """
    Lance N agents LLM en parallèle, chacun analysant les mêmes résultats Linkup.
    """
    if max_workers is None:
        max_workers = len(agents)

    log.info("Parallelism: %d agents, %d workers", len(agents), max_workers)
    for a in agents:
        log.debug("  → %s (%s)", AGENT_REGISTRY[a]["label"], AGENT_REGISTRY[a]["model"])

    results: dict[str, dict] = {}

    def _run_one(agent_name: str) -> tuple[str, str, float, str | None]:
        t0 = time.perf_counter()
        log.debug("[%s] started", agent_name)
        try:
            model = AGENT_REGISTRY[agent_name]["model"]
            raw = _invoke_agent(model, claim, search_results)
            dt = time.perf_counter() - t0
            log.info("[%s] done in %.1fs (%d chars)", agent_name, dt, len(raw))
            return (agent_name, raw, dt, None)
        except Exception as e:
            dt = time.perf_counter() - t0
            log.error("[%s] failed after %.1fs: %s", agent_name, dt, e)
            return (agent_name, "", dt, str(e))

    status_map: dict[str, str] = {name: "[yellow]en cours…[/yellow]" for name in agents}

    def _build_table() -> Table:
        table = Table(title="Analyse en cours", show_header=True, header_style="bold cyan")
        table.add_column("Agent", style="bold")
        table.add_column("Statut")
        table.add_column("Durée", justify="right")
        for name in agents:
            duration = results[name]["duration_s"] if name in results else ""
            dur_str = f"{duration:.1f}s" if duration else "…"
            table.add_row(AGENT_REGISTRY[name]["label"], status_map[name], dur_str)
        return table

    with Live(_build_table(), console=console, refresh_per_second=2) as live:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_one, name): name for name in agents}
            for future in as_completed(futures):
                name, raw, duration, error = future.result()
                results[name] = {
                    "label": AGENT_REGISTRY[name]["label"],
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
        {"topic": "What they disagree about", "positions": {"agent1": "position A", "agent2": "position B"}}
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


def synthesize_verdict(
    query: str,
    research_results: dict[str, dict],
    model: str = "google/gemini-2.5-flash",
) -> dict:
    reports = []
    for name, data in research_results.items():
        if data["error"]:
            reports.append(
                f"--- REPORT FROM {data['label']} ({name}) ---\n"
                f"[FAILED: {data['error']}]\n"
                f"--- END REPORT ---"
            )
        else:
            text = clean_text(data["raw"])[:4000]
            reports.append(
                f"--- REPORT FROM {data['label']} ({name}) ---\n"
                f"{text}\n"
                f"--- END REPORT ---"
            )

    agent_reports = "\n\n".join(reports)
    n_agents = len(research_results)

    llm = ChatOpenAI(
        model=model,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

    messages = [
        SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT.format(
            n_agents=n_agents,
            output_schema=json.dumps(VERDICT_SCHEMA, indent=2),
        )),
        HumanMessage(content=SYNTHESIS_USER_PROMPT.format(
            query=query,
            agent_reports=agent_reports,
            n_agents=n_agents,
        )),
    ]

    console.print("\n[bold magenta]Synthèse en cours…[/bold magenta] "
                  f"({model})\n")

    response = llm.invoke(messages)
    content = response.content

    if isinstance(content, list):
        content = next(
            (block["text"] for block in content if isinstance(block, dict) and block.get("type") == "text"),
            str(content),
        )

    parsed = extract_json(content)
    if parsed:
        return parsed

    return {
        "claim": query,
        "verdict": "UNVERIFIABLE",
        "confidence": 0.0,
        "summary": "Le thinking model n'a pas retourné de JSON valide.",
        "raw_response": content[:2000],
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
) -> dict:
    query = normalize_query(query)
    if agents is None:
        agents = list(DEFAULT_AGENTS)

    for name in agents:
        if name not in AGENT_REGISTRY:
            raise ValueError(f"Agent inconnu : {name}. Disponibles : {list(AGENT_REGISTRY.keys())}")

    console.print(Panel(
        f"[bold]Fact-Check Engine[/bold]\n"
        f"[white]{query}[/white]\n"
        f"[dim]Agents : {', '.join(agents)} | Synthèse : {synthesis_model}[/dim]",
        border_style="cyan",
    ))

    t_total = time.perf_counter()

    # Phase 1 — Recherche Linkup
    console.print(f"\n[bold blue]Recherche Linkup ({search_depth})…[/bold blue]")
    t_search = time.perf_counter()
    search_results = linkup_search(query, depth=search_depth)
    search_duration = time.perf_counter() - t_search

    n_sources = len(search_results.get("sources", []))
    console.print(f"[green]{n_sources} sources trouvées[/green] ({search_duration:.1f}s)\n")

    # Phase 2 — Analyse parallèle par N modèles
    t_analysis = time.perf_counter()
    agent_results = run_research_parallel(query, search_results, agents, max_workers=max_workers)
    analysis_duration = time.perf_counter() - t_analysis

    successful = sum(1 for r in agent_results.values() if not r["error"])
    console.print(f"\n[green]{successful}/{len(agents)} agents terminés avec succès[/green]")

    # Phase 3 — Synthèse
    t_synthesis = time.perf_counter()
    verdict = synthesize_verdict(query, agent_results, synthesis_model)
    synthesis_duration = time.perf_counter() - t_synthesis

    total_duration = time.perf_counter() - t_total

    output = {
        "query": query,
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
}


def display_verdict(output: dict) -> None:
    v = output["verdict"]
    verdict_text = v.get("verdict", "UNKNOWN")
    color = VERDICT_COLORS.get(verdict_text, "white")
    confidence = v.get("confidence", 0)

    console.print(Panel(
        f"[bold {color}]{verdict_text}[/bold {color}]  "
        f"(confiance : {confidence:.0%})\n\n"
        f"{v.get('summary', '')}",
        title="[bold]Verdict[/bold]",
        border_style=color,
    ))

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

    t = output["timing"]
    console.print(f"\n[dim]Total: {t['total_s']:.1f}s | "
                  f"Search: {t['search_s']:.1f}s | "
                  f"Analyse: {t['analysis_s']:.1f}s | "
                  f"Synthèse: {t['synthesis_s']:.1f}s[/dim]")


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

<div class="section agents">
  <h2>Agents — Parallélisme</h2>
  {agents_html}
</div>

<div class="timing">
  <div class="timing-card"><div class="val">{total_s}s</div><div class="lbl">Total</div></div>
  <div class="timing-card"><div class="val">{search_s}s</div><div class="lbl">Recherche</div></div>
  <div class="timing-card"><div class="val">{analysis_s}s</div><div class="lbl">Analyse</div></div>
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
    confidence = round(v.get("confidence", 0) * 100) if v.get("confidence", 0) <= 1 else round(v.get("confidence", 0))

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
            f'</div>'
            f'<div class="agent-bar" style="width:{pct:.0f}%"></div>'
        )

    def _list_section(title: str, items: list, color: str) -> str:
        if not items:
            return ""
        lis = "".join(f"<li>{_esc(i.get('point', i) if isinstance(i, dict) else i)}</li>" for i in items)
        return f'<div class="section"><h2 style="color:{color}">{title}</h2><ul>{lis}</ul></div>'

    def _esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    html = HTML_TEMPLATE.format(
        query_escaped=_esc(output["query"]),
        verdict_text=verdict_text,
        verdict_class=verdict_class,
        confidence=confidence,
        summary=_esc(v.get("summary", "")),
        evidence_for_html=_list_section("Evidence FOR", v.get("evidence_for", []), "var(--green)"),
        evidence_against_html=_list_section("Evidence AGAINST", v.get("evidence_against", []), "var(--red)"),
        nuances_html=_list_section("Nuances", v.get("nuances", []), "var(--yellow)"),
        agents_html="\n".join(agents_rows),
        total_s=f"{t['total_s']:.1f}",
        search_s=f"{t['search_s']:.1f}",
        analysis_s=f"{t['analysis_s']:.1f}",
        synthesis_s=f"{t['synthesis_s']:.1f}",
        timestamp=output.get("timestamp", ""),
    )

    Path(path).write_text(html, encoding="utf-8")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_config(args: argparse.Namespace) -> dict:
    # Mode preset (can be overridden by explicit --agents)
    mode = args.mode or os.getenv("FACTCHECK_MODE", "default")
    preset = MODE_PRESETS.get(mode)
    if not preset:
        raise ValueError(f"Mode inconnu : {mode}. Disponibles : {list(MODE_PRESETS.keys())}")

    agents_str = args.agents or os.getenv("FACTCHECK_AGENTS")
    if agents_str:
        agents = [a.strip() for a in agents_str.split(",")]
    else:
        agents = list(preset["agents"])

    model = args.synthesis_model or os.getenv("FACTCHECK_SYNTHESIS_MODEL", "google/gemini-2.5-flash")
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
    parser.add_argument("--synthesis-model", help="Model for synthesis (default: google/gemini-2.5-flash)")
    parser.add_argument("--output", help="Save JSON output to file")
    parser.add_argument("--html", nargs="?", const="report.html", help="Generate HTML report (dev/demo). Optional path, default: report.html")
    parser.add_argument("--verbose", action="store_true", help="Include raw search & analysis in output")
    parser.add_argument("--no-pretty", action="store_true", help="Compact JSON output")
    parser.add_argument("--log-level", default="warning", choices=["debug", "info", "warning", "error"],
                        help="Log level — use 'debug' or 'info' for parallelism details")

    args = parser.parse_args()

    _setup_logging(args.log_level)

    query = " ".join(args.claim).strip() if args.claim else input("Entrez l'info à vérifier : ").strip()
    if not query:
        console.print("[red]Aucune requête fournie.[/red]")
        return

    config = _resolve_config(args)

    log.info("Mode: %s | Agents: %s | Workers: %s | Search: %s",
             config["mode"], config["agents"], config["max_workers"] or "auto", config["search_depth"])

    output = factcheck(
        query=query,
        agents=config["agents"],
        synthesis_model=config["synthesis_model"],
        verbose=args.verbose,
        max_workers=config["max_workers"],
        search_depth=config["search_depth"],
    )

    display_verdict(output)

    if args.output:
        indent = None if args.no_pretty else 2
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=indent)
        console.print(f"\n[dim]JSON sauvegardé → {args.output}[/dim]")

    if args.html:
        html_path = generate_html_report(output, args.html)
        console.print(f"\n[dim]Rapport HTML → {html_path}[/dim]")


if __name__ == "__main__":
    main()
