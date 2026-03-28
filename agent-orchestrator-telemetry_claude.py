"""
agent-orchestrator-telemetry_claude.py
=======================================
Multi-Agent DeepSearch Orchestrator — CrewAI + OpenTelemetry (spec-compliant).

Agents de recherche (ultra-rapides) :
  • Cerebras — llama-3.3-70b       (>2 000 tokens/s)
  • Groq     — llama-3.1-8b-instant (>750  tokens/s)

Comparateur haute qualité :
  • Claude Sonnet 4.5 (Anthropic)

Observabilité OpenTelemetry :
  • Traces  : hiérarchie de spans par phase / agent / parsing
              + trace_id dans le JSON de sortie pour corrélation
  • Metrics : histogrammes latence, counters tokens/requêtes/erreurs
  • Export  : OTLP-HTTP si OTEL_EXPORTER_OTLP_ENDPOINT défini
              sinon Console (SimpleSpanProcessor — flush synchrone)
  • Shutdown: force_flush() + shutdown() via atexit

Usage:
    python agent-orchestrator-telemetry_claude.py --query "..."
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \\
        python agent-orchestrator-telemetry_claude.py --query "..." --output result.json
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import sys
import textwrap
import time
import argparse
from contextlib import contextmanager
from typing import Generator, Optional

from dotenv import load_dotenv

# ── OpenTelemetry ──────────────────────────────────────────────────────────────
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,       # production (async, batched)
    ConsoleSpanExporter,
    SimpleSpanProcessor,      # dev (sync — flush garanti avant exit)
)
from opentelemetry.semconv.resource import ResourceAttributes

try:
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    _OTLP_AVAILABLE = True
except ImportError:
    _OTLP_AVAILABLE = False

# ── CrewAI ─────────────────────────────────────────────────────────────────────
from crewai import Agent, Crew, LLM, Process, Task

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CEREBRAS_API_KEY  = os.getenv("CEREBRAS_API_KEY",  "")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY",      "")

CEREBRAS_BASE_URL = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
CEREBRAS_MODEL    = os.getenv("CEREBRAS_MODEL",    "llama-3.3-70b")
GROQ_MODEL        = os.getenv("GROQ_MODEL",        "llama-3.1-8b-instant")
CLAUDE_MODEL      = os.getenv("CLAUDE_MODEL",      "claude-sonnet-4-5")

OTEL_ENDPOINT     = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
SERVICE_NAME      = os.getenv("OTEL_SERVICE_NAME", "deepsearch-orchestrator")
SERVICE_VERSION   = "2.0.0"


# ──────────────────────────────────────────────────────────────────────────────
# Telemetry — classe unique, lifecycle complet
# ──────────────────────────────────────────────────────────────────────────────

class Telemetry:
    """
    Encapsule la configuration OTel, les instruments et le lifecycle.

    Patterns respectés (https://opentelemetry.io/docs/languages/python/) :
    - Resource décrit le service
    - SimpleSpanProcessor en dev (flush synchrone, pas de spans perdus)
    - BatchSpanProcessor en prod (OTLP, haute performance)
    - force_flush() + shutdown() via atexit
    - Attributs en types natifs (str / int / float / bool)
    - StatusCode.OK explicite sur succès
    - trace_id extrait depuis le span context actif
    """

    def __init__(self, otlp_endpoint: str = "") -> None:
        self._prod = bool(otlp_endpoint and _OTLP_AVAILABLE)
        resource   = self._build_resource()

        self._tracer_provider = self._setup_traces(resource, otlp_endpoint)
        self._meter_provider  = self._setup_metrics(resource, otlp_endpoint)

        # Récupère tracer/meter APRÈS avoir configuré les providers globaux
        self.tracer = trace.get_tracer(SERVICE_NAME, SERVICE_VERSION)
        self.meter  = metrics.get_meter(SERVICE_NAME, SERVICE_VERSION)

        self._create_instruments()
        atexit.register(self._shutdown)

        mode = f"OTLP → {otlp_endpoint}" if self._prod else "Console (dev)"
        print(f"[OTel] Telemetry initialisée — export : {mode}", file=sys.stderr)

    # ── Setup ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_resource() -> Resource:
        return Resource.create({
            ResourceAttributes.SERVICE_NAME:    SERVICE_NAME,
            ResourceAttributes.SERVICE_VERSION: SERVICE_VERSION,
            "ai.research_models":  f"{CEREBRAS_MODEL}, {GROQ_MODEL}",
            "ai.synthesis_model":  CLAUDE_MODEL,
            "ai.framework":        "crewai",
        })

    def _setup_traces(self, resource: Resource, endpoint: str) -> TracerProvider:
        provider = TracerProvider(resource=resource)

        if self._prod:
            # BatchSpanProcessor : async, haute perf, export OTLP
            exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
            provider.add_span_processor(BatchSpanProcessor(exporter))
        else:
            # SimpleSpanProcessor : synchrone — garantit le flush en dev
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)
        return provider

    def _setup_metrics(self, resource: Resource, endpoint: str) -> MeterProvider:
        if self._prod:
            reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=f"{endpoint}/v1/metrics"),
                export_interval_millis=5_000,
            )
        else:
            reader = PeriodicExportingMetricReader(
                ConsoleMetricExporter(),
                export_interval_millis=15_000,
            )
        provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(provider)
        return provider

    def _create_instruments(self) -> None:
        """
        Instruments créés sur le meter déjà lié au MeterProvider configuré.
        Buckets alignés sur les durées typiques des appels LLM (secondes).
        """
        self.request_duration = self.meter.create_histogram(
            name="deepsearch.request.duration",
            unit="s",
            description="Durée totale end-to-end d'une requête DeepSearch",
        )
        self.agent_duration = self.meter.create_histogram(
            name="deepsearch.agent.duration",
            unit="s",
            description="Durée d'exécution par agent LLM",
        )
        self.request_counter = self.meter.create_counter(
            name="deepsearch.requests.total",
            description="Nombre total de requêtes traitées",
        )
        self.error_counter = self.meter.create_counter(
            name="deepsearch.errors.total",
            description="Nombre total d'erreurs",
        )
        self.token_counter = self.meter.create_counter(
            name="deepsearch.tokens.estimated",
            description="Tokens de sortie estimés (4 chars ≈ 1 token)",
        )

    # ── Shutdown ───────────────────────────────────────────────────────────────

    def _shutdown(self) -> None:
        """Appelé via atexit — flush et fermeture propre des providers."""
        try:
            self._tracer_provider.force_flush(timeout_millis=5_000)
            self._tracer_provider.shutdown()
            self._meter_provider.shutdown()
        except Exception:
            pass

    # ── Span context manager ───────────────────────────────────────────────────

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[dict] = None,
    ) -> Generator[trace.Span, None, None]:
        """
        Context manager conforme OTel :
        - Attributs en types natifs (str/int/float/bool)
        - StatusCode.OK explicite sur succès
        - record_exception() + StatusCode.ERROR sur erreur
        """
        with self.tracer.start_as_current_span(name) as s:
            if attributes:
                for key, value in attributes.items():
                    # OTel accepte str | bool | int | float (pas None)
                    if isinstance(value, (str, bool, int, float)):
                        s.set_attribute(key, value)
                    elif value is not None:
                        s.set_attribute(key, str(value))
            try:
                yield s
                s.set_status(trace.StatusCode.OK)
            except Exception as exc:
                s.record_exception(exc)
                s.set_status(trace.StatusCode.ERROR, description=str(exc))
                raise

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def current_trace_id() -> str:
        """Retourne le trace_id hex du span courant ('' si aucun span actif)."""
        ctx = trace.get_current_span().get_span_context()
        if ctx and ctx.is_valid:
            return format(ctx.trace_id, "032x")
        return ""

    @staticmethod
    def current_span_id() -> str:
        ctx = trace.get_current_span().get_span_context()
        if ctx and ctx.is_valid:
            return format(ctx.span_id, "016x")
        return ""

    def record_agent(
        self, provider: str, model: str, duration: float, output_text: str
    ) -> None:
        """Enregistre latence + tokens estimés pour un appel agent."""
        attrs = {"provider": provider, "model": model}
        self.agent_duration.record(duration, attrs)
        tokens = max(1, len(output_text) // 4)
        self.token_counter.add(tokens, attrs)


# ──────────────────────────────────────────────────────────────────────────────
# Utilitaires texte / JSON
# ──────────────────────────────────────────────────────────────────────────────

def normalize_query(raw: str) -> str:
    return " ".join(raw.strip().split())


def clean_text(text: str) -> str:
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_json(text: str) -> Optional[dict]:
    """Extrait le premier dict JSON valide d'une réponse LLM (4 stratégies)."""
    candidates: list[str] = [text.strip()]
    for pat in [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]:
        m = re.search(pat, text, re.DOTALL)
        if m:
            candidates.append(m.group(1))
    m = re.search(r"\{[\s\S]*\}", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))

    for c in candidates:
        try:
            parsed = json.loads(c.strip())
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def parse_research_response(raw: str) -> dict:
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
        "uncertainties": ["Réponse non parseable en JSON."],
        "sources": [],
    }


def parse_comparison_response(raw: str) -> dict:
    p = extract_json(raw)
    if p:
        ui = p.get("unique_insights", {})
        qa = p.get("quality_assessment", {})
        return {
            "consensus":     list(p.get("consensus",     [])),
            "disagreements": list(p.get("disagreements", [])),
            "unique_insights": {
                "cerebras": list(ui.get("cerebras", [])),
                "groq":     list(ui.get("groq",     [])),
            },
            "blind_spots": list(p.get("blind_spots", [])),
            "quality_assessment": {
                "cerebras": str(qa.get("cerebras", "")),
                "groq":     str(qa.get("groq",     "")),
            },
            "final_synthesis": str(p.get("final_synthesis", "")),
        }
    return {
        "consensus": [], "disagreements": [],
        "unique_insights": {"cerebras": [], "groq": []},
        "blind_spots": [],
        "quality_assessment": {"cerebras": "", "groq": ""},
        "final_synthesis": raw[:2000] if raw else "Parsing échoué.",
    }


# ──────────────────────────────────────────────────────────────────────────────
# LLM Factory
# ──────────────────────────────────────────────────────────────────────────────

def make_cerebras_llm() -> LLM:
    if not CEREBRAS_API_KEY:
        raise ValueError("CEREBRAS_API_KEY manquant dans .env")
    # litellm : préfixe openai/ pour tout endpoint OpenAI-compatible
    return LLM(
        model=f"openai/{CEREBRAS_MODEL}",
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        temperature=0.2,
    )


def make_groq_llm() -> LLM:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY manquant dans .env")
    # litellm : provider natif groq/
    return LLM(
        model=f"groq/{GROQ_MODEL}",
        api_key=GROQ_API_KEY,
        temperature=0.3,
    )


def make_claude_llm() -> LLM:
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY manquant dans .env")
    # litellm : provider natif anthropic/
    return LLM(
        model=f"anthropic/{CLAUDE_MODEL}",
        api_key=ANTHROPIC_API_KEY,
        temperature=0.15,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

_RESEARCH_JSON_FORMAT = textwrap.dedent("""
    Réponds UNIQUEMENT avec un objet JSON valide :
    {
      "executive_summary": "Résumé dense en 2-3 phrases.",
      "key_points":    ["Point 1", "Point 2"],
      "assumptions":   ["Hypothèse 1"],
      "uncertainties": ["Zone d'incertitude 1"],
      "sources":       ["Source connue 1"]
    }
    Aucun préfixe. Aucun commentaire. JSON uniquement.
""").strip()

_COMPARISON_JSON_FORMAT = textwrap.dedent("""
    Réponds UNIQUEMENT avec un objet JSON valide :
    {
      "consensus":     ["Point de convergence 1"],
      "disagreements": ["Désaccord 1 — description + origine (méthodologique/factuelle)"],
      "unique_insights": {
        "cerebras": ["Insight propre à Cerebras"],
        "groq":     ["Insight propre à Groq"]
      },
      "blind_spots": ["Angle non traité par aucun des deux agents"],
      "quality_assessment": {
        "cerebras": "Forces et faiblesses de la réponse Cerebras.",
        "groq":     "Forces et faiblesses de la réponse Groq."
      },
      "final_synthesis": "Synthèse finale (3-5 phrases) exploitant la complémentarité."
    }
    Aucun préfixe. Aucun commentaire. JSON uniquement.
""").strip()

_CEREBRAS_BACKSTORY = textwrap.dedent("""
    Tu es un analyste expert, rapide et rigoureux. Approche : factuelle et structurée.
    Tu vas à l'essentiel, hiérarchises l'information, distingues faits établis /
    inférences raisonnables / spéculations. Tu valorises la densité informationnelle.
""").strip()

_GROQ_BACKSTORY = textwrap.dedent("""
    Tu es un chercheur critique et créatif. Approche : exploratoire et contre-intuitive.
    Tu cherches les contre-arguments, remets en question les présupposés dominants,
    identifies les implications systémiques cachées et les biais potentiels.
""").strip()

_CLAUDE_BACKSTORY = textwrap.dedent("""
    Tu es un méta-analyste senior spécialisé en synthèse comparative multi-sources.
    Tu reçois deux analyses aux approches délibérément différentes et tu produis :
    consensus solides, divergences analysées à leur source, angles morts communs,
    évaluation honnête de chaque output, synthèse finale à valeur ajoutée.
    Tu n'es pas un résumeur : tu analyses et tu enrichis.
""").strip()


# ──────────────────────────────────────────────────────────────────────────────
# Builders Agents & Tasks
# ──────────────────────────────────────────────────────────────────────────────

def build_cerebras_agent() -> Agent:
    return Agent(
        role="Analyste DeepSearch Cerebras — Factuel / Structuré",
        goal="Analyse approfondie, factuelle et hiérarchisée de la requête.",
        backstory=_CEREBRAS_BACKSTORY,
        llm=make_cerebras_llm(),
        verbose=False,
        allow_delegation=False,
    )


def build_groq_agent() -> Agent:
    return Agent(
        role="Explorateur DeepSearch Groq — Critique / Alternatif",
        goal="Analyse exploratoire centrée sur les angles non conventionnels.",
        backstory=_GROQ_BACKSTORY,
        llm=make_groq_llm(),
        verbose=False,
        allow_delegation=False,
    )


def build_claude_comparator() -> Agent:
    return Agent(
        role="Analyste Comparatif Claude — Synthèse Haute Qualité",
        goal="Comparer les deux outputs de recherche et produire une synthèse à valeur ajoutée.",
        backstory=_CLAUDE_BACKSTORY,
        llm=make_claude_llm(),
        verbose=False,
        allow_delegation=False,
    )


def build_research_task(agent: Agent, query: str, directive: str) -> Task:
    return Task(
        description=textwrap.dedent(f"""
            Effectue une recherche approfondie sur :

            REQUÊTE : {query}
            DIRECTIVE : {directive}

            {_RESEARCH_JSON_FORMAT}
        """).strip(),
        expected_output="JSON valide : executive_summary, key_points, assumptions, uncertainties, sources.",
        agent=agent,
    )


def build_comparison_task(
    agent: Agent, query: str, cerebras_out: str, groq_out: str
) -> Task:
    return Task(
        description=textwrap.dedent(f"""
            Analyse comparative de deux réponses de recherche sur la même requête.

            REQUÊTE : {query}

            ── OUTPUT CEREBRAS (factuel / structuré) ──
            {cerebras_out[:3500]}

            ── OUTPUT GROQ (critique / alternatif) ────
            {groq_out[:3500]}

            {_COMPARISON_JSON_FORMAT}
        """).strip(),
        expected_output="JSON valide : consensus, disagreements, unique_insights, blind_spots, quality_assessment, final_synthesis.",
        agent=agent,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrateur
# ──────────────────────────────────────────────────────────────────────────────

class DeepSearchOrchestrator:
    """
    Orchestre les agents de recherche (Cerebras + Groq) et le comparateur (Claude).
    Chaque phase est enveloppée dans un span OTel enfant du span racine.

    Extension : ajouter un make_<x>_llm(), un build_<x>_agent(),
    une tâche dans _run_research_phase(), et passer l'output au comparateur.
    """

    def __init__(self, tel: Telemetry) -> None:
        self.tel = tel
        with tel.span("deepsearch.init", {"service.name": SERVICE_NAME}):
            self.cerebras_agent    = build_cerebras_agent()
            self.groq_agent        = build_groq_agent()
            self.claude_comparator = build_claude_comparator()

    # ── Phase 1 : recherche ───────────────────────────────────────────────────

    def _run_agent(
        self, agent: Agent, task: Task, provider: str, model: str
    ) -> tuple[str, float]:
        """Exécute un seul agent dans son propre crew + span."""
        t0 = time.perf_counter()
        with self.tel.span(f"deepsearch.agent.{provider}", {
            "ai.provider": provider,
            "ai.model":    model,
            "gen_ai.system": provider,
        }) as s:
            crew   = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
            result = crew.kickoff()
            raw    = clean_text(str(result.raw))
            duration = time.perf_counter() - t0

            s.add_event("agent.completed", {
                "duration_s":       round(duration, 3),
                "output_chars":     len(raw),
                "tokens_estimated": max(1, len(raw) // 4),
            })

        self.tel.record_agent(provider, model, duration, raw)
        return raw, duration

    def _run_research_phase(self, query: str) -> tuple[str, str, float]:
        with self.tel.span("deepsearch.research_phase", {
            "query.hash": hashlib.md5(query.encode()).hexdigest(),
            "agents":     "cerebras, groq",
        }) as s:
            t0 = time.perf_counter()

            cerebras_raw, t_cerebras = self._run_agent(
                self.cerebras_agent,
                build_research_task(
                    self.cerebras_agent, query,
                    "Factuel et structuré : rigueur, hiérarchisation, densité informationnelle.",
                ),
                "cerebras", CEREBRAS_MODEL,
            )

            groq_raw, t_groq = self._run_agent(
                self.groq_agent,
                build_research_task(
                    self.groq_agent, query,
                    "Critique et exploratoire : contre-arguments, angles alternatifs, biais potentiels.",
                ),
                "groq", GROQ_MODEL,
            )

            total = time.perf_counter() - t0
            s.set_attribute("phase.duration_s", round(total, 3))
            s.set_attribute("cerebras.duration_s", round(t_cerebras, 3))
            s.set_attribute("groq.duration_s",     round(t_groq,     3))

        return cerebras_raw, groq_raw, total

    # ── Phase 2 : comparaison ─────────────────────────────────────────────────

    def _run_comparison_phase(
        self, query: str, cerebras_raw: str, groq_raw: str
    ) -> tuple[str, float]:
        with self.tel.span("deepsearch.comparison_phase", {
            "ai.provider": "anthropic",
            "ai.model":    CLAUDE_MODEL,
        }):
            raw, duration = self._run_agent(
                self.claude_comparator,
                build_comparison_task(
                    self.claude_comparator, query, cerebras_raw, groq_raw
                ),
                "anthropic", CLAUDE_MODEL,
            )
        return raw, duration

    # ── Entrée principale ─────────────────────────────────────────────────────

    def run(self, query: str) -> dict:
        query   = normalize_query(query)
        t_start = time.perf_counter()

        self.tel.request_counter.add(1, {"query.length": len(query)})

        try:
            with self.tel.span("deepsearch.orchestrate", {
                "query":           query,
                "query.hash":      hashlib.md5(query.encode()).hexdigest(),
                "model.cerebras":  CEREBRAS_MODEL,
                "model.groq":      GROQ_MODEL,
                "model.claude":    CLAUDE_MODEL,
            }) as root:

                trace_id = self.tel.current_trace_id()
                span_id  = self.tel.current_span_id()

                # Phase 1
                cerebras_raw, groq_raw, t_research = self._run_research_phase(query)

                with self.tel.span("deepsearch.parse_research"):
                    cerebras_structured = parse_research_response(cerebras_raw)
                    groq_structured     = parse_research_response(groq_raw)

                # Phase 2
                comparison_raw, t_comparison = self._run_comparison_phase(
                    query, cerebras_raw, groq_raw
                )

                with self.tel.span("deepsearch.parse_comparison"):
                    comparative_analysis = parse_comparison_response(comparison_raw)

                t_total = time.perf_counter() - t_start
                root.set_attribute("duration.total_s",      round(t_total,      3))
                root.set_attribute("duration.research_s",   round(t_research,   3))
                root.set_attribute("duration.comparison_s", round(t_comparison, 3))
                root.add_event("orchestration.completed", {"total_s": round(t_total, 3)})

            self.tel.request_duration.record(t_total, {"status": "success"})

            return {
                "query": query,
                "telemetry": {
                    "service":       SERVICE_NAME,
                    "trace_id":      trace_id,   # corrélation logs ↔ traces
                    "span_id":       span_id,
                    "otlp_endpoint": OTEL_ENDPOINT or "console",
                    "timing": {
                        "total_seconds":      round(t_total,      3),
                        "research_seconds":   round(t_research,   3),
                        "comparison_seconds": round(t_comparison, 3),
                    },
                    "models": {
                        "cerebras": CEREBRAS_MODEL,
                        "groq":     GROQ_MODEL,
                        "claude":   CLAUDE_MODEL,
                    },
                },
                "agent_outputs": {
                    "cerebras": {"raw": cerebras_raw, "structured": cerebras_structured},
                    "groq":     {"raw": groq_raw,     "structured": groq_structured},
                },
                "comparative_analysis": comparative_analysis,
            }

        except Exception as exc:
            self.tel.error_counter.add(1, {"error.type": type(exc).__name__})
            self.tel.request_duration.record(
                time.perf_counter() - t_start, {"status": "error"}
            )
            raise


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DeepSearch — Cerebras + Groq + Claude + OpenTelemetry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Exemples :
              python agent-orchestrator-telemetry_claude.py --query "Impact de l'IA sur l'emploi"

              # Export vers Jaeger / Grafana Tempo / Honeycomb
              OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \\
                python agent-orchestrator-telemetry_claude.py --query "..." --output result.json

              # Stack locale : docker run -p 4318:4318 -p 16686:16686 jaegertracing/all-in-one
        """),
    )
    parser.add_argument("--query",     required=True,         help="Requête DeepSearch")
    parser.add_argument("--output",    default=None,          help="Fichier de sortie JSON")
    parser.add_argument("--no-pretty", action="store_true",   help="JSON compact")
    args = parser.parse_args()

    # OTel init — DOIT précéder tout appel à get_tracer() / get_meter()
    tel = Telemetry(otlp_endpoint=OTEL_ENDPOINT)

    indent = None if args.no_pretty else 2

    try:
        orchestrator = DeepSearchOrchestrator(tel)
        result       = orchestrator.run(args.query)
        output_json  = json.dumps(result, ensure_ascii=False, indent=indent)
        print(output_json)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(output_json)
            print(f"[OK] Sauvegardé : {args.output}", file=sys.stderr)

    except ValueError as exc:
        print(json.dumps({"error": f"Configuration : {exc}"}, ensure_ascii=False, indent=2))
        sys.exit(1)
    except KeyboardInterrupt:
        print(json.dumps({"error": "Interrompu."}, ensure_ascii=False, indent=2))
        sys.exit(130)
    except Exception as exc:
        print(json.dumps({"error": f"Erreur : {exc}"}, ensure_ascii=False, indent=2))
        sys.exit(1)
    # atexit déclenche Telemetry._shutdown() → force_flush() + shutdown()
