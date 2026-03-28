"""
agent-orchestrator-telemetry.py
================================
Fast Multi-Agent DeepSearch Orchestrator with OpenTelemetry traces.

Usage:
  python agent-orchestrator-telemetry.py --query "your deepsearch query"
  python agent-orchestrator-telemetry.py --query "..." --telemetry-export otlp
"""

import argparse
import json
import os
import re
import textwrap
import time
from typing import Optional

from dotenv import load_dotenv
from crewai import Agent, Crew, LLM, Process, Task

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Status, StatusCode

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
except ImportError:  # optional dependency
    OTLPSpanExporter = None


load_dotenv()


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


def parse_research_response(raw: str) -> dict:
    parsed = extract_json(raw)
    if parsed:
        return {
            "executive_summary": str(parsed.get("executive_summary", "")),
            "key_points": list(parsed.get("key_points", [])),
            "assumptions": list(parsed.get("assumptions", [])),
            "uncertainties": list(parsed.get("uncertainties", [])),
            "sources": list(parsed.get("sources", [])),
        }
    return {
        "executive_summary": raw[:500] if raw else "Parsing failed.",
        "key_points": [],
        "assumptions": [],
        "uncertainties": ["Response could not be parsed as JSON."],
        "sources": [],
    }


def parse_comparison_response(raw: str) -> dict:
    parsed = extract_json(raw)
    if parsed:
        unique_insights = parsed.get("unique_insights", {})
        return {
            "consensus": list(parsed.get("consensus", [])),
            "disagreements": list(parsed.get("disagreements", [])),
            "unique_insights": {
                "openai_fast": list(unique_insights.get("openai_fast", unique_insights.get("openai", []))),
                "mistral_fast": list(unique_insights.get("mistral_fast", unique_insights.get("mistral", []))),
            },
            "blind_spots": list(parsed.get("blind_spots", [])),
            "final_synthesis": str(parsed.get("final_synthesis", "")),
        }
    return {
        "consensus": [],
        "disagreements": [],
        "unique_insights": {"openai_fast": [], "mistral_fast": []},
        "blind_spots": [],
        "final_synthesis": raw[:1200] if raw else "Comparison parsing failed.",
    }


def build_telemetry(
    service_name: str,
    export_mode: str,
    otlp_endpoint: Optional[str],
):
    resource = Resource.create(
        attributes={
            SERVICE_NAME: service_name,
            "orchestrator.type": "deepsearch",
            "orchestrator.profile": "fast",
        }
    )
    provider = TracerProvider(resource=resource)

    if export_mode == "otlp":
        if OTLPSpanExporter is None:
            raise ValueError(
                "OTLP exporter package missing. Install: opentelemetry-exporter-otlp-proto-http"
            )
        endpoint = (
            otlp_endpoint
            or os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
            or "http://localhost:4318/v1/traces"
        )
        exporter = OTLPSpanExporter(endpoint=endpoint)
    else:
        exporter = ConsoleSpanExporter()

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("agent_orchestrator_telemetry")
    return tracer, provider


_PROVIDER_REGISTRY = {
    "openai_fast": {
        "model_env": "FAST_OPENAI_MODEL",
        "default_model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 0.1,
    },
    "mistral_fast": {
        "model_env": "FAST_MISTRAL_MODEL",
        "default_model": "mistral/mistral-small-latest",
        "api_key_env": "MISTRAL_API_KEY",
        "temperature": 0.2,
    },
}


def make_llm(provider: str, temperature_override: Optional[float] = None) -> LLM:
    cfg = _PROVIDER_REGISTRY[provider]
    api_key = os.getenv(cfg["api_key_env"], "")
    if not api_key:
        raise ValueError(f"Missing API key: {cfg['api_key_env']}")
    model = os.getenv(cfg["model_env"], cfg["default_model"])
    temperature = cfg["temperature"] if temperature_override is None else temperature_override
    return LLM(model=model, api_key=api_key, temperature=temperature)


RESEARCH_OUTPUT_FORMAT = textwrap.dedent(
    """
    Respond ONLY with valid JSON:
    {
      "executive_summary": "2-3 concise sentences",
      "key_points": ["...", "..."],
      "assumptions": ["..."],
      "uncertainties": ["..."],
      "sources": ["..."]
    }
    """
).strip()


class TelemetryFastOrchestrator:
    def __init__(self, tracer) -> None:
        self.tracer = tracer
        self.openai_agent = Agent(
            role="Fast Analytical Researcher",
            goal="Produce a compact, evidence-first analysis quickly.",
            backstory=(
                "You optimize for speed and correctness. Prefer concise factual answers, "
                "clear assumptions, and minimal verbosity."
            ),
            llm=make_llm("openai_fast"),
            verbose=False,
            allow_delegation=False,
        )
        self.mistral_agent = Agent(
            role="Fast Alternative Researcher",
            goal="Produce a compact alternative analysis quickly.",
            backstory=(
                "You optimize for speed while surfacing counter-arguments and edge-cases."
            ),
            llm=make_llm("mistral_fast"),
            verbose=False,
            allow_delegation=False,
        )
        self.comparator_agent = Agent(
            role="Fast Comparative Synthesizer",
            goal="Find consensus/disagreements and provide a final synthesis quickly.",
            backstory="You compare two analyses and deliver only high-value synthesis.",
            llm=make_llm("openai_fast", temperature_override=0.1),
            verbose=False,
            allow_delegation=False,
        )

    def _run_single_research(self, label: str, agent: Agent, query: str, methodology: str) -> str:
        with self.tracer.start_as_current_span(f"phase.research.{label}") as span:
            span.set_attribute("query.length", len(query))
            span.set_attribute("methodology", methodology)
            task = Task(
                description=textwrap.dedent(
                    f"""
                    Research the following query quickly but reliably:

                    QUERY: {query}
                    METHODOLOGY: {methodology}

                    {RESEARCH_OUTPUT_FORMAT}
                    """
                ).strip(),
                expected_output=(
                    "A valid JSON object with: executive_summary, key_points, "
                    "assumptions, uncertainties, sources."
                ),
                agent=agent,
            )
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
            )
            started = time.perf_counter()
            try:
                result = crew.kickoff()
                raw = clean_text(str(result.raw))
                span.set_attribute("duration.seconds", round(time.perf_counter() - started, 3))
                span.set_attribute("response.length", len(raw))
                span.set_status(Status(StatusCode.OK))
                return raw
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    def _run_comparison(self, query: str, openai_raw: str, mistral_raw: str) -> str:
        with self.tracer.start_as_current_span("phase.compare") as span:
            task = Task(
                description=textwrap.dedent(
                    f"""
                    Compare two fast research outputs for the same query.

                    QUERY: {query}

                    OPENAI_FAST OUTPUT:
                    {openai_raw[:2500]}

                    MISTRAL_FAST OUTPUT:
                    {mistral_raw[:2500]}

                    Respond ONLY with valid JSON:
                    {{
                      "consensus": ["..."],
                      "disagreements": ["..."],
                      "unique_insights": {{
                        "openai_fast": ["..."],
                        "mistral_fast": ["..."]
                      }},
                      "blind_spots": ["..."],
                      "final_synthesis": "3-5 sentences"
                    }}
                    """
                ).strip(),
                expected_output=(
                    "A valid JSON object with consensus, disagreements, unique_insights, "
                    "blind_spots, final_synthesis."
                ),
                agent=self.comparator_agent,
            )
            crew = Crew(
                agents=[self.comparator_agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
            )
            started = time.perf_counter()
            try:
                result = crew.kickoff()
                raw = clean_text(str(result.raw))
                span.set_attribute("duration.seconds", round(time.perf_counter() - started, 3))
                span.set_attribute("response.length", len(raw))
                span.set_status(Status(StatusCode.OK))
                return raw
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    def run(self, query: str) -> dict:
        query = normalize_query(query)
        total_started = time.perf_counter()

        with self.tracer.start_as_current_span("orchestrator.run") as span:
            span.set_attribute("query.length", len(query))
            span.set_attribute("model.openai_fast", os.getenv("FAST_OPENAI_MODEL", "gpt-4o-mini"))
            span.set_attribute(
                "model.mistral_fast",
                os.getenv("FAST_MISTRAL_MODEL", "mistral/mistral-small-latest"),
            )
            try:
                research_started = time.perf_counter()
                openai_raw = self._run_single_research(
                    "openai_fast",
                    self.openai_agent,
                    query,
                    "Facts-first, concise and robust.",
                )
                mistral_raw = self._run_single_research(
                    "mistral_fast",
                    self.mistral_agent,
                    query,
                    "Counter-arguments and alternative framing, concise.",
                )
                research_duration = time.perf_counter() - research_started

                comparison_started = time.perf_counter()
                comparison_raw = self._run_comparison(query, openai_raw, mistral_raw)
                comparison_duration = time.perf_counter() - comparison_started

                total_duration = time.perf_counter() - total_started
                span.set_attribute("duration.total_seconds", round(total_duration, 3))
                span.set_status(Status(StatusCode.OK))

                return {
                    "query": query,
                    "timing": {
                        "total_seconds": round(total_duration, 3),
                        "research_phase_seconds": round(research_duration, 3),
                        "comparison_phase_seconds": round(comparison_duration, 3),
                    },
                    "model_config": {
                        "openai_fast": os.getenv("FAST_OPENAI_MODEL", "gpt-4o-mini"),
                        "mistral_fast": os.getenv("FAST_MISTRAL_MODEL", "mistral/mistral-small-latest"),
                    },
                    "agent_outputs": {
                        "openai_fast": {
                            "raw": openai_raw,
                            "structured": parse_research_response(openai_raw),
                        },
                        "mistral_fast": {
                            "raw": mistral_raw,
                            "structured": parse_research_response(mistral_raw),
                        },
                    },
                    "comparative_analysis": parse_comparison_response(comparison_raw),
                }
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast Multi-Agent DeepSearch Orchestrator with OpenTelemetry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python agent-orchestrator-telemetry.py --query "Latest AI chip trends"
              python agent-orchestrator-telemetry.py --query "..." --telemetry-export otlp
            """
        ),
    )
    parser.add_argument(
        "--query",
        default="le venezuela est un etat des usa depusi 1903",
        help="DeepSearch query",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output file")
    parser.add_argument("--no-pretty", action="store_true", help="Compact JSON output")
    parser.add_argument(
        "--telemetry-export",
        choices=["console", "otlp"],
        default=os.getenv("OTEL_EXPORTER_MODE", "console"),
        help="Where spans are exported",
    )
    parser.add_argument(
        "--otlp-endpoint",
        default=os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"),
        help="OTLP traces endpoint (example: http://localhost:4318/v1/traces)",
    )
    parser.add_argument(
        "--service-name",
        default=os.getenv("OTEL_SERVICE_NAME", "agent-orchestrator-telemetry"),
        help="OpenTelemetry service.name",
    )
    parser.add_argument(
        "--openai-model",
        default=None,
        help="Override FAST_OPENAI_MODEL for this run",
    )
    parser.add_argument(
        "--mistral-model",
        default=None,
        help="Override FAST_MISTRAL_MODEL for this run",
    )
    args = parser.parse_args()

    if args.openai_model:
        os.environ["FAST_OPENAI_MODEL"] = args.openai_model
    if args.mistral_model:
        os.environ["FAST_MISTRAL_MODEL"] = args.mistral_model

    indent = None if args.no_pretty else 2

    tracer = None
    provider = None
    try:
        tracer, provider = build_telemetry(
            service_name=args.service_name,
            export_mode=args.telemetry_export,
            otlp_endpoint=args.otlp_endpoint,
        )
        orchestrator = TelemetryFastOrchestrator(tracer=tracer)
        result = orchestrator.run(args.query)
        output_json = json.dumps(result, ensure_ascii=False, indent=indent)
        print(output_json)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as file_handle:
                file_handle.write(output_json)
    except ValueError as exc:
        print(json.dumps({"error": f"Configuration error: {exc}"}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    except KeyboardInterrupt:
        print(json.dumps({"error": "Interrupted by user."}, ensure_ascii=False, indent=2))
        raise SystemExit(130)
    except Exception as exc:
        print(json.dumps({"error": f"Unexpected error: {exc}"}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    finally:
        if provider is not None:
            provider.force_flush()
            provider.shutdown()
