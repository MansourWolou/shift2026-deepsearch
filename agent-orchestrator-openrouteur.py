"""
agent-orchestrator-openrouteur.py
=================================
Fast multi-agent orchestrator using OpenRouter Quickstart patterns.

Quickstart alignment:
- OpenAI SDK client with base_url="https://openrouter.ai/api/v1"
- API key via OPENROUTER_API_KEY
- Optional OpenRouter attribution headers:
  HTTP-Referer, X-OpenRouter-Title
"""

import argparse
import json
import os
import re
import time
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def normalize_query(raw: str) -> str:
    return " ".join(raw.strip().split())


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
        insights = parsed.get("unique_insights", {})
        return {
            "consensus": list(parsed.get("consensus", [])),
            "disagreements": list(parsed.get("disagreements", [])),
            "unique_insights": {
                "analytic_fast": list(insights.get("analytic_fast", [])),
                "alternative_fast": list(insights.get("alternative_fast", [])),
            },
            "blind_spots": list(parsed.get("blind_spots", [])),
            "final_synthesis": str(parsed.get("final_synthesis", "")),
        }
    return {
        "consensus": [],
        "disagreements": [],
        "unique_insights": {"analytic_fast": [], "alternative_fast": []},
        "blind_spots": [],
        "final_synthesis": raw[:1200] if raw else "Comparison parsing failed.",
    }


def _router_headers() -> dict:
    headers = {}
    site_url = os.getenv("OPENROUTER_SITE_URL", "").strip()
    app_title = os.getenv("OPENROUTER_APP_NAME", "").strip()
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_title:
        headers["X-OpenRouter-Title"] = app_title
    return headers


class OpenRouteurOrchestrator:
    def __init__(self) -> None:
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Missing OPENROUTER_API_KEY in environment.")

        self.client = OpenAI(
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=api_key,
        )
        self.extra_headers = _router_headers()

        self.analytic_model = os.getenv("OPENROUTER_FAST_ANALYTIC_MODEL", "qwen/qwen-2.5-7b-instruct")
        self.alternative_model = os.getenv(
            "OPENROUTER_FAST_ALTERNATIVE_MODEL",
            "meta-llama/llama-3.2-3b-instruct",
        )
        self.synthesis_model = os.getenv("OPENROUTER_FAST_SYNTH_MODEL", "openrouter/auto")
        self.max_tokens = int(os.getenv("OPENROUTER_MAX_TOKENS", "900"))

    def _chat_json(self, model: str, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        completion = self.client.chat.completions.create(**kwargs)
        content = completion.choices[0].message.content
        return content.strip() if isinstance(content, str) else str(content)

    def _run_research(self, query: str, mode: str, model: str, methodology: str) -> tuple[str, float]:
        system_prompt = (
            "You are a fast DeepSearch agent. You must be concise, factual, and structured. "
            "Always return valid JSON only."
        )
        user_prompt = f"""
Research this query quickly but reliably.

QUERY: {query}
MODE: {mode}
METHODOLOGY: {methodology}

Return ONLY valid JSON with exactly this shape:
{{
  "executive_summary": "2-3 concise sentences",
  "key_points": ["...", "..."],
  "assumptions": ["..."],
  "uncertainties": ["..."],
  "sources": ["..."]
}}
"""
        started = time.perf_counter()
        raw = self._chat_json(model=model, system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.2)
        return raw, time.perf_counter() - started

    def _run_comparison(
        self,
        query: str,
        analytic_raw: str,
        alternative_raw: str,
    ) -> tuple[str, float]:
        system_prompt = (
            "You are a fast comparison and synthesis agent. "
            "Return valid JSON only with high signal and no fluff."
        )
        user_prompt = f"""
Compare these two fast analyses for the same query.

QUERY: {query}

ANALYTIC_FAST:
{analytic_raw[:3000]}

ALTERNATIVE_FAST:
{alternative_raw[:3000]}

Return ONLY valid JSON:
{{
  "consensus": ["..."],
  "disagreements": ["..."],
  "unique_insights": {{
    "analytic_fast": ["..."],
    "alternative_fast": ["..."]
  }},
  "blind_spots": ["..."],
  "final_synthesis": "3-5 sentences"
}}
"""
        started = time.perf_counter()
        raw = self._chat_json(
            model=self.synthesis_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
        )
        return raw, time.perf_counter() - started

    def run(self, query: str) -> dict:
        normalized_query = normalize_query(query)
        total_started = time.perf_counter()

        analytic_raw, analytic_secs = self._run_research(
            query=normalized_query,
            mode="analytic_fast",
            model=self.analytic_model,
            methodology="Facts first, uncertainty explicit, concise.",
        )
        alternative_raw, alternative_secs = self._run_research(
            query=normalized_query,
            mode="alternative_fast",
            model=self.alternative_model,
            methodology="Alternative angles, counter-arguments, concise.",
        )
        comparison_raw, comparison_secs = self._run_comparison(
            query=normalized_query,
            analytic_raw=analytic_raw,
            alternative_raw=alternative_raw,
        )
        total_secs = time.perf_counter() - total_started

        return {
            "query": normalized_query,
            "timing": {
                "total_seconds": round(total_secs, 3),
                "analytic_phase_seconds": round(analytic_secs, 3),
                "alternative_phase_seconds": round(alternative_secs, 3),
                "comparison_phase_seconds": round(comparison_secs, 3),
            },
            "model_config": {
                "analytic_fast": self.analytic_model,
                "alternative_fast": self.alternative_model,
                "synthesis_fast": self.synthesis_model,
            },
            "agent_outputs": {
                "analytic_fast": {
                    "raw": analytic_raw,
                    "structured": parse_research_response(analytic_raw),
                },
                "alternative_fast": {
                    "raw": alternative_raw,
                    "structured": parse_research_response(alternative_raw),
                },
            },
            "comparative_analysis": parse_comparison_response(comparison_raw),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast OpenRouter agent orchestrator (OpenRouteur)",
    )
    parser.add_argument(
        "--query",
        default="il y a entre 2 et 400 milliard d'étoiles dans le système solaire",
        help="DeepSearch query",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--no-pretty", action="store_true", help="Compact JSON output")
    parser.add_argument("--analytic-model", default=None, help="Override analytic fast model")
    parser.add_argument("--alternative-model", default=None, help="Override alternative fast model")
    parser.add_argument("--synthesis-model", default=None, help="Override synthesis model")
    args = parser.parse_args()

    if args.analytic_model:
        os.environ["OPENROUTER_FAST_ANALYTIC_MODEL"] = args.analytic_model
    if args.alternative_model:
        os.environ["OPENROUTER_FAST_ALTERNATIVE_MODEL"] = args.alternative_model
    if args.synthesis_model:
        os.environ["OPENROUTER_FAST_SYNTH_MODEL"] = args.synthesis_model

    indent = None if args.no_pretty else 2

    try:
        orchestrator = OpenRouteurOrchestrator()
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
