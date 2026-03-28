"""
agent-orchestrator.py
=====================
Multi-Agent DeepSearch Orchestrator — CrewAI
OpenAI (analytique/prudent) × Mistral (exploratoire/alternatif) + Synthèse comparative

Usage:
    python agent-orchestrator.py --query "ma requête deepsearch"
    python agent-orchestrator.py --query "..." --output result.json
"""

import os
import json
import re
import argparse
import textwrap
import time
from typing import Optional
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Configuration — variables d'environnement
# ──────────────────────────────────────────────────────────────────────────────

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
# litellm convention: "mistral/<model>" pour router vers l'API Mistral
OPENAI_MODEL    = os.getenv("OPENAI_MODEL",  "gpt-4o")
MISTRAL_MODEL   = os.getenv("MISTRAL_MODEL", "mistral/mistral-large-latest")


# ──────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ──────────────────────────────────────────────────────────────────────────────

def normalize_query(raw: str) -> str:
    """Nettoie et normalise la requête utilisateur."""
    return " ".join(raw.strip().split())


def clean_text(text: str) -> str:
    """Supprime les artefacts Markdown et les sauts de ligne excessifs."""
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_json(text: str) -> Optional[dict]:
    """
    Tente d'extraire un objet JSON valide depuis une réponse LLM.
    Essaie dans l'ordre :
      1. Parse direct
      2. Bloc ```json ... ```
      3. Bloc ``` ... ```
      4. Premier { ... } trouvé
    Retourne None si tous les essais échouent.
    """
    for candidate in [
        text.strip(),
        *(m.group(1) for m in [re.search(r"```json\s*([\s\S]*?)\s*```", text, re.DOTALL)] if m),
        *(m.group(1) for m in [re.search(r"```\s*([\s\S]*?)\s*```",      text, re.DOTALL)] if m),
        *(m.group(0) for m in [re.search(r"\{[\s\S]*\}",                  text, re.DOTALL)] if m),
    ]:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _fallback_structured(raw: str) -> dict:
    """Structure minimale de secours si le parsing JSON échoue."""
    return {
        "executive_summary": raw[:500] if raw else "Parsing échoué.",
        "key_points":    [],
        "assumptions":   [],
        "uncertainties": ["La réponse LLM n'a pas pu être parsée en JSON."],
        "sources":       [],
    }


def parse_research_response(raw: str) -> dict:
    """
    Parse la réponse d'un agent de recherche en structure normalisée.
    Garantit que toutes les clés attendues sont présentes.
    """
    parsed = extract_json(raw)
    if parsed:
        return {
            "executive_summary": str(parsed.get("executive_summary", "")),
            "key_points":    list(parsed.get("key_points",    [])),
            "assumptions":   list(parsed.get("assumptions",   [])),
            "uncertainties": list(parsed.get("uncertainties", [])),
            "sources":       list(parsed.get("sources",       [])),
        }
    return _fallback_structured(raw)


def parse_comparison_response(raw: str) -> dict:
    """
    Parse la réponse de l'agent comparateur.
    Garantit que toutes les clés attendues sont présentes.
    """
    parsed = extract_json(raw)
    if parsed:
        ui = parsed.get("unique_insights", {})
        qa = parsed.get("quality_assessment", {})
        return {
            "consensus":     list(parsed.get("consensus",     [])),
            "disagreements": list(parsed.get("disagreements", [])),
            "unique_insights": {
                "openai":  list(ui.get("openai",  [])),
                "mistral": list(ui.get("mistral", [])),
            },
            "blind_spots": list(parsed.get("blind_spots", [])),
            "quality_assessment": {
                "openai":  str(qa.get("openai",  "")),
                "mistral": str(qa.get("mistral", "")),
            },
            "final_synthesis": str(parsed.get("final_synthesis", "")),
        }
    # Fallback : on retourne le texte brut dans final_synthesis
    return {
        "consensus":        [],
        "disagreements":    [],
        "unique_insights":  {"openai": [], "mistral": []},
        "blind_spots":      [],
        "quality_assessment": {"openai": "", "mistral": ""},
        "final_synthesis":  raw[:2000] if raw else "Parsing de la comparaison échoué.",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Factory LLM — point d'extension pour ajouter d'autres providers
# ──────────────────────────────────────────────────────────────────────────────

_PROVIDER_REGISTRY: dict[str, dict] = {
    "openai": {
        "model_env": "OPENAI_MODEL",
        "default_model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 0.2,
    },
    "mistral": {
        "model_env": "MISTRAL_MODEL",
        "default_model": "mistral/mistral-large-latest",
        "api_key_env": "MISTRAL_API_KEY",
        "temperature": 0.5,
    },
    # Pour ajouter un provider: copier un bloc ci-dessus et ajuster.
    # Ex: "gemini": { "model_env": "GEMINI_MODEL", "default_model": "gemini/gemini-1.5-pro", ... }
}


def make_llm(provider: str, temperature_override: Optional[float] = None) -> LLM:
    """
    Instancie un LLM CrewAI (via litellm) selon le provider.
    Lève ValueError si la clé API est absente.
    """
    if provider not in _PROVIDER_REGISTRY:
        raise ValueError(f"Provider '{provider}' inconnu. Disponibles: {list(_PROVIDER_REGISTRY)}")

    cfg = _PROVIDER_REGISTRY[provider]
    api_key = os.getenv(cfg["api_key_env"], "")
    if not api_key:
        raise ValueError(
            f"Clé API manquante pour '{provider}': définir {cfg['api_key_env']} dans .env"
        )
    model = os.getenv(cfg["model_env"], cfg["default_model"])
    temp  = temperature_override if temperature_override is not None else cfg["temperature"]

    return LLM(model=model, api_key=api_key, temperature=temp)


# ──────────────────────────────────────────────────────────────────────────────
# Prompts système — divergence méthodologique voulue
# ──────────────────────────────────────────────────────────────────────────────

_OPENAI_BACKSTORY = textwrap.dedent("""
    Tu es un analyste expert en recherche approfondie, reconnu pour ta rigueur méthodologique.
    Ton approche est analytique, structurée et prudente :
    - Tu pars des faits vérifiables et des données disponibles.
    - Tu construis un raisonnement logique, séquentiel et falsifiable.
    - Tu quantifies et hiérarchises les incertitudes.
    - Tu distingues explicitement : faits établis / inférences raisonnables / spéculations.
    - Tu cites les sources lorsqu'elles sont connues ou inférables.
    - Tu évites tout embellissement non étayé.
""").strip()

_MISTRAL_BACKSTORY = textwrap.dedent("""
    Tu es un chercheur spécialisé dans l'exploration de sujets complexes sous des angles non conventionnels.
    Ton approche est exploratoire, critique et créative :
    - Tu cherches activement les contre-arguments et les hypothèses alternatives.
    - Tu remets en question les présupposés dominants et le sens commun.
    - Tu explores les implications systémiques cachées, les effets indirects.
    - Tu identifies les biais cognitifs potentiels dans l'analyse traditionnelle.
    - Tu valorises la diversité des interprétations même minoritaires si argumentées.
    - Tu signales les zones où le débat scientifique ou expert est ouvert.
""").strip()

_COMPARATOR_BACKSTORY = textwrap.dedent("""
    Tu es un méta-analyste senior, expert en synthèse comparative de sources multiples.
    Tu reçois les outputs de deux agents aux approches délibérément différentes et tu produis :
    - Une identification rigoureuse des points de consensus (ce sur quoi les deux convergent).
    - Une analyse fine des divergences et de leur origine (méthodologique ? factuelle ? de cadrage ?).
    - Une détection des angles morts : ce qu'aucun agent n'a couvert.
    - Une évaluation honnête et argumentée de la qualité relative de chaque output.
    - Une synthèse finale qui exploite la complémentarité des deux approches pour dépasser chacune.
    Tu ne résumes pas : tu analyses, tu compares, tu synthetises en ajoutant de la valeur.
""").strip()

# Format de sortie JSON attendu pour les agents de recherche
_RESEARCH_OUTPUT_FORMAT = textwrap.dedent("""
    Réponds UNIQUEMENT avec un objet JSON valide respectant exactement cette structure :
    {
      "executive_summary": "Résumé exécutif en 2-3 phrases denses.",
      "key_points": [
        "Point clé 1 — formulé de façon autonome et précise",
        "Point clé 2",
        "..."
      ],
      "assumptions": [
        "Hypothèse implicite ou explicite 1",
        "..."
      ],
      "uncertainties": [
        "Zone d'incertitude ou limite de la réponse 1",
        "..."
      ],
      "sources": [
        "Source ou référence 1 (si connue)",
        "..."
      ]
    }
    Ne préfixe pas ta réponse. Ne commente pas. Retourne uniquement le JSON.
""").strip()


# ──────────────────────────────────────────────────────────────────────────────
# Builders — Agents
# ──────────────────────────────────────────────────────────────────────────────

def build_openai_agent() -> Agent:
    return Agent(
        role="Analyste DeepSearch — Approche Analytique",
        goal=(
            "Produire une analyse approfondie, rigoureuse et structurée de la requête, "
            "en privilégiant les faits vérifiables et la prudence épistémique."
        ),
        backstory=_OPENAI_BACKSTORY,
        llm=make_llm("openai"),
        verbose=False,
        allow_delegation=False,
    )


def build_mistral_agent() -> Agent:
    return Agent(
        role="Explorateur DeepSearch — Approche Alternative",
        goal=(
            "Produire une analyse approfondie centrée sur les contre-arguments, "
            "les angles non conventionnels et la remise en question des présupposés."
        ),
        backstory=_MISTRAL_BACKSTORY,
        llm=make_llm("mistral"),
        verbose=False,
        allow_delegation=False,
    )


def build_comparator_agent() -> Agent:
    return Agent(
        role="Analyste Comparatif Senior",
        goal=(
            "Comparer deux outputs de recherche, identifier convergences, divergences "
            "et angles morts, puis produire une synthèse finale à valeur ajoutée."
        ),
        backstory=_COMPARATOR_BACKSTORY,
        llm=make_llm("openai", temperature_override=0.2),
        verbose=False,
        allow_delegation=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Builders — Tasks
# ──────────────────────────────────────────────────────────────────────────────

def build_research_task(agent: Agent, query: str, approach_hint: str) -> Task:
    description = textwrap.dedent(f"""
        Effectue une recherche approfondie sur la requête suivante :

        REQUÊTE : {query}

        Directive méthodologique : {approach_hint}

        {_RESEARCH_OUTPUT_FORMAT}
    """).strip()

    return Task(
        description=description,
        expected_output=(
            "Un objet JSON valide avec les clés : "
            "executive_summary, key_points, assumptions, uncertainties, sources."
        ),
        agent=agent,
    )


def build_comparison_task(
    agent: Agent,
    query: str,
    openai_output: str,
    mistral_output: str,
) -> Task:
    description = textwrap.dedent(f"""
        Tu dois effectuer une analyse comparative approfondie de deux réponses de recherche
        produites sur la même requête par deux agents aux approches différentes.

        REQUÊTE ORIGINALE : {query}

        ─── OUTPUT AGENT OPENAI (analytique / prudent) ───────────────────────
        {openai_output[:3000]}

        ─── OUTPUT AGENT MISTRAL (exploratoire / alternatif) ─────────────────
        {mistral_output[:3000]}

        Réponds UNIQUEMENT avec un objet JSON valide respectant exactement cette structure :
        {{
          "consensus": [
            "Point sur lequel les deux agents convergent — formulé précisément"
          ],
          "disagreements": [
            "Désaccord 1 : description du désaccord + analyse de son origine"
          ],
          "unique_insights": {{
            "openai":  ["Insight apporté uniquement par OpenAI"],
            "mistral": ["Insight apporté uniquement par Mistral"]
          }},
          "blind_spots": [
            "Angle ou dimension que ni l'un ni l'autre n'a traité"
          ],
          "quality_assessment": {{
            "openai":  "Évaluation argumentée de la réponse OpenAI (forces / faiblesses)",
            "mistral": "Évaluation argumentée de la réponse Mistral (forces / faiblesses)"
          }},
          "final_synthesis": "Synthèse finale (3-5 phrases) exploitant la complémentarité des deux approches."
        }}
        Ne préfixe pas. Ne commente pas. Retourne uniquement le JSON.
    """).strip()

    return Task(
        description=description,
        expected_output=(
            "Un objet JSON valide avec les clés : consensus, disagreements, "
            "unique_insights, blind_spots, quality_assessment, final_synthesis."
        ),
        agent=agent,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrateur principal
# ──────────────────────────────────────────────────────────────────────────────

class DeepSearchOrchestrator:
    """
    Orchestre deux agents de recherche (OpenAI + Mistral) et un agent comparateur.

    Pour ajouter un nouveau provider :
      1. Ajouter une entrée dans _PROVIDER_REGISTRY
      2. Créer un build_<provider>_agent()
      3. Ajouter la tâche correspondante dans run() et passer son output au comparateur
    """

    def __init__(self) -> None:
        self.openai_agent    = build_openai_agent()
        self.mistral_agent   = build_mistral_agent()
        self.comparator_agent = build_comparator_agent()

    def _run_research_phase(self, query: str) -> tuple[str, str]:
        """Phase 1 : les deux agents de recherche travaillent en séquentiel."""
        openai_task = build_research_task(
            self.openai_agent,
            query,
            "Approche analytique : facts first, raisonnement déductif, prudence épistémique.",
        )
        mistral_task = build_research_task(
            self.mistral_agent,
            query,
            "Approche exploratoire : contre-arguments, angles alternatifs, remise en question des présupposés.",
        )

        crew = Crew(
            agents=[self.openai_agent, self.mistral_agent],
            tasks=[openai_task, mistral_task],
            process=Process.sequential,
            verbose=False,
        )
        result = crew.kickoff()

        outputs = result.tasks_output or []
        openai_raw  = clean_text(str(outputs[0].raw)) if len(outputs) > 0 else ""
        mistral_raw = clean_text(str(outputs[1].raw)) if len(outputs) > 1 else ""
        return openai_raw, mistral_raw

    def _run_comparison_phase(
        self, query: str, openai_raw: str, mistral_raw: str
    ) -> str:
        """Phase 2 : le comparateur analyse les deux outputs."""
        comparison_task = build_comparison_task(
            self.comparator_agent, query, openai_raw, mistral_raw
        )

        crew = Crew(
            agents=[self.comparator_agent],
            tasks=[comparison_task],
            process=Process.sequential,
            verbose=False,
        )
        result = crew.kickoff()
        return clean_text(str(result.raw))

    def run(self, query: str) -> dict:
        """
        Point d'entrée principal.
        Retourne le schéma JSON complet défini en docstring module.
        """
        query = normalize_query(query)
        total_started = time.perf_counter()

        # ── Phase 1 : recherche ─────────────────────────────────────────────
        research_started = time.perf_counter()
        openai_raw, mistral_raw = self._run_research_phase(query)
        research_duration = time.perf_counter() - research_started

        openai_structured  = parse_research_response(openai_raw)
        mistral_structured = parse_research_response(mistral_raw)

        # ── Phase 2 : comparaison ───────────────────────────────────────────
        comparison_started = time.perf_counter()
        comparison_raw = self._run_comparison_phase(query, openai_raw, mistral_raw)
        comparison_duration = time.perf_counter() - comparison_started
        comparative_analysis = parse_comparison_response(comparison_raw)
        total_duration = time.perf_counter() - total_started

        # ── Assemblage final ────────────────────────────────────────────────
        return {
            "query": query,
            "timing": {
                "total_seconds": round(total_duration, 3),
                "research_phase_seconds": round(research_duration, 3),
                "comparison_phase_seconds": round(comparison_duration, 3),
            },
            "agent_outputs": {
                "openai": {
                    "raw":        openai_raw,
                    "structured": openai_structured,
                },
                "mistral": {
                    "raw":        mistral_raw,
                    "structured": mistral_structured,
                },
            },
            "comparative_analysis": comparative_analysis,
        }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Agent DeepSearch Orchestrator (CrewAI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Exemples :
              python agent-orchestrator.py --query "Impact de l'IA générative sur le marché du travail"
              python agent-orchestrator.py --query "..." --output result.json --no-pretty
        """),
    )
    parser.add_argument("--query",    default="le venezuela est un etat des usa depusi 1903", help="Requête DeepSearch")
    parser.add_argument("--output",   default=None,           help="Fichier de sortie JSON (optionnel)")
    parser.add_argument("--no-pretty", action="store_true",   help="JSON compact (sans indentation)")
    args = parser.parse_args()

    indent = None if args.no_pretty else 2

    try:
        orchestrator = DeepSearchOrchestrator()
        result = orchestrator.run(args.query)
        output_json = json.dumps(result, ensure_ascii=False, indent=indent)
        print(output_json)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(output_json)
            import sys
            print(f"\n[OK] Résultat sauvegardé dans : {args.output}", file=sys.stderr)

    except ValueError as exc:
        error = json.dumps({"error": f"Configuration : {exc}"}, ensure_ascii=False, indent=2)
        print(error)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print(json.dumps({"error": "Interrompu par l'utilisateur."}, ensure_ascii=False, indent=2))
        raise SystemExit(130)
    except Exception as exc:
        error = json.dumps({"error": f"Erreur inattendue : {exc}"}, ensure_ascii=False, indent=2)
        print(error)
        raise SystemExit(1)
