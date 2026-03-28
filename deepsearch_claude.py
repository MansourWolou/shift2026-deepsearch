"""
DeepSearch Agent — LangChain + Anthropic Claude
Modèles disponibles :
  • claude-sonnet-4-5   → équilibre performance / coût (défaut)
  • claude-opus-4-5     → plus puissant
  • claude-haiku-4-5    → plus rapide / moins coûteux
"""

import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage
from langchain.agents import create_agent

load_dotenv()
console = Console()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ---------------------------------------------------------------------------
# Système prompt DeepSearch
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = SystemMessage(content="""Tu es un agent de recherche approfondie (DeepSearch).
Ton objectif est de répondre à la question de l'utilisateur de façon exhaustive et précise
en effectuant plusieurs recherches successives et complémentaires.

Stratégie de recherche :
1. Décompose la question en sous-thèmes si nécessaire.
2. Effectue plusieurs recherches ciblées (minimum 3, maximum 10).
3. Croise et synthétise les informations trouvées.
4. Identifie les lacunes et relance des recherches spécifiques.
5. Fournis une réponse finale structurée en markdown, avec les sources citées.""")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def build_agent(model: str = "claude-sonnet-4-5", temperature: float = 0.1):
    """Construit un agent ReAct DeepSearch basé sur Claude + Tavily."""
    llm = ChatAnthropic(
        model=model,
        temperature=temperature,
        api_key=ANTHROPIC_API_KEY,
    )

    tools = [
        TavilySearch(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
            include_images=False,
        )
    ]

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )


# ---------------------------------------------------------------------------
# Interface CLI
# ---------------------------------------------------------------------------

def run(query: str, model: str = "claude-sonnet-4-5") -> str:
    """Lance une recherche approfondie et retourne la réponse finale."""
    console.print(
        Panel(f"[bold orange1]DeepSearch Claude[/bold orange1] · [dim]{model}[/dim]\n[white]{query}[/white]",
              border_style="orange1")
    )

    graph = build_agent(model=model)
    result = graph.invoke({"messages": [("user", query)]})
    answer = result["messages"][-1].content

    console.print(Panel(Markdown(answer), title="[bold green]Résultat[/bold green]", border_style="green"))
    return answer


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="DeepSearch Claude (Anthropic)")
    parser.add_argument("query", nargs="*", help="Question à rechercher")
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-5",
        choices=["claude-sonnet-4-5", "claude-opus-4-5", "claude-haiku-4-5"],
        help="Modèle Claude à utiliser (défaut: claude-sonnet-4-5)",
    )
    args = parser.parse_args()

    query = " ".join(args.query).strip() if args.query else input("Entrez votre question : ").strip()

    if not query:
        console.print("[red]Aucune question fournie.[/red]")
        sys.exit(1)

    run(query, model=args.model)
