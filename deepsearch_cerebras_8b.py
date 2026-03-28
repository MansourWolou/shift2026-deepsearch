"""
DeepSearch Agent — LangChain + Cerebras (8B)
Par defaut, utilise un modele 8B via l'API OpenAI-compatible de Cerebras.
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()
console = Console()

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
CEREBRAS_BASE_URL = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
DEFAULT_MODEL = os.getenv("CEREBRAS_MODEL", "llama-3.1-8b")

SYSTEM_PROMPT = SystemMessage(content="""Tu es un agent de recherche approfondie (DeepSearch).
Ton objectif est de repondre a la question de l'utilisateur de facon exhaustive et precise
en effectuant plusieurs recherches successives et complementaires.

Strategie de recherche :
1. Decompose la question en sous-themes si necessaire.
2. Effectue plusieurs recherches ciblees (minimum 3, maximum 10).
3. Croise et synthetise les informations trouvees.
4. Identifie les lacunes et relance des recherches specifiques.
5. Fournis une reponse finale structuree en markdown, avec les sources citees.""")


def build_agent(model: str = DEFAULT_MODEL, temperature: float = 0.1):
    """Construit un agent DeepSearch base sur Cerebras + Tavily."""
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
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

    return create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)


def run(query: str, model: str = DEFAULT_MODEL) -> str:
    """Lance une recherche approfondie et retourne la reponse finale."""
    console.print(
        Panel(
            f"[bold cyan]DeepSearch Cerebras[/bold cyan] · [dim]{model}[/dim]\n[white]{query}[/white]",
            border_style="cyan",
        )
    )

    graph = build_agent(model=model)
    result = graph.invoke({"messages": [("user", query)]})
    answer = result["messages"][-1].content
    console.print(
        Panel(
            Markdown(answer),
            title="[bold green]Resultat[/bold green]",
            border_style="green",
        )
    )
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSearch Cerebras (8B)")
    parser.add_argument("query", nargs="*", help="Question a rechercher")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Modele Cerebras (defaut: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    query = " ".join(args.query).strip() if args.query else input("Entrez votre question : ").strip()
    if not query:
        console.print("[red]Aucune question fournie.[/red]")
        sys.exit(1)

    run(query, model=args.model)
