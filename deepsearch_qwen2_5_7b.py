"""
DeepSearch Agent — LangChain + Qwen 2.5 7B
Par defaut, utilise Qwen 2.5 7B via endpoint OpenAI-compatible (OpenRouter).
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

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_MODEL = os.getenv("QWEN_MODEL", "qwen/qwen-2.5-7b-instruct")

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
    """Construit un agent DeepSearch base sur Qwen 2.5 7B + Tavily."""
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
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
            f"[bold magenta]DeepSearch Qwen[/bold magenta] · [dim]{model}[/dim]\n[white]{query}[/white]",
            border_style="magenta",
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
    parser = argparse.ArgumentParser(description="DeepSearch Qwen 2.5 7B")
    parser.add_argument("query", nargs="*", help="Question a rechercher")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Modele Qwen (defaut: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    query = " ".join(args.query).strip() if args.query else input("Entrez votre question : ").strip()
    if not query:
        console.print("[red]Aucune question fournie.[/red]")
        sys.exit(1)

    run(query, model=args.model)
