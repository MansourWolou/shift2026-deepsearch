"""
DeepSearch Agent — LangChain + LangGraph + OpenAI
Effectue une recherche itérative et approfondie sur un sujet donné.
"""

import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage
from langchain.agents import create_agent

load_dotenv()
console = Console()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

def build_agent(model: str = "gpt-4o", temperature: float = 0.1):
    """Construit et retourne le graph ReAct DeepSearch."""
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=OPENAI_API_KEY,
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

def run(query: str, model: str = "gpt-4o") -> str:
    """Lance une recherche approfondie et retourne la réponse finale."""
    console.print(
        Panel(f"[bold cyan]DeepSearch[/bold cyan]\n[white]{query}[/white]",
              border_style="cyan")
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

    if len(sys.argv) < 2:
        query = input("Entrez votre question : ").strip()
    else:
        query = " ".join(sys.argv[1:])

    if not query:
        console.print("[red]Aucune question fournie.[/red]")
        sys.exit(1)

    run(query)
