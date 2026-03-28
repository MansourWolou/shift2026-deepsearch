"""
DeepSearch Agent — LangChain + LangGraph + Mistral
Effectue une recherche itérative et approfondie sur un sujet donné.
"""

import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from langchain_mistralai import ChatMistralAI
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage
from langchain.agents import create_agent

load_dotenv()
console = Console()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

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

def build_agent(model: str = "mistral-large-latest", temperature: float = 0.1):
    """Construit et retourne le graph ReAct DeepSearch."""
    llm = ChatMistralAI(
        model=model,
        temperature=temperature,
        api_key=MISTRAL_API_KEY,
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

def run(query: str, model: str = "mistral-large-latest") -> str:
    """Lance une recherche approfondie et retourne la réponse finale."""
    console.print(
        Panel(f"[bold cyan]DeepSearch — Mistral[/bold cyan]\n[white]{query}[/white]",
              border_style="cyan")
    )

    graph = build_agent(model=model)
    result = graph.invoke({"messages": [("user", query)]})
    raw = result["messages"][-1].content
    answer = "".join(c["text"] if isinstance(c, dict) else c for c in raw) if isinstance(raw, list) else raw

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
