"""
DeepSearch Agent — LangChain + Perplexity AI
Deux modes :
  • sonar-deep-research  → recherche native Perplexity (pas d'outils externes)
  • sonar-pro            → agent ReAct + Tavily (même pattern que les autres agents)
"""

import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents import create_agent

load_dotenv()
console = Console()
PPLX_API_KEY = os.getenv("PPLX_API_KEY")
PERPLEXITY_BASE_URL = "https://api.perplexity.ai"

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
# Helpers LLM
# ---------------------------------------------------------------------------

def _make_llm(model: str, temperature: float = 0.1) -> ChatOpenAI:
    """Instancie un ChatOpenAI pointant vers l'API Perplexity."""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=PPLX_API_KEY,
        base_url=PERPLEXITY_BASE_URL,
    )


# ---------------------------------------------------------------------------
# Mode 1 : agent ReAct + Tavily (sonar-pro)
# ---------------------------------------------------------------------------

def build_agent(model: str = "sonar-pro", temperature: float = 0.1):
    """Construit un agent ReAct DeepSearch avec Tavily comme outil de recherche."""
    llm = _make_llm(model, temperature)

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
# Mode 2 : recherche native sonar-deep-research (sans outils externes)
# ---------------------------------------------------------------------------

def deep_research(query: str, temperature: float = 0.1) -> str:
    """Appel direct à sonar-deep-research — recherche gérée nativement par Perplexity."""
    llm = _make_llm("sonar-deep-research", temperature)
    messages = [SYSTEM_PROMPT, HumanMessage(content=query)]
    response = llm.invoke(messages)
    return response.content


# ---------------------------------------------------------------------------
# Interface CLI
# ---------------------------------------------------------------------------

def run(query: str, model: str = "sonar-deep-research") -> str:
    """Lance une recherche approfondie et retourne la réponse finale.

    Par défaut utilise sonar-deep-research (recherche native Perplexity).
    Passer model='sonar-pro' pour le mode agent ReAct + Tavily.
    """
    console.print(
        Panel(f"[bold blue]DeepSearch Perplexity[/bold blue] · [dim]{model}[/dim]\n[white]{query}[/white]",
              border_style="blue")
    )

    if model == "sonar-deep-research":
        answer = deep_research(query)
    else:
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

    parser = argparse.ArgumentParser(description="DeepSearch Perplexity")
    parser.add_argument("query", nargs="*", help="Question à rechercher")
    parser.add_argument(
        "--model",
        default="sonar-deep-research",
        choices=["sonar-deep-research", "sonar-pro", "sonar"],
        help="Modèle Perplexity à utiliser (défaut: sonar-deep-research)",
    )
    args = parser.parse_args()

    query = " ".join(args.query).strip() if args.query else input("Entrez votre question : ").strip()

    if not query:
        console.print("[red]Aucune question fournie.[/red]")
        sys.exit(1)

    run(query, model=args.model)
