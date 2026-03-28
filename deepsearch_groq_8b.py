"""
DeepSearch Agent — LangChain + Groq (8B)
Par defaut, utilise un modele 8B via l'API OpenAI-compatible de Groq.
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from openai import APIStatusError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()
console = Console()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

SYSTEM_PROMPT = SystemMessage(content="""Tu es un agent de recherche approfondie (DeepSearch).
Ton objectif est de repondre a la question de l'utilisateur de facon exhaustive et precise
en effectuant plusieurs recherches successives et complementaires.

Strategie de recherche :
1. Decompose la question en sous-themes si necessaire.
2. Effectue plusieurs recherches ciblees (minimum 2, maximum 4).
3. Croise et synthetise les informations trouvees.
4. Identifie les lacunes et relance des recherches specifiques.
5. Garde un raisonnement concis et limite les repetitions.
6. Fournis une reponse finale structuree en markdown, avec les sources citees.""")

LITE_SYSTEM_PROMPT = SystemMessage(content="""Tu es un assistant de recherche web.
Tu dois fournir une reponse concise, fiable et structuree en markdown avec sources.
Utilise uniquement le contexte fourni et signale clairement les incertitudes.""")


def _make_llm(model: str = DEFAULT_MODEL, temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
    )


def build_agent(model: str = DEFAULT_MODEL, temperature: float = 0.1):
    """Construit un agent DeepSearch base sur Groq + Tavily."""
    llm = _make_llm(model=model, temperature=temperature)

    tools = [
        TavilySearch(
            max_results=3,
            search_depth="basic",
            include_answer=False,
            include_raw_content=False,
            include_images=False,
        )
    ]

    return create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    out.append(text)
            elif isinstance(item, str):
                out.append(item)
        return "\n".join(out)
    return str(content)


def _format_search_context(payload: dict, max_chars: int = 2400) -> str:
    results = payload.get("results", []) if isinstance(payload, dict) else []
    chunks = []
    for r in results[:3]:
        title = r.get("title", "") if isinstance(r, dict) else ""
        url = r.get("url", "") if isinstance(r, dict) else ""
        content = r.get("content", "") if isinstance(r, dict) else ""
        snippet = (content or "")[:350].replace("\n", " ")
        chunks.append(f"- {title}\n  URL: {url}\n  Extrait: {snippet}")
    context = "\n".join(chunks).strip()
    return context[:max_chars]


def run_lite(query: str, model: str = DEFAULT_MODEL) -> str:
    """Fallback legere: une recherche web + une synthese LLM."""
    search = TavilySearch(
        max_results=3,
        search_depth="basic",
        include_answer=False,
        include_raw_content=False,
        include_images=False,
    )
    payload = search.invoke({"query": query})
    context = _format_search_context(payload)
    llm = _make_llm(model=model, temperature=0.1)
    response = llm.invoke(
        [
            LITE_SYSTEM_PROMPT,
            HumanMessage(
                content=(
                    f"Question:\n{query}\n\n"
                    f"Contexte web (Tavily):\n{context}\n\n"
                    "Produis une reponse utile avec sources (URLs)."
                )
            ),
        ]
    )
    return _extract_text(response.content)


def run(query: str, model: str = DEFAULT_MODEL) -> str:
    """Lance une recherche approfondie et retourne la reponse finale."""
    console.print(
        Panel(
            f"[bold yellow]DeepSearch Groq[/bold yellow] · [dim]{model}[/dim]\n[white]{query}[/white]",
            border_style="yellow",
        )
    )

    graph = build_agent(model=model)
    try:
        result = graph.invoke(
            {"messages": [("user", query)]},
            config={"recursion_limit": 8},
        )
        answer = _extract_text(result["messages"][-1].content)
    except APIStatusError as exc:
        if exc.status_code == 413:
            console.print(
                "[yellow]Requete trop grande pour le quota TPM du modele. "
                "Retry automatique en mode lite.[/yellow]"
            )
            answer = run_lite(query, model=model)
        else:
            raise

    console.print(
        Panel(
            Markdown(answer),
            title="[bold green]Resultat[/bold green]",
            border_style="green",
        )
    )
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSearch Groq (8B)")
    parser.add_argument("query", nargs="*", help="Question a rechercher")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Modele Groq (defaut: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    query = " ".join(args.query).strip() if args.query else input("Entrez votre question : ").strip()
    if not query:
        console.print("[red]Aucune question fournie.[/red]")
        sys.exit(1)

    run(query, model=args.model)
