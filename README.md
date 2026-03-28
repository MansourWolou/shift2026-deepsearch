# Fact-Check Engine

Multi-agent fact-checking : plusieurs LLMs analysent en parallele les memes sources web, puis un modele de synthese produit un verdict structure.

## Comment ca marche

```
                          ┌─────────────────┐
                          │  1. LINKUP SEARCH │
                          │  (recherche web)  │
                          └────────┬──────────┘
                                   │
                          answer + sources
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
     ┌────────▼────────┐ ┌────────▼────────┐  ┌────────▼────────┐
     │  Agent Gemini    │ │  Agent Mistral   │  │  Agent o4-mini   │
     │  (analyse LLM)   │ │  (analyse LLM)   │  │  (analyse LLM)   │
     └────────┬────────┘ └────────┬────────┘  └────────┬────────┘
              │                    │                     │
              └────────────────────┼────────────────────┘
                                   │
                          N rapports d'analyse
                                   │
                          ┌────────▼──────────┐
                          │  3. SYNTHESE       │
                          │  (thinking model)  │
                          └────────┬──────────┘
                                   │
                            Verdict JSON
```

### Etape 1 — Recherche web (Linkup)

**Qui :** API Linkup (pas un LLM)
**Quoi :** Recherche web sur la claim, retourne une reponse sourcee + liste de sources (url, snippet, nom)
**Profondeur :** `standard` (rapide, ~2-3s) ou `deep` (plus complet, ~6-8s)

> Linkup fait TOUTE la recherche web. Les LLMs ne font PAS de recherche eux-memes.
> Ils n'ont pas acces a internet, pas de Tavily, pas de browsing. Ils recoivent
> uniquement le texte que Linkup a trouve.

### Etape 2 — Analyse parallele (N agents LLM)

**Qui :** N modeles LLM differents, appeles via OpenRouter
**Quoi :** Chaque agent recoit exactement le meme input :
- La claim a verifier
- La reponse Linkup (texte)
- Les sources Linkup (url + snippet)

**Ce qu'ils font :** Chaque agent analyse les resultats et produit :
1. Ce que les sources confirment ou infirment
2. La qualite et fiabilite des sources
3. Les nuances ou contradictions
4. Son evaluation preliminaire (vrai/faux/partiellement vrai/inverifiable/trompeur)

**Parallelisme :** Tous les agents tournent en meme temps via `ThreadPoolExecutor`.
Le wall-clock time = le temps du plus lent. En mode fast (2 agents), ca prend ~15s.
En mode thorough (5 agents), ~40-100s selon les modeles.

**Pourquoi plusieurs modeles ?** Diversite de raisonnement. Un modele peut rater un detail
qu'un autre capte. La synthese repere les consensus et les desaccords.

> Les agents n'ont AUCUN outil. Pas de web search, pas de code execution.
> Ce sont des appels LLM purs (chat completion) qui analysent du texte.

### Etape 3 — Synthese (thinking model)

**Qui :** Un seul modele de synthese (par defaut Gemini 2.5 Flash via OpenRouter, configurable)
**Quoi :** Recoit les N rapports d'analyse et produit un verdict JSON structure :

```json
{
  "claim": "La claim originale",
  "verdict": "TRUE | FALSE | PARTIALLY TRUE | UNVERIFIABLE | MISLEADING",
  "confidence": 0.95,
  "summary": "Explication en 2-3 phrases",
  "evidence_for": [{"point": "...", "sources": ["url"], "agents": ["agent"]}],
  "evidence_against": [...],
  "consensus": ["Points sur lesquels tous les agents sont d'accord"],
  "disagreements": [{"topic": "...", "positions": {"agent1": "...", "agent2": "..."}}],
  "blind_spots": ["Ce qu'aucun agent n'a couvert"],
  "nuances": ["Caveats importants"]
}
```

**Son role :** Cross-reference les rapports, detecter les biais partages, evaluer la qualite
des sources, et rendre un verdict avec un score de confiance calibre.

## Quick Start

```bash
# Install
uv sync

# Remplir les cles API dans .env
#   OPENROUTER_API_KEY=sk-or-...    (requis — tous les LLMs passent par OpenRouter)
#   LINKUP_API_KEY=lk-...           (requis — recherche web)

# Lancer (mode default : 3 agents)
uv run python factcheck.py "Emmanuel Macron a ete elu president en 2017"
```

## Modes

3 presets selon le use-case :

| Mode | Agents | Recherche | Temps approx. | Usage |
|------|--------|-----------|---------------|-------|
| `--mode fast` | 2 (gemini-flash, o4-mini) | standard | ~30s | **Prod** — rapide, pas cher |
| `--mode default` | 3 (gemini-flash, mistral, o4-mini) | deep | ~50s | Equilibre |
| `--mode thorough` | 5 (claude, openai, gemini, mistral, grok) | deep | ~120s | **Demo** — max diversite |

```bash
uv run python factcheck.py --mode fast "..."       # prod-like
uv run python factcheck.py --mode thorough "..."    # demo
```

## Catalogue d'agents

Tous appeles via OpenRouter. Aucun n'a acces au web — ils analysent les resultats Linkup.

### Frontier (+ cher, + capables)

| Alias | Modele reel (OpenRouter) | Pourquoi |
|-------|--------------------------|----------|
| `claude` | `google/gemini-2.5-flash` | Bon raisonnement, nuance |
| `openai` | `openai/gpt-5.4` | Fort en factuel, structure |
| `gemini` | `google/gemini-2.5-pro` | Large contexte, bon sur sources |
| `grok` | `x-ai/grok-4` | Perspective differente, actualite |

### Rapport qualite/prix

| Alias | Modele reel (OpenRouter) | Pourquoi |
|-------|--------------------------|----------|
| `mistral` | `mistralai/mistral-small-3.2-24b-instruct` | Rapide, bon en francais |
| `deepseek` | `deepseek/deepseek-v3.2` | Raisonnement solide, pas cher |
| `qwen` | `qwen/qwen3-max` | Diversite provider |

### Rapides / pas cher

| Alias | Modele reel (OpenRouter) | Pourquoi |
|-------|--------------------------|----------|
| `gemini-flash` | `google/gemini-2.5-flash` | Tres rapide (~5s), bon niveau |
| `gpt-mini` | `openai/gpt-5-mini` | Rapide, pas cher |
| `o4-mini` | `openai/o4-mini` | Reasoning model, bonne precision |

## Ce que chaque composant fait (et ne fait PAS)

| Composant | Fait | Ne fait PAS |
|-----------|------|-------------|
| **Linkup** | Recherche web, retourne sources + reponse | Pas d'analyse, pas de verdict |
| **Agents LLM** | Analysent le texte Linkup, evaluent les sources | Pas de web search, pas d'outils, pas d'acces internet |
| **Synthese** | Cross-reference N rapports, produit verdict JSON | Pas de recherche supplementaire |

## Options CLI

```
--mode fast|default|thorough   Preset (agents + profondeur recherche)
--agents claude,openai,...     Override manuel des agents (ecrase le preset)
--synthesis-model MODEL        Modele de synthese (default: google/gemini-2.5-flash)
--output result.json           Export JSON
--html [report.html]           Rapport HTML avec visu du parallelisme (dev/demo)
--verbose                      Inclut les donnees brutes dans le JSON
--log-level debug|info|warning Logs detailles sur le parallelisme
--no-pretty                    JSON compact
```

## Rapport HTML (dev/demo)

```bash
uv run python factcheck.py --mode thorough --html "..."
# → report.html avec verdict, evidence, barres de timing des agents
```

> Le HTML est un outil de dev/demo. En prod on recup l'input et on renvoie le JSON.

## Logs parallelisme

```bash
# Voir quel agent demarre/finit et en combien de temps
uv run python factcheck.py --log-level info "..."

# Tout le detail (modeles, chars, etc)
uv run python factcheck.py --log-level debug --mode thorough "..."
```

## Variables d'environnement

```
OPENROUTER_API_KEY=         # Requis — cle OpenRouter (tous les LLMs)
LINKUP_API_KEY=             # Requis — cle Linkup search
FACTCHECK_MODE=             # Optionnel — fast/default/thorough
FACTCHECK_AGENTS=           # Optionnel — override agents (comma-separated)
FACTCHECK_SYNTHESIS_MODEL=  # Optionnel — modele de synthese
```

## Priorite de config

```
CLI args  >  mode preset  >  env vars  >  defaults hardcodes
```

Exemple : `--mode fast` ecrase `FACTCHECK_AGENTS` du `.env`.
Mais `--agents gemini,openai` ecrase tout (mode + env).

## Setup dev

```bash
uv sync                        # install deps + dev tools
uv run pre-commit install      # active black + ruff sur chaque commit
uv run black factcheck.py      # formattage
uv run ruff check factcheck.py # lint
```

## Exemples

```bash
# Test rapide prod-like
uv run python factcheck.py --mode fast "La tour Eiffel mesure 330 metres"

# Demo complete avec HTML + logs
uv run python factcheck.py --mode thorough --html --log-level info "La France a 67 millions d'habitants"

# 2 agents specifiques + export JSON
uv run python factcheck.py --agents gemini,openai --output result.json "Le Bitcoin a ete cree en 2009"

# Changer le modele de synthese
uv run python factcheck.py --synthesis-model openai/o3 "..."
```
