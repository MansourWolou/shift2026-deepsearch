# Fact-Check Engine

Multi-agent fact-checking : plusieurs LLMs analysent en parallele les memes sources web, puis un modele de synthese produit un verdict structure.

## Comment ca marche

```
  Claim
    │
    ▼
┌─────────────────┐
│  0. PRE-SCREENING│  ← Gemini Flash (~1s) : la claim est-elle fact-checkable ?
│  (idiom/opinion/ │     Si non → NOT_CHECKABLE, pipeline stoppe
│   question/...)  │
└────────┬─────────┘
         │ checkable=true
         ▼
┌─────────────────┐
│  1. LINKUP SEARCH│  ← API Linkup (pas un LLM) : recherche web
│  (recherche web) │     Retourne answer + sources (url, snippet)
└────────┬─────────┘
         │
         ▼
┌────────┬────────┬────────┐
│Agent 1 │Agent 2 │Agent N │  ← N LLMs en parallele via OpenRouter
│(LLM)   │(LLM)   │(LLM)  │     Analysent les MEMES sources Linkup
└────┬───┘────┬───┘────┬───┘     Meme prompt (generic) ou roles differents (--specialized)
     │        │        │
     └────────┼────────┘
              │
              ▼
┌──────────────────┐
│  3. SYNTHESE     │  ← 1 thinking model cross-reference les N rapports
│  (thinking model)│     Produit verdict JSON structure
└────────┬─────────┘
         │
    Verdict JSON
```

### Etape 0 — Pre-screening

**Qui :** Gemini Flash (rapide, ~1s)
**Quoi :** Classifie la claim avant de lancer le pipeline complet :
- `factual` → checkable, on continue
- `opinion`, `idiom`, `question`, `vague`, `subjective`, `future` → NOT_CHECKABLE, on stoppe

**Pourquoi :** Evite de cramer des tokens et du temps sur des claims non factuelles.
Sans ca, "le verre est a moitie plein" lancait 5 agents pendant 90s pour produire
un verdict UNVERIFIABLE avec 92% de confiance — de la bullshit structuree.

> Bypass avec `--no-screen` si besoin.

### Etape 1 — Recherche web (Linkup)

**Qui :** API Linkup (pas un LLM)
**Quoi :** Recherche web sur la claim, retourne une reponse sourcee + liste de sources (url, snippet, nom)
**Profondeur :** `standard` (rapide, ~2-3s) ou `deep` (plus complet, ~6-8s)

Linkup a son propre index web (pas Google). En mode `deep`, il fait un workflow
agentique iteratif si la premiere passe ne trouve pas assez d'info.

> Linkup fait TOUTE la recherche web. Les LLMs ne font PAS de recherche eux-memes.
> Ils n'ont pas acces a internet, pas de Tavily, pas de browsing. Ils recoivent
> uniquement le texte que Linkup a trouve.

### Etape 2 — Analyse parallele (N agents LLM)

**Qui :** N modeles LLM differents, appeles via OpenRouter
**Quoi :** Chaque agent recoit exactement le meme input (sources Linkup)

**Deux modes :**

**Generic (par defaut)** — tous les agents ont le meme prompt. On compare comment
differents modeles raisonnent sur les memes faits. Utile pour benchmarker.

**Specialized (`--specialized`)** — chaque agent a un role different :

| Role | Mission |
|------|---------|
| **Verificateur** | Faits bruts : dates, chiffres, noms. Confirme ou infirme chaque donnee |
| **Avocat du diable** | Cherche activement a infirmer la claim. Contre-arguments, failles |
| **Analyste sources** | Evalue la fiabilite des sources : type, biais, primaire vs secondaire |
| **Contextualiste** | Contexte historique, temporel, geographique. Blind spots, nuances |

Les roles sont assignes dans l'ordre de la liste d'agents (cyclique si plus de 4 agents).

**Parallelisme :** Tous les agents tournent en meme temps via `ThreadPoolExecutor`.
Le wall-clock time = le temps du plus lent.

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
  "evidence_for": [...],
  "evidence_against": [...],
  "consensus": ["Points d'accord entre agents"],
  "disagreements": [{"topic": "...", "positions": {...}}],
  "blind_spots": ["Ce qu'aucun agent n'a couvert"],
  "nuances": ["Caveats importants"]
}
```

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

| Mode | Agents | Recherche | Temps approx. | Usage |
|------|--------|-----------|---------------|-------|
| `--mode fast` | 2 (gemini-flash, o4-mini) | standard | ~30s | **Prod** — rapide, pas cher |
| `--mode default` | 3 (gemini-flash, mistral, o4-mini) | deep | ~50s | Equilibre |
| `--mode thorough` | 5 (claude, openai, gemini, mistral, grok) | deep | ~120s | **Demo** — max diversite |

## Options CLI

```
--mode fast|default|thorough   Preset (agents + profondeur recherche)
--agents claude,openai,...     Override manuel des agents (ecrase le preset)
--specialized                  Roles differents par agent (vs meme prompt pour tous)
--synthesis-model MODEL        Modele de synthese (default: google/gemini-2.5-flash)
--output result.json           Export JSON
--html [reports/]              Rapport HTML unique + index (dev/demo)
--verbose                      Inclut les donnees brutes dans le JSON
--log-level debug|info|warning Logs detailles sur le parallelisme
--no-screen                    Bypass le pre-screening
--no-pretty                    JSON compact
--test-all                     Lance le test complet (toutes combinaisons)
```

## Catalogue d'agents

Tous appeles via OpenRouter. Aucun n'a acces au web — ils analysent les resultats Linkup.

### Frontier

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
| **Pre-screening** | Classifie la claim (factuelle ou pas) | Pas de recherche, pas de verdict |
| **Linkup** | Recherche web, retourne sources + reponse | Pas d'analyse, pas de verdict |
| **Agents LLM** | Analysent le texte Linkup, evaluent les sources | Pas de web search, pas d'outils, pas d'acces internet |
| **Synthese** | Cross-reference N rapports, produit verdict JSON | Pas de recherche supplementaire |

## Rapports HTML (dev/demo)

Chaque `--html` genere un fichier unique dans `reports/` + reconstruit `reports/index.html` :

```bash
uv run python factcheck.py --mode fast --html -- "La tour Eiffel mesure 330 metres"
# → reports/20260329-011450_la-tour-eiffel-mesure-330-metres.html
# → reports/index.html (liste tous les rapports, clic pour ouvrir)
```

Contient : verdict, evidence, timeline pipeline (Linkup/Agents/Synthese), barres parallelisme agents.

> Le HTML est un outil de dev/demo. En prod on recup l'input et on renvoie le JSON.

## Test complet

Lance toutes les combinaisons sur 4 claims de test (factuelle vraie, factuelle fausse, idiom, opinion) :

```bash
uv run python factcheck.py --test-all --log-level info
```

Couvre :
- Pre-screening sur les 4 claims
- Mode fast (generic) sur les claims factuelles
- Mode thorough (generic) sur les claims factuelles
- Mode thorough (specialized) sur les claims factuelles
- Tableau recapitulatif + rapports HTML + summary JSON

## Logs parallelisme

```bash
# Timing de chaque agent
uv run python factcheck.py --log-level info "..."

# Tout le detail (modeles, roles, chars)
uv run python factcheck.py --log-level debug --mode thorough --specialized "..."
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
uv run python factcheck.py --mode fast -- "La tour Eiffel mesure 330 metres"

# Demo complete avec roles specialises + HTML + logs
uv run python factcheck.py --mode thorough --specialized --html --log-level info -- "La France a 67 millions d'habitants"

# Comparer les modeles (meme prompt, pas de specialisation)
uv run python factcheck.py --mode thorough --html --log-level info -- "Le Bitcoin a ete cree en 2009"

# Custom : 2 agents specifiques + export JSON
uv run python factcheck.py --agents gemini,openai --output result.json -- "Le Mont Blanc culmine a 4808 metres"

# Bypass le pre-screening (forcer le pipeline sur une expression)
uv run python factcheck.py --no-screen --mode fast -- "Le verre est a moitie plein"

# Test complet de toutes les combinaisons
uv run python factcheck.py --test-all --log-level info
```
