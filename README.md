# Fact-Check Engine

Multi-agent fact-checking: N LLMs analysent en parallele les memes sources web, puis un thinking model synthetise un verdict structure.

```
Query → Linkup search → N agents LLM (parallele) → Thinking model → Verdict JSON
```

## Quick Start

```bash
# Install
uv sync

# Copier et remplir les cles API
cp .env.example .env

# Lancer un fact-check (mode default : 3 agents)
uv run python factcheck.py "Emmanuel Macron a ete elu president en 2017"
```

## Modes

3 presets selon le use-case :

| Mode | Agents | Recherche | Usage |
|------|--------|-----------|-------|
| `--mode fast` | 2 (gemini-flash, o4-mini) | standard | **Prod** — rapide, pas cher |
| `--mode default` | 3 (gemini-flash, mistral, o4-mini) | deep | Equilibre |
| `--mode thorough` | 5 (claude, openai, gemini, mistral, grok) | deep | **Demo** — complet |

```bash
# Prod-like : rapide, 2 agents legers
uv run python factcheck.py --mode fast "..."

# Demo : 5 agents frontier, analyse complete
uv run python factcheck.py --mode thorough "..."
```

## Options

```
--mode fast|default|thorough   Preset (agents + profondeur recherche)
--agents claude,openai,...     Override manuel des agents (ecrase le preset)
--synthesis-model MODEL        Modele de synthese (default: google/gemini-2.5-flash)
--output result.json           Export JSON
--html [report.html]           Genere un rapport HTML interactif (dev/demo)
--verbose                      Inclut les donnees brutes dans le JSON
--log-level debug|info|warning Log level — debug/info pour voir le parallelisme
--no-pretty                    JSON compact
```

## Rapport HTML (dev/demo)

Genere une page HTML standalone avec le verdict, l'evidence, et une visu du parallelisme des agents :

```bash
# Rapport HTML par defaut (report.html)
uv run python factcheck.py --mode thorough --html "..."

# Chemin custom
uv run python factcheck.py --html analysis.html "..."
```

> **Note :** Le HTML est un outil de dev/demo. En prod, on recup l'input, on envoie l'output JSON — pas de HTML.

## Logs parallelisme

Pour voir le detail de l'execution parallele (quel agent demarre, finit, combien de temps) :

```bash
# Logs info : timing de chaque agent
uv run python factcheck.py --log-level info "..."

# Logs debug : tout le detail
uv run python factcheck.py --log-level debug --mode thorough "..."
```

## Agents disponibles

### Frontier
- `claude` — Claude Sonnet 4.6
- `openai` — GPT-5.4
- `gemini` — Gemini 2.5 Pro
- `grok` — Grok 4

### Bon rapport qualite/prix
- `mistral` — Mistral Small 3.2
- `deepseek` — DeepSeek V3.2
- `qwen` — Qwen3 Max

### Rapides/pas cher
- `gemini-flash` — Gemini 2.5 Flash
- `gpt-mini` — GPT-5 Mini
- `o4-mini` — o4-mini

## Variables d'environnement

```
OPENROUTER_API_KEY=     # Requis — cle OpenRouter
LINKUP_API_KEY=         # Requis — cle Linkup search
FACTCHECK_MODE=         # Optionnel — fast/default/thorough
FACTCHECK_AGENTS=       # Optionnel — override agents
FACTCHECK_SYNTHESIS_MODEL=  # Optionnel — modele de synthese
```

## Exemples

```bash
# Test rapide en prod
uv run python factcheck.py --mode fast "La tour Eiffel mesure 330 metres"

# Demo complete avec HTML + logs
uv run python factcheck.py --mode thorough --html --log-level info --verbose "La France a 67 millions d'habitants"

# Custom : 2 agents specifiques + export JSON
uv run python factcheck.py --agents gemini,openai --output result.json "..."
```
