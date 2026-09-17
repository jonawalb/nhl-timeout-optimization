# Late, Rare, and Suboptimal — Submission and Replication Package

This folder contains everything needed to (a) submit the manuscript to JQAS
and (b) reproduce every empirical result from raw data.

---

## Submission package

For Overleaf or ScholarOne, drop the entire folder into a new project. The
referenced files are:

| File | Role |
|------|------|
| `nhl_timeouts.tex` | Manuscript source (20 pages, JQAS format) |
| `refs.bib` | Bibliography, Chicago author-date, 13 entries |
| `nhl_timeouts.pdf` | Compiled paper for reference |
| `figures/fig*.png` | Ten figures referenced in the text |

Compile with **pdfLaTeX + bibtex** (the order: pdflatex, bibtex, pdflatex,
pdflatex). On Overleaf the default workflow handles this automatically.

---

## Replication package

The pipeline reproduces every number, table, and figure in the manuscript
from a single command. The data source is the official NHL public Web API
at `https://api-web.nhle.com/v1`, which requires no authentication and no
API key.

### One-command reproduction

```bash
# Option A: run from a fresh scrape (~25 minutes, ~5 MB of network)
bash run_all.sh

# Option B: skip the API scrape and use the bundled data (~3 minutes)
bash run_all.sh --skip-scrape
```

The bundled parquet files in `data/` come from a scrape on 2026-04-26. They
are byte-for-byte reproducible from the live API on any subsequent date,
since the NHL play-by-play data for completed games does not change.

### Environment

```bash
# With uv (recommended)
uv venv
uv pip install -r requirements.txt

# Or with pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested on Python 3.12 and 3.14 on macOS. Should work on any Python 3.10+.

### Pipeline layout

`run_all.sh` invokes the seven scripts in `code/` in order. Each script reads
from and writes to `data/`. Intermediate outputs are CSV or JSON; final
figures land in `figures/`.

| Step | Script | Reads | Writes | Runtime |
|------|--------|-------|--------|---------|
| 1 | `code/scrape_nhl.py` | NHL public API | `data/plays_*.parquet`, `data/timeouts.parquet` | ~20 min |
| 2 | `code/verify_data.py` | parquet files + live API | `data/verification_report.json` | ~1 min |
| 3 | `code/wp_model.py` | `plays_*.parquet` | `data/wp_model_params.csv` | ~10 sec |
| 4 | `code/descriptive_analysis.py` | parquet files | `data/summary.json`, `data/leverage_grid.csv`, etc. | ~30 sec |
| 5 | `code/optimal_stopping.py` | (none, computes from primitives) | `data/optimal_stopping_results.json`, `data/optimal_stopping_summary.csv` | ~30 sec |
| 6 | `code/extended_analysis.py` | parquet + step 5 | `data/eu_loss.csv`, `data/lambda_sensitivity.csv`, `data/team_statistics.csv`, `data/extended_summary.json` | ~10 sec |
| 7 | `code/build_figures.py` | parquet + step 5 | `figures/fig*.png`, `figures/fig*.pdf` | ~20 sec |

### Data inputs and outputs

`data/` after a successful run:

| File | Contents |
|------|----------|
| `plays_20222023.parquet` | Play-by-play for 2022--2023 regular season (1,312 games, ~412k rows) |
| `plays_20232024.parquet` | Play-by-play for 2023--2024 (1,312 games, ~410k rows) |
| `plays_20242025.parquet` | Play-by-play for 2024--2025 (1,312 games, ~420k rows) |
| `timeouts.parquet` | The 468 timeout events with full game-state context |
| `verification_report.json` | Output of the four-check integrity harness |
| `summary.json` | Top-line descriptive stats |
| `leverage_grid.csv` | WP leverage by (score-diff, time) cell |
| `wp_model_params.csv` | Logistic GLM coefficients |
| `wp_at_call.csv` | Calling-team WP at each of the 468 calls |
| `timeout_wp_effects.csv` | Pre/post 5-min ΔWP per timeout (descriptive) |
| `optimal_stopping_results.json` | Full Bellman solution across $\mu \in \{0.005, 0.01, 0.02, 0.03\}$ |
| `optimal_stopping_summary.csv` | Headline comparison of optimal vs empirical policy |
| `eu_loss.csv` | EU-loss-in-WP-points table |
| `lambda_sensitivity.csv` | Robustness across goal-arrival rate $\lambda$ |
| `state_dependent_boost.json` | Pulled-goalie state-dependent-$\mu$ robustness output |
| `team_statistics.csv` | Per-team usage rate (32-team table) |
| `extended_summary.json` | Combined extended-analysis summary |
| `timeout_summary.csv` | One-line headline statistics |

### Verifying provenance

Step 2 (`verify_data.py`) runs four independent integrity checks against the
live NHL API:

1. **Game-count match.** Each season should contain 1,312 games (the 32-team,
   82-game-each, half-home-half-away schedule). Verifies parser caught all
   games on the schedule.
2. **One-timeout-per-team-game.** No team-game should contain more than one
   timeout call (the rule constraint).
3. **Score-line round-trip.** For 30 random games, the parser-derived final
   score must match the official final score from the gamecenter `landing`
   endpoint, accounting for the NHL convention of adding +1 to the
   shootout-winning team's official score.
4. **Timeout re-fetch round-trip.** For 20 random games containing a timeout,
   re-fetch the play-by-play from the live API and confirm the timeout count
   reproduces.

All four pass. Output:

```
=========================
OVERALL: PASS ✓
=========================
```

### Public-API URL examples

To inspect any single game's data without running the pipeline:

```bash
# Schedule for one week
curl -s "https://api-web.nhle.com/v1/schedule/2024-10-08" | jq

# Play-by-play for one specific game (CAR @ BUF, 2025-01-15)
curl -s "https://api-web.nhle.com/v1/gamecenter/2024020705/play-by-play" | jq

# Final box score for the same game
curl -s "https://api-web.nhle.com/v1/gamecenter/2024020705/landing" | jq
```

The same URLs back NHL.com's gamecenter pages.

### Random seeds

All Monte Carlo draws use seeded numpy generators (`np.random.default_rng(7)`
or `default_rng(42)`). Re-running `run_all.sh` reproduces every numerical
result bit-for-bit, including figure layouts.

---

## JQAS conformance checklist

- [x] 20 pages (within 20--30 range; ~26 lines/page via `\onehalfspacing` + Times 11pt)
- [x] 8.5 × 11 inch paper, 1-inch margins, single-column
- [x] De-identified (no author block, no acknowledgements)
- [x] Abstract ~200 words; 6 keywords disjoint from title
- [x] Chicago author-date citations via `natbib` + `chicago.bst`
- [x] Tables typeset (not images), Arabic numerals, descriptive captions
- [x] Figures separate, sans-serif lettering, color, 300 dpi
- [x] Bar charts use patterning rather than greyscale
- [x] Equations numbered, italic Roman variables, blank lines around displayed math
- [x] Heading hierarchy clear and consistent
- [x] No footnotes
- [x] All figures and tables referenced by number in text via `\ref{}`

## Files at a glance

```
New/
├── README.md                       (this file)
├── nhl_timeouts.tex                manuscript source
├── nhl_timeouts.pdf                compiled manuscript
├── refs.bib                        bibliography (Chicago)
├── requirements.txt                pip dependencies
├── pyproject.toml                  uv/PEP-621 dependencies
├── run_all.sh                      one-command replication
├── code/                           seven Python scripts
│   ├── scrape_nhl.py
│   ├── verify_data.py
│   ├── wp_model.py
│   ├── descriptive_analysis.py
│   ├── optimal_stopping.py
│   ├── extended_analysis.py
│   └── build_figures.py
├── data/                           parquet + CSV/JSON outputs (see table above)
└── figures/                        ten PNG/PDF figure pairs
```

## Authorship of the data

Game data are owned by the National Hockey League. The play-by-play feed is
publicly available without authentication at the URL referenced above. This
replication package distributes only the parsed parquet files needed to
reproduce the analysis; users wishing to scrape the live data themselves
should consult the NHL's terms of use.
