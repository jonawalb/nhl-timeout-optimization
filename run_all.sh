#!/usr/bin/env bash
# Replication pipeline for "Late, Rare, and Suboptimal: A Real-Options Analysis
# of NHL Coaches' Timeout Decisions."
#
# Usage:
#   bash run_all.sh                  # full pipeline including a fresh API scrape
#   bash run_all.sh --skip-scrape    # use the bundled parquet data, skip the API scrape
#
# Total runtime: ~25 minutes including scrape, ~3 minutes without.
# All scripts read from / write to ../data/ relative to the script directory.

set -euo pipefail

cd "$(dirname "$0")"

SKIP_SCRAPE=0
if [[ "${1:-}" == "--skip-scrape" ]]; then
    SKIP_SCRAPE=1
fi

# Choose Python runner. Prefer `uv run python` if uv is on PATH; otherwise
# fall back to `python3` (which must have the deps from requirements.txt).
if [[ -z "${PYTHON:-}" ]]; then
    if command -v uv >/dev/null 2>&1; then
        PYTHON="uv run python"
    else
        PYTHON="python3"
    fi
fi
echo "Python runner: $PYTHON"

echo "============================================================"
echo "STEP 1/7: Scrape NHL play-by-play (api-web.nhle.com, public)"
echo "============================================================"
if [[ $SKIP_SCRAPE -eq 0 ]]; then
    $PYTHON code/scrape_nhl.py
else
    echo "  --skip-scrape: using bundled data/plays_*.parquet."
fi

echo "============================================================"
echo "STEP 2/7: Verify data integrity (4-check harness)"
echo "============================================================"
$PYTHON code/verify_data.py

echo "============================================================"
echo "STEP 3/7: Fit win-probability model (logistic GLM)"
echo "============================================================"
$PYTHON code/wp_model.py

echo "============================================================"
echo "STEP 4/7: Descriptive analysis"
echo "============================================================"
$PYTHON code/descriptive_analysis.py

echo "============================================================"
echo "STEP 5/7: Solve optimal-stopping Bellman recursion"
echo "============================================================"
$PYTHON code/optimal_stopping.py

echo "============================================================"
echo "STEP 6/7: Extended analyses (EU loss, sensitivity, robustness)"
echo "============================================================"
$PYTHON code/extended_analysis.py

echo "============================================================"
echo "STEP 7/7: Build publication figures"
echo "============================================================"
$PYTHON code/build_figures.py

echo
echo "============================================================"
echo "DONE.  All seven steps completed successfully."
echo "  - Manuscript:  nhl_timeouts.tex   (compile with pdflatex+bibtex)"
echo "  - Figures:     figures/"
echo "  - Data:        data/"
echo "============================================================"
