#!/usr/bin/env bash
# Run the full analysis. Put the data file at data/patients.csv first (see README for column schema).
set -e
cd "$(dirname "$0")"
REPEATS=${REPEATS:-10}      # use 20 for the final run
python src/01_run_resampling.py --repeats $REPEATS --folds 5 --jobs 2
python src/02_analyses.py
python src/02b_audit_published_tables.py
python src/03_imputation_pooling.py
python src/04_multiseed_single_split.py --seeds 30 --jobs 2
python src/05_make_figures.py
echo "done: see results/*.md and figures/*.png"
