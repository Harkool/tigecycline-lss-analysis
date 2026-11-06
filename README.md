# Analysis code — machine-learning prediction of tigecycline exposure with limited sampling strategies

Code for:

> Dai A, Liu H, Yang B, et al. *Application of Machine Learning Models to Predict Tigecycline
> Exposure in Critically Ill Patients Using Different Limited Sampling Strategies.*
> Drug Des Devel Ther. 2026;20:630414. doi:10.2147/DDDT.S630414

The repository contains the full analysis pipeline for the article, together with the
extended analyses reported in the correspondence: repeated nested cross-validation under
one protocol for all algorithms (imputation, scaling, early stopping and hyper-parameter
selection repeated inside every training fold); simple reference models and a reference
AUC that contains none of the input concentrations; a feed-forward comparator matched to
the LSTM in the number of trainable parameters, evaluated over multiple random seeds;
agreement and target-attainment metrics; a consistency audit of the reported R²/RMSE
pairs against the identity `RMSE² = (1 − R²) × Var(y)`; and the complete missing-data,
imputation and outlier accounting. Precomputed outputs (Tables R1–R9 and Figures R1–R4;
the R-numbering follows the correspondence) are shipped in `results/` and `figures/` —
delete them and run the pipeline again to regenerate everything from scratch.

## Code and data availability

**Code.** This repository contains the full analysis pipeline: data preparation, the
per-time-point LSTM, capacity-matched feed-forward comparators, repeated
cross-validation, and the scripts that generate every table and figure in the article
and its supplementary materials.

**Data.** The patient-level dataset contains sensitive clinical information and its use
is governed by the approval of the institutional ethics committee, which does not permit
public redistribution. A data dictionary describing every variable is provided
(`data/data_dictionary.md`), and a fully synthetic dataset with the same schema
(`data/synthetic_patients.csv`) is included so that the entire pipeline can be executed
and inspected end-to-end without any patient data.

**Access to patient-level data.** De-identified patient-level data remain available upon
reasonable request, subject to review by the institutional ethics committee and a signed
data use agreement. Requests should be addressed to the corresponding author (Zhenwei Yu,
yzw_srrsh@zju.edu.cn).

## Setup

```bash
pip install -r requirements.txt      # Python >= 3.9
```

## Running on the synthetic example

```bash
python src/00_make_synthetic_data.py            # (re)creates data/synthetic_patients.csv
cp data/synthetic_patients.csv data/patients.csv  # data/patients.csv is gitignored
REPEATS=10 ./run_all.sh                          # ~16 min on 2 CPU cores; REPEATS=20 for the final run
```

`run_all.sh` needs bash (Git Bash or WSL on Windows). Individual steps can also be run
directly, e.g.:

```bash
python src/01_run_resampling.py --data data/synthetic_patients.csv --repeats 10 --folds 5 --jobs 2
python src/02b_audit_published_tables.py          # needs no data at all
```

## Scripts

| Script | Purpose | Outputs |
|---|---|---|
| `00_make_synthetic_data.py` | Generates the synthetic example dataset | `data/synthetic_patients.csv` |
| `01_run_resampling.py` | Repeated nested CV: all models × 5 sampling strategies; independent-reference-AUC variant | out-of-fold predictions (npz, gitignored) |
| `02_analyses.py` | Aggregates to Tables R1–R5, R7a/b, R8, R9; Riley sample size; LASSO stability | `results/table_R*.csv/md` |
| `02b_audit_published_tables.py` | Checks published R²/RMSE pairs for internal consistency (no data needed) | Tables R8a/R8b |
| `03_imputation_pooling.py` | Averaging 20 imputations vs pooling by Rubin's rules | Table R7c |
| `04_multiseed_single_split.py` | 30 network seeds on the original 78/20 split; time-order permutation test; capacity curve | Tables R6a/b/c |
| `05_make_figures.py` | Figures R1–R4 | `figures/*.png` |
| `06_published_arch_diagnostic.py` | Trains the architectures listed in Supplementary Table S2 across epoch counts and batch sizes (training-length sensitivity diagnostic) | `results/published_arch_diagnostic*` |

## Key design points

* **Per-time-point LSTM (`models.SeqLSTM`).** For a strategy using k concentrations, each
  patient is a time-ordered sequence of length k. Step j takes [7 standardised covariates
  (repeated at every step), the standardised concentration at the j-th time point] — an
  input tensor of shape (n, k, 8). The last hidden state feeds a linear head. Parameter
  count = `4H(8+H+2)+H+1`: one layer of 8 units → 585; three layers of 256 units →
  1,325,313, matching the article's Supplementary Table S3. For k = 1 (C0 strategy) the
  initial states are zero, the recurrence contributes nothing, and the network is
  functionally a gated feed-forward net.
* **Capacity-matched comparator (`models.FFNet`).** The same inputs flattened to
  [7 covariates, k concentrations]; two ReLU hidden layers whose width is chosen so the
  trainable-parameter count matches the LSTM configuration.
* **Unified tuning budget.** RF, XGB, MLP and LSTM each get 4 preset configurations,
  selected on an inner 85/15 split (networks with early stopping); the winner is refitted
  on the whole training fold at its best epoch. Imputation (MICE, BayesianRidge, 20 draws
  averaged), scaling and all preprocessing are fitted on the training fold only.
* **Independent reference AUC.** Recomputed from C1, C2, C3, C4 and C12 alone (C12
  standing in for the pre-dose value at steady state), so it contains none of the model
  inputs (C0, C0.5, C6); requires a measured C12 (90 of 98 patients).

## Input data schema

See `data/data_dictionary.md` for the full dictionary. In short: one row per patient;
covariate columns as listed there; concentration columns `C0 … C12` in mg/L (empty = not
measured); `AUC0-24h` is recomputed by the code and must not be supplied.

## Repository layout

```
.
├── data/                  data dictionary + synthetic example (no patient data)
├── figures/               Figures R1–R4
├── results/               Tables R1–R9 and audit outputs
├── src/                   analysis code (00–06 + shared modules)
├── run_all.sh             one-command reproduction
└── requirements.txt
```

## License

MIT — see [LICENSE](LICENSE).
