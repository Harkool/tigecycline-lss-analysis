# Machine-Learning Prediction of Tigecycline Exposure with Limited Sampling Strategies

**Source code and reproducible analysis pipeline for:**

Dai A, Liu H, Yang B, et al. **Application of Machine Learning Models to Predict Tigecycline Exposure in Critically Ill Patients Using Different Limited Sampling Strategies.** *Drug Design, Development and Therapy.* 2026;20:630414.

- **DOI:** https://doi.org/10.2147/DDDT.S630414
- **PubMed:** https://pubmed.ncbi.nlm.nih.gov/42894723/
- **Full text:** https://pmc.ncbi.nlm.nih.gov/articles/PMC13676858/
- **Publisher:** https://www.dovepress.com/application-of-machine-learning-models-to-predict-tigecycline-exposure-peer-reviewed-fulltext-article-DDDT

## Overview

This repository provides the Python implementation and supporting materials for machine-learning prediction of tigecycline exposure in critically ill patients using different limited sampling strategies.

The codebase covers data preparation, model development, performance evaluation, and additional reproducibility and robustness analyses. It includes:

- **Repeated nested cross-validation:** A unified evaluation framework for random forest (RF), XGBoost (XGB), multilayer perceptron (MLP), and long short-term memory (LSTM) models, with imputation, scaling, hyperparameter selection, and early stopping performed within the appropriate training folds.
- **Reference-model comparisons:** Simple predictive benchmarks and an alternative reference AUC calculated without using the plasma concentrations included as model inputs.
- **Neural-network comparisons:** Feed-forward networks matched to the LSTM architectures by trainable-parameter count, evaluated across multiple random seeds.
- **Prediction performance:** Conventional prediction metrics, agreement measures, and target-attainment assessments.
- **Statistical consistency checks:** Evaluation of reported R² and RMSE values using the relationship `RMSE² = (1 − R²) × Var(y)`, under the conventional sum-of-squares definition of R².
- **Missing-data analysis:** Documentation of missing observations, imputation procedures, outlier handling, and comparison of imputation averaging with uncertainty pooling using Rubin's rules.
- **Additional diagnostics:** Sample-size assessment, LASSO selection stability, time-order permutation tests, and sensitivity to model capacity and training duration.

Precomputed outputs are included in [`results/`](results/) and [`figures/`](figures/), comprising **Tables R1–R9 and Figures R1–R4**. These identifiers correspond to the repository's output files.

The pipeline can be rerun to regenerate the outputs using the appropriate input data. A fully synthetic dataset is provided for testing the complete workflow without accessing clinical records.

## Code and Data Availability

### Source code

The repository contains the analysis scripts for data preparation, per-time-point LSTM modeling, capacity-matched feed-forward comparisons, repeated cross-validation, performance evaluation, and generation of the included tables and figures.

All analysis scripts are available in [`src/`](src/).

### Clinical data

The patient-level dataset contains sensitive clinical information and is subject to institutional ethics approval and data-governance requirements that do not permit unrestricted public redistribution.

To facilitate reproducibility and code inspection, the repository includes:

- A [data dictionary](data/data_dictionary.md) describing the input variables and their definitions.
- A [synthetic patient dataset](data/synthetic_patients.csv) with the same structure as the clinical data.
- An executable pipeline that can be tested end-to-end using synthetic data.

**Note:** The synthetic dataset is provided for demonstration and computational testing. Results obtained from synthetic data are not equivalent to those obtained from the original clinical dataset.

### Access to patient-level data

De-identified patient-level data are available upon reasonable request, subject to review by the institutional ethics committee and execution of an appropriate data use agreement.

Data requests should be directed to the corresponding author:

**Zhenwei Yu**  
Email: [yzw_srrsh@zju.edu.cn](mailto:yzw_srrsh@zju.edu.cn)

## Installation

**Requirements:**

- Python ≥ 3.9
- pip
- Bash (Git Bash or WSL for Windows)

Install the dependencies:

```bash
pip install -r requirements.txt
```

See [`requirements.txt`](requirements.txt) for the package dependencies.

## Quick Start

### 1. Generate the synthetic dataset

```bash
python src/00_make_synthetic_data.py
```

This creates or regenerates `data/synthetic_patients.csv`.

### 2. Prepare the input dataset

```bash
cp data/synthetic_patients.csv data/patients.csv
```

The file `data/patients.csv` is excluded from version control through `.gitignore` to prevent accidental inclusion of patient-level data.

### 3. Run the complete pipeline

```bash
REPEATS=10 ./run_all.sh
```

The example configuration takes approximately 16 minutes on two CPU cores. Actual runtime may vary depending on hardware and software configuration.

For the full 20-repeat configuration:

```bash
REPEATS=20 ./run_all.sh
```

The pipeline generates analysis tables in `results/` and figures in `figures/`.

### 4. Run individual analyses

Individual scripts can also be executed independently.

For example, to run repeated nested cross-validation:

```bash
python src/01_run_resampling.py \
    --data data/synthetic_patients.csv \
    --repeats 10 \
    --folds 5 \
    --jobs 2
```

To perform the R²/RMSE consistency audit without patient-level data:

```bash
python src/02b_audit_published_tables.py
```

### 5. Regenerate the results

To verify end-to-end execution, remove the precomputed outputs from `results/` and `figures/`, then rerun the pipeline.

The synthetic dataset allows regeneration of demonstration outputs. Reproduction of numerical results based on the original clinical cohort requires access to the corresponding patient-level dataset.

## Analysis Scripts

| Script | Purpose | Outputs |
|---|---|---|
| [`00_make_synthetic_data.py`](src/00_make_synthetic_data.py) | Generates the synthetic example dataset | `data/synthetic_patients.csv` |
| [`01_run_resampling.py`](src/01_run_resampling.py) | Repeated nested cross-validation across models and five sampling strategies, including the independent-reference-AUC analysis | Out-of-fold predictions (`.npz`, gitignored) |
| [`02_analyses.py`](src/02_analyses.py) | Aggregates results, evaluates sample-size requirements, and examines LASSO selection stability | Tables R1–R5, R7a/b, R8, R9 |
| [`02b_audit_published_tables.py`](src/02b_audit_published_tables.py) | Checks internal consistency of reported R²/RMSE pairs | Tables R8a/R8b |
| [`03_imputation_pooling.py`](src/03_imputation_pooling.py) | Compares averaging across 20 imputations with pooling based on Rubin's rules | Table R7c |
| [`04_multiseed_single_split.py`](src/04_multiseed_single_split.py) | Evaluates 30 neural-network seeds on the original 78/20 split, time-order permutation, and model-capacity sensitivity | Tables R6a/b/c |
| [`05_make_figures.py`](src/05_make_figures.py) | Generates analysis figures | Figures R1–R4 |
| [`06_published_arch_diagnostic.py`](src/06_published_arch_diagnostic.py) | Evaluates the architectures reported in Supplementary Table S2 across training epochs and batch sizes | `results/published_arch_diagnostic*` |

## Key Implementation Details

### Per-time-point LSTM (`models.SeqLSTM`)

For a sampling strategy containing `k` plasma concentration measurements, each patient is represented by a time-ordered sequence of length `k`.

At each time step, the input consists of seven standardized clinical covariates (repeated across time steps) and the standardized plasma concentration measured at that time point.

The input tensor therefore has the shape:

```text
(n_patients, k_timepoints, 8_features)
```

The final hidden state is passed to a linear output layer for AUC prediction.

For a single-layer LSTM with hidden dimension `H`, the trainable-parameter count is:

```text
4H(8 + H + 2) + H + 1
```

Representative configurations are:

| Configuration | Trainable parameters |
|---|---:|
| 1 layer, 8 hidden units | 585 |
| 3 layers, 256 hidden units | 1,325,313 |

These counts are consistent with Supplementary Table S3 of the published article.

For the single-concentration C0 strategy (`k = 1`), the LSTM processes only one time step. Although its gating operations remain active, it does not exploit recurrence across multiple observed time points and effectively operates as a gated feed-forward model.

### Capacity-matched feed-forward comparator (`models.FFNet`)

The feed-forward comparator receives the same input information in flattened form:

```text
[7 clinical covariates, k plasma concentrations]
```

It uses two ReLU hidden layers, with hidden-layer widths selected to approximately match the trainable-parameter counts of the corresponding LSTM configurations.

Multiple random seeds are evaluated to assess performance variability and model stability.

### Unified model selection and cross-validation

The repeated nested cross-validation implementation uses a common model-selection protocol for RF, XGB, MLP, and LSTM.

Each algorithm is evaluated using four predefined hyperparameter configurations, selected using an internal 85/15 training–validation split.

Neural networks use early stopping. The selected configuration is subsequently refitted on the complete training fold using the selected number of training epochs.

Imputation, scaling, and other data-dependent preprocessing steps are fitted within the training folds to prevent leakage from held-out evaluation data.

### Missing-data imputation

Missing predictor values are handled using multiple imputation by chained equations (MICE), with Bayesian ridge regression as the conditional imputation model.

Twenty imputation draws are generated, and their imputed values are averaged to construct a completed predictor matrix for modeling.

A separate analysis examines the distinction between averaging imputed values and pooling estimates and uncertainty using Rubin's rules.

### Independent-reference-AUC analysis

An alternative reference AUC is calculated from the concentration measurements:

```text
C1, C2, C3, C4, C12
```

This excludes the model-input concentrations:

```text
C0, C0.5, C6
```

C12 is used as a surrogate for the predose concentration under the steady-state assumption.

The analysis requires an observed C12 measurement, which is available for **90 of the 98 patients**.

This additional analysis examines model performance when the reference AUC is constructed without directly reusing the concentrations supplied as predictors.

### Performance evaluation and diagnostics

The repository includes analyses of predictive accuracy, agreement, and target attainment, together with:

- Repeated-resampling performance estimates.
- Comparisons across model architectures and random seeds.
- LASSO feature-selection stability.
- Missing-data and outlier accounting.
- R²/RMSE consistency checks.
- Neural-network training-duration and capacity sensitivity.

## Input Data Schema

The expected input is a CSV file containing one row per patient.

The dataset includes:

- Clinical covariates defined in the [data dictionary](data/data_dictionary.md).
- Plasma concentration variables `C0` through `C12`, representing the specified sampling times.
- Concentrations recorded in mg/L.
- Empty entries for measurements that were not obtained.

The outcome variable `AUC0-24h` is reconstructed by the analysis code and must not be included as an input variable.

See [`data/data_dictionary.md`](data/data_dictionary.md) for the complete variable definitions and input requirements.

## Repository Structure

```text
.
├── data/
│   ├── data_dictionary.md
│   └── synthetic_patients.csv
│
├── figures/
│   └── Figures R1–R4
│
├── results/
│   └── Tables R1–R9 and diagnostic outputs
│
├── src/
│   ├── 00_make_synthetic_data.py
│   ├── 01_run_resampling.py
│   ├── 02_analyses.py
│   ├── 02b_audit_published_tables.py
│   ├── 03_imputation_pooling.py
│   ├── 04_multiseed_single_split.py
│   ├── 05_make_figures.py
│   ├── 06_published_arch_diagnostic.py
│   └── shared modules
│
├── run_all.sh
├── requirements.txt
└── LICENSE
```

## Citation

If you use this repository or its analysis code, please cite the original publication:

Dai A, Liu H, Yang B, et al. Application of Machine Learning Models to Predict Tigecycline Exposure in Critically Ill Patients Using Different Limited Sampling Strategies. *Drug Des Devel Ther.* 2026;20:630414. doi:[10.2147/DDDT.S630414](https://doi.org/10.2147/DDDT.S630414).

## License

This project is licensed under the [MIT License](LICENSE).
