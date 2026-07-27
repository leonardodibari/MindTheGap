# MindTheGap: HOMO–LUMO Gap Prediction

## Project Overview

This repository addresses molecular-property regression for the HOMO–LUMO energy gap (`gap`) from a molecule's lowest-energy conformer. Standard model inputs are derived exclusively from `json_conformer`; identifiers, `gap`, and the other supplied quantum-mechanical properties are not used as predictive features.

The solution progresses from data validation and exploratory analysis to classical molecular representations, a geometry-aware SchNet model, and a validation-selected ensemble. Both fixed random and Bemis–Murcko scaffold splits are evaluated. The emphasis is on reproducible comparisons, explicit leakage controls, and analysis of when the models agree or fail rather than extensive hyperparameter search.

## Repository Structure

The tree below was generated from the current repository (local environment and Python cache directories omitted):

```text
MindTheGap/
├── assignment.md
├── datasets/
│   └── base.csv
├── environment.yml
├── notebooks/
│   ├── 01_preliminary_data_analysis.ipynb
│   ├── 02_classical_baselines.ipynb
│   ├── 03_schnet.ipynb
│   ├── 04_final_analysis.ipynb
│   └── 05_additional_results_analysis.ipynb
├── outputs/
│   ├── additional_analysis/     # Supplementary tables and figures
│   ├── cache/                   # Dataset-keyed descriptors, fingerprints, and graphs
│   ├── figures/eda/             # Exploratory-analysis figures
│   └── tables/                  # EDA and molecular-metadata tables
├── requirements.txt
├── results/
│   ├── figures/                 # Classical and SchNet diagnostics
│   ├── final_figures/           # Presentation-ready comparison figures
│   ├── models/                  # Fitted classical models and SchNet checkpoints
│   ├── classical_baselines.csv
│   ├── classical_predictions_random.csv
│   ├── classical_predictions_scaffold.csv
│   ├── final_conclusions.md
│   ├── final_model_comparison.csv
│   ├── final_predictions_random_test.csv
│   ├── final_predictions_scaffold_test.csv
│   ├── final_predictions_validation.csv
│   ├── schnet_metrics.csv
│   ├── schnet_predictions_random.csv
│   ├── schnet_predictions_scaffold.csv
│   ├── schnet_vs_lightgbm_molecule_analysis.csv
│   ├── split_indices.npz
│   └── worst_predicted_molecules.csv
└── src/
    ├── analysis.py
    ├── classical.py
    ├── config.py
    ├── data.py
    ├── evaluation.py
    ├── features.py
    ├── loaders.py
    ├── plotting.py
    ├── schnet.py
    └── utils.py
```

`notebooks/` contains the scientific narrative and experiment definitions. `src/` contains reusable loading, feature, modeling, evaluation, plotting, and reproducibility utilities. `outputs/` holds deterministic preprocessing products and supplementary analyses; `results/` holds fitted models, aligned predictions, metrics, and final figures.

## Methodology

1. **Exploratory data analysis.** Validate the schema, labeled/unlabeled rows, missingness, duplicates, target scale and extremes, molecular composition, geometry, fingerprint sparsity, similarity, and scaffold diversity. Auxiliary target columns are explicitly identified as leakage risks.
2. **Feature engineering.** Parse molecular JSON with RDKit and derive size/composition counts, bond types, simple 3D geometry, RDKit 2D descriptors, radius-2 2048-bit Morgan fingerprints, and Bemis–Murcko scaffolds. Expensive deterministic representations are cached with a dataset-dependent key.
3. **Classical machine learning.** Compare a mean predictor, Ridge models, and LightGBM models using size/composition, RDKit descriptors, Morgan fingerprints, and a combined 2D/3D representation. Preprocessing is contained in scikit-learn pipelines, with limited selection on validation MAE.
4. **SchNet.** Train a PyTorch Geometric SchNet from atomic numbers, Cartesian coordinates, and molecule batch membership. Training uses MAE loss, deterministic seeding, validation monitoring, early stopping, and saved best checkpoints.
5. **Ensemble.** Combine the selected LightGBM and SchNet predictions as `w × SchNet + (1 − w) × LightGBM`. The weight is chosen on validation predictions only and then fixed for test evaluation.
6. **Final evaluation.** Report MAE as the primary metric, with RMSE and R² as secondary metrics, on a fixed 70/15/15 random split and an approximately matched scaffold split.
7. **Additional analyses.** Examine residuals, molecular-size and chemical-novelty strata, feature-family ablations, paired bootstrap comparisons, error complementarity, and difficult molecules.

## Repository Workflow

The intended execution order is:

```text
01 → 02 → 03 → 04 → 05
```

Notebooks 01–03 generate the EDA, feature, split, model, metric, and prediction artifacts. Notebooks 04 and 05 consume previously generated artifacts: they do not construct or retrain the main LightGBM or SchNet models. Required-artifact checks report the missing path, the notebook that creates it, and the required execution order.

## Notebook Description

| Notebook | Purpose | Required inputs | Principal outputs | Model training |
| --- | --- | --- | --- | --- |
| `01_preliminary_data_analysis.ipynb` | Data quality, target, molecular diversity, leakage, and pre-modeling validation | `datasets/base.csv` | `outputs/tables/`, `outputs/figures/eda/` | No |
| `02_classical_baselines.ipynb` | Construct conformer-derived features; select and evaluate classical baselines | Labeled rows in `datasets/base.csv` | `outputs/cache/classical_*`, `results/split_indices.npz`, classical model files, predictions, metrics, and diagnostics | Yes |
| `03_schnet.ipynb` | Select, train, and evaluate SchNet; compare against saved classical predictions | Dataset, split indices, classical predictions/metrics, graph cache if available | `outputs/cache/schnet_graphs_*`, SchNet checkpoints, predictions, metrics, and diagnostics | Yes—SchNet only |
| `04_final_analysis.ipynb` | Align both models, analyze errors, select the ensemble weight, and produce final comparisons | Dataset, split/feature artifacts, classical and SchNet predictions/metrics | Final prediction tables, `results/final_model_comparison.csv`, `results/final_figures/`, and `results/final_conclusions.md` | No |
| `05_additional_results_analysis.ipynb` | Technical appendix for robustness, subgroup, bootstrap, and worst-case analysis | Final predictions/metrics and deterministic feature caches | Tables and figures in `outputs/additional_analysis/` | No; the saved fixed-configuration ablation is loaded |

## Installation

The repository was developed and tested with Python 3.11. Clone the private repository and enter its root directory first:

```bash
git clone <repository-url>
cd MindTheGap
```

### Recommended: Conda

Create the project-local environment from the supplied `environment.yml`:

```bash
conda env create --prefix ./.venv -f environment.yml
conda activate ./.venv
```

This installs Python 3.11, RDKit, the scientific Python stack, PyTorch, and PyTorch Geometric.

### Alternative: pip

The same declared dependencies are available in `requirements.txt` for reviewers who prefer a Python 3.11 virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

CUDA is optional. SchNet selects a CUDA device when available and automatically falls back to CPU otherwise.

## Quick Start

1. Clone the repository with `git clone <repository-url>` and run `cd MindTheGap`.
2. Create and activate the recommended Conda environment using the commands above.
3. Place the dataset supplied with the assignment at `datasets/base.csv`.
4. From the repository root, launch Jupyter:

   ```bash
   jupyter notebook
   ```

5. Execute the notebooks in order:

   ```text
   01 → 02 → 03 → 04 → 05
   ```

For a fast review using the submitted models and predictions, execute only:

```text
04 → 05
```

This fast path regenerates the final and supplementary analyses without retraining LightGBM or SchNet.

## System Requirements

- Python 3.11.
- A CUDA-capable GPU is optional but recommended for notebook 03; all notebooks support CPU execution.
- Allow roughly 3 GB for the environment, supplied dataset, current artifacts, and caches. Additional temporary working space is advisable when regenerating the full workflow.

## Dataset

Place the original dataset supplied with the assignment at `datasets/base.csv`. The repository expects exactly this file; no external download is required or assumed.

The dataset is intentionally excluded from Git because it exceeds GitHub's per-file size limit. It must therefore be copied into `datasets/` after cloning and before running any notebook.

## Running the Project

### Full Reproduction

After completing the Quick Start setup, run all five notebooks in order. Notebook 01 creates the exploratory outputs, notebook 02 trains the classical models, and notebook 03 trains SchNet. Notebooks 04 and 05 then load those saved artifacts to produce the final and supplementary analyses.

Feature and graph caches are regenerated when absent. This is the appropriate workflow for reproducing the complete submission from the supplied raw dataset.

### Fast Review

With the submitted artifacts present, run:

```text
04 → 05
```

This path uses `datasets/base.csv`, `results/split_indices.npz`, the classical feature cache, both models' metric files, and the four random/scaffold prediction files in `results/`. Missing inputs produce an actionable error naming the notebook that generates them.

## Runtime Expectations

- **Notebook 01:** relatively quick exploratory analysis and deterministic preprocessing.
- **Notebook 02:** typically a few minutes, depending on CPU resources and whether feature caches already exist.
- **Notebook 03:** the most computationally expensive stage; a CUDA GPU is recommended. CPU execution is supported but substantially slower.
- **Notebooks 04–05:** quick artifact-based analysis with no main-model retraining.

## Results

Model selection uses validation MAE; test sets are reserved for final evaluation. The random and scaffold protocols each contain 39,995 training, 8,570 validation, and 8,571 test molecules. Predictions are saved with molecule identifiers, dataset indices, targets, split labels, and errors, and are validated for alignment and finite values.

The following values are read from the current `results/final_model_comparison.csv`:

| Evaluation split | LightGBM MAE | SchNet MAE | Ensemble MAE |
| --- | ---: | ---: | ---: |
| Random test | 0.01426 | 0.01457 | **0.01313** |
| Scaffold test | 0.02182 | 0.01995 | **0.01968** |

The ensemble uses the validation-selected SchNet weight `w = 0.46`. The same fixed weight is evaluated on both test sets. Full MAE, RMSE, R², timing, parameter-count, and relative-difference fields remain available in `results/final_model_comparison.csv`.

## Saved Artifacts

- `results/split_indices.npz`: exact random and scaffold train/validation/test membership.
- `results/models/classical_best_random.joblib` and `classical_best_scaffold.joblib`: selected fitted classical pipelines.
- `results/models/schnet_best_random.pt` and `schnet_best_scaffold.pt`: selected SchNet checkpoints.
- `results/classical_predictions_random.csv` and `classical_predictions_scaffold.csv`: aligned classical validation/test predictions.
- `results/schnet_predictions_random.csv` and `schnet_predictions_scaffold.csv`: aligned SchNet validation/test predictions.
- `results/final_predictions_validation.csv` and `final_predictions_*_test.csv`: aligned model/ensemble predictions used by the final analyses.
- `results/classical_baselines.csv`, `schnet_metrics.csv`, and `final_model_comparison.csv`: experiment and consolidated metrics.
- `outputs/cache/`: deterministic dense features, Morgan fingerprints, and serialized molecular graphs for the current dataset key.
- `results/figures/`, `results/final_figures/`, and `outputs/additional_analysis/`: diagnostic, presentation, and technical-appendix outputs.

## Design Decisions

- Shared implementation is separated by concern in `src/`, while model configurations and scientific interpretation remain visible in notebooks.
- A single seed (`42`) and saved split indices support repeatable comparisons across model families.
- Dataset-keyed caches avoid repeating deterministic molecular preprocessing without replacing the raw dataset as the source of truth.
- Prediction artifacts are aligned by `mol_id` and `dataset_index`; split overlap, target mismatch, duplicate identifiers, and non-finite predictions are checked explicitly.
- Training notebooks and analysis notebooks have separate responsibilities, enabling a quick artifact-based review without hidden retraining.

## Notes

- Only rows with observed `gap` are used for supervised training; unlabeled rows are analyzed but are not assumed to be a hidden test set.
- All predictive inputs are derived from `json_conformer`. Other molecular-property columns are excluded to avoid target leakage.
- Scaffold evaluation is a robustness experiment for chemical novelty, not a guarantee of performance on every out-of-domain chemistry regime.
- PyTorch/CUDA operations can exhibit small platform-dependent numerical variation despite deterministic seeding.
