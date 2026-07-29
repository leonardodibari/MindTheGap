# MindTheGap: HOMO–LUMO Gap Prediction

## Overview

This project predicts the molecular HOMO–LUMO energy gap from the lowest-energy conformer supplied in `json_conformer`. Predictive inputs are restricted to that conformer; identifiers, the target, and other supplied quantum-mechanical properties are excluded to prevent leakage.

The workflow compares classical molecular representations with a geometry-aware SchNet model on fixed random and Bemis–Murcko scaffold splits. A validation-selected LightGBM–SchNet ensemble provides the final prediction.

## Repository structure

```text
MindTheGap/
├── datasets/base.csv                 # Supplied dataset; excluded from Git
├── notebooks/
│   ├── 01_preliminary_data_analysis.ipynb
│   ├── 02_classical_baselines.ipynb
│   ├── 03_schnet.ipynb
│   ├── 04_final_analysis.ipynb
│   ├── 05_additional_results_analysis.ipynb
│   └── 06_presentation_figures.ipynb
├── src/                              # Reusable loading, modeling, evaluation, plotting
├── results/                          # Models, predictions, metrics, final conclusions
├── outputs/                          # Regenerable tables, figures, caches, review artifacts
├── environment.yml
└── requirements.txt
```

`results/` and `outputs/` intentionally remain separate. `results/` contains experiment records that define the reported model comparison: fitted checkpoints, fixed splits, aligned predictions, and metrics. `outputs/` contains derived analysis products, figures, temporary caches, and compact reviewer inputs. Merging them would obscure which files are scientific results and which are regenerable presentation or diagnostic products without simplifying execution.

## Installation

The verified environment uses Python 3.11.15, RDKit 2023.09.1, NumPy 1.26.4, scikit-learn 1.9.0, LightGBM 4.7.0, CPU PyTorch 2.13.0, and PyTorch Geometric 2.8.0.post1.

Conda is the reference installation route:

```bash
git clone https://github.com/leonardodibari/MindTheGap.git
cd MindTheGap
conda env create --prefix ./.venv -f environment.yml
conda activate ./.venv
```

Create a folder 'datasets' in the 'MindTheGap' folder and copy the original assignment dataset to `datasets/base.csv`. It is not committed because it exceeds GitHub's per-file size limit. No external download is required.

## Execution

For a fast scientific review, run only notebooks using the '.venv (Python 3.11.15)' kernel from the Python environment:

```text
01 → 04 → 05 → 06
```

This path performs data analysis, then uses generated and committed prediction tables and compact analysis artifacts. It does not load fitted models or retrain LightGBM or SchNet.

Notebook 06 is independent of kernel state and may be run directly after Notebook 01 to regenerate all PowerPoint figures in `outputs/presentation_figures/`.


For the full analysis and model training instead run the scientific workflow in order:

```text
01 → 02 → 03 → 04 → 05 → 06
```



## Notebook guide

| Notebook | Purpose | Main outputs | Training |
| --- | --- | --- | --- |
| 01 | Validate schema, target, molecular composition, geometry, similarity, scaffolds, and leakage controls | `outputs/tables/`, `outputs/figures/eda/` | No |
| 02 | Build molecular features; select and evaluate classical baselines | Feature artifacts, fixed splits, classical models and predictions | LightGBM/Ridge |
| 03 | Train and evaluate SchNet on the exact notebook-02 splits | SchNet checkpoints, predictions, metrics, diagnostics | SchNet |
| 04 | Align both model families, select the ensemble, and perform final evaluation | Final predictions, comparison table, conclusions | No |
| 05 | Analyze residuals, molecular size, novelty, feature families, bootstrap uncertainty, and difficult cases | `outputs/additional_analysis/` | No |
| 06 | Recreate every presentation figure from saved tabular artifacts | `outputs/presentation_figures/` | No |

## Reproducibility

- Random and scaffold protocols each contain 39,995 training, 8,570 validation, and 8,571 test molecules. Their exact membership is stored in `results/split_indices.npz`.
- Model selection uses validation MAE; test data are reserved for final evaluation.
- Predictions carry molecule IDs, dataset indices, targets, and split labels. Alignment, overlap, target equality, and finite-value checks fail loudly.
- Expensive deterministic preprocessing uses stable cache paths guarded by an ordered molecule-ID hash. Missing caches are regenerated and are never required for fast review.
- `outputs/artifacts/classical_analysis_features.npz` and `classical_analysis_morgan.npz` are compact committed inputs for notebooks 04–05.
- All notebooks locate the repository root explicitly and can start from a fresh kernel.
- A single seed (`42`) is used throughout. PyTorch can still show small platform-dependent numerical variation, especially across CPU/GPU implementations.

RDKit 2026.03.4 raises `AtomValenceException` during hydrogen removal for 377 supplied conformers. The pinned RDKit 2023.09.1 environment parses all 116,492 rows. The complete failure audit is stored in `results/rdkit_2026_remove_hs_failures.csv` and can be reproduced with:

```bash
python scripts/audit_rdkit_compatibility.py \
  --dataset datasets/base.csv \
  --output /tmp/rdkit_audit.csv
```

## Results

| Evaluation split | LightGBM MAE | SchNet MAE | Ensemble MAE |
| --- | ---: | ---: | ---: |
| Random test | 0.01426 | 0.01457 | **0.01313** |
| Scaffold test | 0.02182 | 0.01995 | **0.01968** |

The ensemble uses the validation-selected SchNet weight `w = 0.46` on both test sets. SchNet is most useful under scaffold shift, while the ensemble benefits from complementary molecule-level errors and has the lowest held-out MAE in both protocols. Detailed MAE, RMSE, R², timing, parameter-count, bootstrap, subgroup, and per-molecule results remain in `results/` and `outputs/additional_analysis/`.

## Runtime and implementation notes

- Notebook 01 takes roughly 2 minutes on the verification CPU.
- Notebook 02 takes roughly 10–15 minutes from an empty cache.
- Notebook 03 is the expensive stage: roughly 40 minutes CPU-only; CUDA is optional.
- Notebooks 04–06 perform artifact-based analysis and do not train the main models.
- Only rows with observed `gap` are used for supervised training.
- Scaffold evaluation probes chemical novelty; it is not a guarantee for every out-of-domain regime.
