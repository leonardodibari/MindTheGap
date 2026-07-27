# Aqemia Machine Learning take-home exercise <!-- omit in toc -->

As an applied scientist, you'll have the opportunity to work on machine learning research
projects. This take-home introduces some of the data formats and topics we work with at
Aqemia.

## Local setup

Create and activate the reproducible project-local Conda environment:

```bash
conda env create --prefix ./.venv -f environment.yml
conda activate ./.venv
```

The supplied dataset is intentionally not committed because `datasets/base.csv` exceeds
GitHub's 100 MB per-file limit. Keep the original file at
`datasets/base.csv` after cloning the repository.

## Reproducing the analysis

### Full reproduction from scratch

1. Create and activate the environment shown above.
2. Place the supplied dataset at `datasets/base.csv`.
3. Start Jupyter from the repository root.
4. Run the notebooks in this order:

```text
01 → 02 → 03 → 04 → 05
```

- `01_preliminary_data_analysis.ipynb` performs data validation and exploratory analysis.
- `02_classical_baselines.ipynb` constructs conformer-derived features, trains the
  classical baselines, and saves split, model, metric, validation-prediction, and
  test-prediction artifacts.
- `03_schnet.ipynb` trains and evaluates SchNet and saves checkpoints, histories,
  metrics, and validation/test predictions. A CUDA GPU is optional; CPU execution is
  supported but substantially slower.
- `04_final_analysis.ipynb` consolidates the saved model outputs and produces the final
  comparison and ensemble analysis.
- `05_additional_results_analysis.ipynb` is a technical appendix built from the saved
  predictions and deterministic feature metadata.

Notebooks can be launched from either the repository root or `notebooks/`. Shared
implementation lives in `src/`, while methodological choices and scientific
interpretations remain visible in the notebooks.

Deterministic molecular feature and graph caches accelerate repeated execution, but they
are not the only source of truth: notebooks 02 and 03 regenerate them when absent using a
dataset-dependent cache key.

### Fast review of submitted results

When the submitted artifacts are present, run only:

```text
04 → 05
```

This path does not construct, fit, or tune LightGBM or SchNet. It requires:

- `datasets/base.csv`;
- `results/split_indices.npz`;
- `outputs/cache/classical_dense_*.npz`;
- `results/classical_baselines.csv`;
- `results/schnet_metrics.csv`;
- `results/classical_predictions_random.csv` and `classical_predictions_scaffold.csv`;
- `results/schnet_predictions_random.csv` and `schnet_predictions_scaffold.csv`.

Missing artifacts produce an actionable message naming the notebook that generates them
and the required execution order.

## Repository structure

```text
datasets/                       local input data (not committed)
notebooks/                      scientific workflow and interpretation
src/config.py                   paths and shared constants
src/data.py                     loading and ID/split validation
src/features.py                 molecular feature definitions
src/classical.py                classical model pipelines
src/schnet.py                   SchNet training and inference
src/evaluation.py               metrics and bootstrap comparisons
src/analysis.py                 project-specific post-hoc analyses
src/plotting.py                 shared plotting conventions
outputs/cache/                  deterministic feature cache
outputs/tables/                 EDA and validation tables
outputs/additional_analysis/    technical-appendix artifacts
results/models/                 trained checkpoints
results/                        metrics and aligned prediction artifacts
results/figures/                classical and SchNet figures
results/final_figures/          presentation-ready final figures
```

The existing `outputs/` and `results/` paths are retained as the canonical artifact
contract to preserve backward compatibility:

| Producer | Main saved artifacts |
| --- | --- |
| Notebook 01 | `outputs/tables/molecular_metadata.csv`, other EDA tables, `outputs/figures/eda/` |
| Notebook 02 | `results/split_indices.npz`, `outputs/cache/classical_*`, `results/models/classical_*`, `results/classical_predictions_*.csv`, `results/classical_baselines.csv` |
| Notebook 03 | `outputs/cache/schnet_graphs_*`, `results/models/schnet_*`, `results/schnet_predictions_*.csv`, `results/schnet_training_history_*.csv`, `results/schnet_metrics.csv` |
| Notebook 04 | `results/final_model_comparison.csv`, `results/final_predictions_validation.csv`, `results/final_predictions_*_test.csv`, `results/final_figures/` |
| Notebook 05 | `outputs/additional_analysis/` |

Saved prediction tables include molecule IDs and are validated against source targets;
split helpers reject duplicate or overlapping memberships rather than relying silently on
row order.

- [Overview](#overview)
- [Problem](#problem)
- [Deliverables](#deliverables)
- [Evaluation](#evaluation)
- [Time](#time)
- [Datasets](#datasets)
  - [Chemistry primer](#chemistry-primer)
  - [Dataset](#dataset)
    - [Features](#features)
    - [Tasks](#tasks)
  - [Data loaders](#data-loaders)

## Overview

- The task and dataset are chemistry-related and showcase some of the modalities we work
  with at Aqemia.
- The goal is _not_ to assess your chemistry knowledge. No prior knowledge of chemistry is
  required. See the provided [loaders](#data-loaders) to read the chemical data.
- The goal of the model is to predict the `gap` column in the `datasets/base.csv` (see
  [datasets](#datasets) below for details).
- You may use any machine learning technique. We do not expect specific techniques used in
  the chemistry ML field (though if you know them, feel free to use them!).
- Taking code, models, etc... from the internet is fine. Cite your sources (a link will
  suffice).

## Problem

Aqemia focuses on small molecules, one of the many possible types of drugs (among
biologics, vaccines, etc...).

When designing molecules for therapeutic purposes, many properties of the molecules are of
interest, such as solubility, toxicity, distribution in the body, time before the drug is
purged from the body, etc... We refer to these as molecular properties. A key element of
drug design is being able to predict molecular properties of interest, as experiments are
costly and time-consuming.

The goal of this exercise is to predict a molecular property, the HOMO-LUMO gap. Note that
the HOMO-LUMO gap is not itself a molecular property useful for drug discovery
(it comes from material science).

## Deliverables

Your deliverables should include:

- **Dataset analysis**: some analysis of the data.
- **Methods used**: explanation of the methods you used and why, with (quick) references.
- **Experiments**: the experiments you performed.
- **Results**: analysis of your results.

You must provide the code you used to generate results, and we should be able to reproduce your
figures with minimal effort. We should also be able to test your models on our held-out test set
with minimal modifications.

The email sent to you contains the expected format to present your results.

If no specific instructions were included, the format is up to you, in which case we do not expect
a stand-alone report or slides. A notebook or plots to support the discussion, which can be followed
when we read your submission in advance, is enough.

Following your submission, you will have an open discussion to present your work and
conclusions.

You can use any **deep learning framework** of your choosing on python 3.11+ (e.g. Pytorch,
Tensoflow, Jax, ...).

## Evaluation

1. You will be evaluated on your methodology, not on the performance of your model. You
   are tasked to explore different approaches for tackling the problem, not to build a
   production model.
2. You will be evaluated on your ability to communicate to a technical audience, both in
   writing and during the follow-up discussion.
3. Code quality will be taken into account.

## Time

We estimate the exercise to take about 5-6 hrs.

We do not expect an in-depth literature review or extensive experimentation worthy of a
full-fledged project, this remains an exercise to give you the opportunity to show-case
your applied research skills.

## Datasets

### Chemistry primer

A _molecule_ is composed of atoms, and links (bonds) between those atoms. Atoms can be of
different types (a categorical variable). Bonds can also be of different types (also
another categorical variables).

A _conformer_ is a 3D position of the atoms of a molecule, with respect to each other. A
given molecule may adopt many conformers. Some conformers are favorable (i.e. have low
energy), others are unfavorable (i.e. have high energy). The lowest-energy conformer plays
an important role in determining key properties of the molecule.

### Dataset

#### Features

- `mol_id`: a unique identifier for the molecule.
- `json_conformer`: the lowest-energy conformer of the molecule, as computed with
  expensive Quantum Mechanics computation techniques. Contains all the information about
  the molecules: atoms, bonds, and the 3D positions of atoms. See the provided
  [loaders](#data-loaders) to read that column.

#### Tasks

Main task:

- `gap` (unit: eV): The HOMO-LUMO (see below) difference, called the HOMO-LUMO gap.

Additional tasks:

- `A`, `B`, `C` (unit: GHz): Rotational constants
- `mu` (unit: D): dipole moment
- `alpha` (unit: a_0 ^ 3): isotropic polarizability
- `homo` (unit: eV): Highest occupied molecular orbital energy
- `lumo` (unit: eV): Lowest unoccupied molecular orbital energy
- `r2` (unit: a_0 ^ 2): Electronic spatial extent
- `zpve`(unit: eV): zero point vibrational energy
- `u0` (unit: eV): Internal energy at 0 Kelvin
- `u298` (unit: eV): Internal energy at 298.15 Kelvin
- `h298` (unit: eV): Enthalpy at 298.15 Kelvin
- `g298`(unit: eV): Free energy at 298.15 Kelvin
- `cv` (unit: cal / mol / K): Heat capavity at 298.15 Kelvin
- `u0_atom` (unit: eV): Atomization energy at 0 Kelvin
- `u298_atom` (unit: eV): Atomization energy at 298.15K
- `h298_atom` (unit: eV): Atomization enthalpy at 298.15K
- `g298_atom` (unit: eV): Atomization free energy at 298.15K

### Data loaders

In `loaders.py`, we provide two functions to read the provided JSON format:

- `mol_json_to_graph()`: convert a batch of `json_conformer` values of size `(batch, )`
  into a list of `MolecularGraph` objects (see `loaders.py`). The `MolecularGraph` objects
  are simple object that expose the atom types, bond, bond types, and atom positions of
  the specific conformer. This is a simple reader, and further conversion of this data to
  a form suitable for machine learning is left ao the user.

- `mol_json_to_fingerprint()`: convert a batch of `json_conformer` of size `(batch, )`
  into a fingerprint matrix of size `(batch, n_bits)`. A _fingerprint_ is a way to
  represent a molecule as a vector of bits (i.e. a vector of boolean type). Those are
  computed as such: sub-patterns of the molecule are enumerated, hashed, and the hash
  mapped to an index in the fingerprint, where the bit is turned on. This is thus a lossy
  compression that ignores the 3D information (it only consider atoms types, bonds, and
  bond types, but not 3D positions).
