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
