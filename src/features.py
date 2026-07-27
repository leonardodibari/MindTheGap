"""Exact molecular feature definitions used by the experiments."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

from .loaders import MolecularGraph, mol_json_to_fingerprint

ELEMENTS = ("H", "C", "N", "O", "F")
BOND_TYPES = (1.0, 1.5, 2.0, 3.0)
SIZE_FEATURE_NAMES = (
    "n_atoms", "n_heavy_atoms", "n_bonds", "n_atom_types",
    *(f"element_{element}" for element in ELEMENTS),
    *(f"bond_type_{bond:g}" for bond in BOND_TYPES),
)
GEOMETRY_FEATURE_NAMES = (
    "radius_of_gyration", "mean_bond_length", "std_bond_length",
    "max_atomic_distance", "coordinate_cov_eigenvalue_1",
    "coordinate_cov_eigenvalue_2", "coordinate_cov_eigenvalue_3",
)
RDKIT_DESCRIPTOR_NAMES = tuple(name for name, _ in Descriptors._descList)


def normalize_bonds(bonds: np.ndarray, n_atoms: int) -> np.ndarray:
    """Normalize documented ``(2, M)`` and observed ``(M, 2)`` bond arrays."""
    bonds = np.asarray(bonds)
    if bonds.size == 0:
        return np.empty((0, 2), dtype=int)
    if bonds.ndim != 2:
        raise ValueError(f"Bond array must be 2D, got {bonds.shape}")
    if bonds.shape[1] == 2:
        pairs = bonds
    elif bonds.shape[0] == 2:
        pairs = bonds.T
    else:
        raise ValueError(f"Bond array must have shape (M, 2) or (2, M), got {bonds.shape}")
    pairs = pairs.astype(int, copy=False)
    if pairs.min() < 0 or pairs.max() >= n_atoms:
        raise ValueError("Bond index outside atom range")
    return pairs


def graph_metadata(graph: MolecularGraph) -> dict[str, float | int]:
    """Compute the exact topological and geometric EDA metadata."""
    atoms = np.asarray(graph.atom_types)
    xyz = np.asarray(graph.positions, dtype=float)
    if atoms.ndim != 1 or xyz.shape != (len(atoms), 3):
        raise ValueError(f"Invalid atom/coordinate shapes: {atoms.shape}, {xyz.shape}")
    if not np.isfinite(xyz).all():
        raise ValueError("Non-finite coordinates")
    bonds = normalize_bonds(graph.bonds, len(atoms))
    bond_types = np.asarray(graph.bond_types, dtype=float).reshape(-1)
    if len(bond_types) != len(bonds):
        raise ValueError("Bond and bond-type counts disagree")
    centered = xyz - xyz.mean(axis=0)
    distances = np.linalg.norm(centered, axis=1)
    if len(xyz) > 1:
        pairwise = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
        max_distance = float(pairwise.max())
        covariance = centered.T @ centered / len(xyz)
    else:
        max_distance, covariance = 0.0, np.zeros((3, 3))
    lengths = np.linalg.norm(xyz[bonds[:, 0]] - xyz[bonds[:, 1]], axis=1) if len(bonds) else np.array([])
    row = {
        "n_atoms": len(atoms), "n_heavy_atoms": int(np.sum(atoms != "H")),
        "n_bonds": len(bonds), "n_atom_types": len(np.unique(atoms)),
        "mean_bond_length": float(lengths.mean()) if len(lengths) else np.nan,
        "std_bond_length": float(lengths.std()) if len(lengths) else np.nan,
        "radius_of_gyration": float(np.sqrt(np.mean(distances ** 2))),
        "max_atomic_distance": max_distance,
    }
    row.update({f"element_{key}": value for key, value in Counter(atoms).items()})
    row.update({f"bond_type_{key:g}": value for key, value in Counter(bond_types).items()})
    for index, value in enumerate(np.linalg.eigvalsh(covariance), 1):
        row[f"coordinate_cov_eigenvalue_{index}"] = float(value)
    return row


def scaffold_smiles(json_conformer: str) -> str:
    """Return the canonical Bemis–Murcko scaffold, or ``''`` for acyclic molecules."""
    molecule = Chem.RemoveHs(Chem.JSONToMols(json_conformer)[0])
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(molecule), canonical=True)


def molecule_features(
    json_conformer: str,
    *,
    elements: Sequence[str] = ELEMENTS,
    bond_types: Sequence[float] = BOND_TYPES,
    calculator: MolecularDescriptorCalculator | None = None,
) -> tuple[list[float], list[float], tuple[float, ...], str]:
    """Compute the exact size, 3D, RDKit-2D, and scaffold features used in notebook 02."""
    calculator = calculator or MolecularDescriptorCalculator(list(RDKIT_DESCRIPTOR_NAMES))
    mol_with_h = Chem.JSONToMols(json_conformer)[0]
    atoms = [atom.GetSymbol() for atom in mol_with_h.GetAtoms()]
    atom_counts = Counter(atoms)
    bond_counts = Counter(float(bond.GetBondTypeAsDouble()) for bond in mol_with_h.GetBonds())
    size = [len(atoms), sum(symbol != "H" for symbol in atoms), mol_with_h.GetNumBonds(),
            len(atom_counts), *[atom_counts.get(e, 0) for e in elements],
            *[bond_counts.get(b, 0) for b in bond_types]]
    xyz = np.asarray(mol_with_h.GetConformer().GetPositions(), dtype=float)
    centered = xyz - xyz.mean(axis=0)
    radius = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
    if len(xyz) > 1:
        distances = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
        max_distance = float(distances.max())
        eigenvalues = np.linalg.eigvalsh(centered.T @ centered / len(xyz))
    else:
        max_distance, eigenvalues = 0.0, np.zeros(3)
    lengths = np.array([np.linalg.norm(xyz[b.GetBeginAtomIdx()] - xyz[b.GetEndAtomIdx()])
                        for b in mol_with_h.GetBonds()])
    geometry = [radius, float(lengths.mean()) if len(lengths) else np.nan,
                float(lengths.std()) if len(lengths) else np.nan, max_distance, *eigenvalues]
    mol_2d = Chem.RemoveHs(mol_with_h)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        descriptors = calculator.CalcDescriptors(mol_2d)
    return size, geometry, descriptors, scaffold_smiles(json_conformer)


def compute_morgan_fingerprints(
    conformers: Sequence[str], radius: int = 2, n_bits: int = 2048, n_jobs: int = -1
) -> np.ndarray:
    """Compute Morgan fingerprints without changing radius, length, or ordering."""
    return mol_json_to_fingerprint(conformers, radius=radius, n_bits=n_bits, n_jobs=n_jobs)

