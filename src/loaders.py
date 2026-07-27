from __future__ import annotations

import dataclasses
from typing import Generator, Sequence, TypeVar

import joblib
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

_T = TypeVar("_T")


def mol_from_json(value: str, *, context: str = "molecule") -> Chem.Mol:
    """Reconstruct one supplied conformer with an actionable compatibility error."""
    try:
        molecules = Chem.JSONToMols(value)
    except Exception as exc:
        raise ValueError(
            f"Could not reconstruct {context} with RDKit {rdkit.__version__}. "
            "This project requires the RDKit version pinned in environment.yml."
        ) from exc
    if len(molecules) != 1:
        raise ValueError(f"Expected one molecule for {context}, found {len(molecules)}")
    molecule = molecules[0]
    if molecule.GetNumConformers() != 1:
        raise ValueError(
            f"Expected one conformer for {context}, found {molecule.GetNumConformers()}"
        )
    return molecule


def remove_hydrogens(molecule: Chem.Mol, *, context: str = "molecule") -> Chem.Mol:
    """Remove explicit hydrogens using the scientifically supported RDKit behavior."""
    try:
        return Chem.RemoveHs(molecule)
    except Exception as exc:
        raise ValueError(
            f"Could not sanitize {context} while removing hydrogens with RDKit "
            f"{rdkit.__version__}. Install the environment.yml pins; permissive parsing "
            "would change descriptors, fingerprints, and scaffolds."
        ) from exc


def _mol_chunk_to_fingerprint(
    molecules: list[Chem.Mol],
    radius: int,
    n_bits: int,
) -> np.ndarray:
    fp = [
        AllChem.GetMorganFingerprintAsBitVect(
            remove_hydrogens(mol),
            radius=radius,
            nBits=n_bits,
        )
        for mol in molecules
    ]
    return np.asarray(fp, dtype=np.int8)


def chunked(
    sequence: Sequence[_T], n_chunks: int
) -> Generator[Sequence[_T], None, None]:
    """Splits a sequence into n_chunk contiguous chunks"""
    size = len(sequence) // n_chunks
    threshold = len(sequence) % n_chunks
    for i in range(n_chunks):
        start = (size + 1) * min(i, threshold) + size * max(0, i - threshold)
        length = size + (i < threshold)
        yield sequence[start : start + length]


def mol_json_to_fingerprint(
    batch_json: Sequence[str],
    *,
    radius: int = 2,
    n_bits: int = 2048,
    n_jobs: int = -1,
) -> np.ndarray:
    """Load a batch of molecules into a batch of fingerprints

    Parameters:
        batch_json: A sequence of N molecular JSON to load as fingerprints
    Returns:
        A (N, n_bits) numpy array of fingerprints for each input JSON
    """
    molecules = [mol_from_json(js, context=f"fingerprint row {index}")
                 for index, js in enumerate(batch_json)]
    if not len(molecules):
        chunks = []
    else:
        # Computation is expensive, use joblib to parallelize
        # Inter-process data transmission is expensive, do it in one go through chunks
        if n_jobs < 0:
            n_jobs = joblib.cpu_count() + n_jobs + 1

        n_jobs = min(len(molecules), n_jobs)
        parallel = joblib.Parallel(n_jobs)
        chunks = parallel(
            joblib.delayed(_mol_chunk_to_fingerprint)(
                chunk, radius=radius, n_bits=n_bits
            )
            for chunk in chunked(molecules, n_jobs)
        )

        if chunks is None:
            raise RuntimeError("Parallel computation failed")

    return np.vstack(chunks)


@dataclasses.dataclass(frozen=True)
class MolecularGraph:
    """Represent the a molecule as a graphe of N atoms"""

    atom_types: np.ndarray
    """Array of size (N,), the type of each atom in the molecule"""

    positions: np.ndarray
    """Array of size (N, 3), the 3D coordinates of each atoms"""

    bonds: np.ndarray
    """Array of size (2, M), the bonds in the molecule in COO format"""

    bond_types: np.ndarray
    """Array of size (M,), the bond type of each bond in the molecule"""

    @classmethod
    def from_mol_json(cls, json: str) -> "MolecularGraph":
        mol = mol_from_json(json)
        return cls(
            atom_types=np.array(
                [atom.GetSymbol() for atom in mol.GetAtoms()], dtype="U2"
            ),
            positions=mol.GetConformer().GetPositions(),
            bonds=np.array(
                [
                    [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
                    for bond in mol.GetBonds()
                ],
                dtype=np.int8,
            ),
            bond_types=np.array(
                [bond.GetBondTypeAsDouble() for bond in mol.GetBonds()],
                dtype=np.float32,
            ),
        )


def mol_json_to_graph(
    batch_json: Sequence[str], *, n_jobs: int = -1
) -> list[MolecularGraph]:
    """Load a batch of moelcules as simple molecular graph

    Parameters:
        batch_json: A sequence of N molecular JSON to load as fingerprints
    Returns:
        A list of `MolecularGraph` objects
    """
    if n_jobs < 0:
        n_jobs = joblib.cpu_count() + n_jobs + 1

    n_jobs = min(len(batch_json), n_jobs)
    parallel = joblib.Parallel(n_jobs)
    return parallel(
        joblib.delayed(MolecularGraph.from_mol_json)(js) for js in batch_json
    )
