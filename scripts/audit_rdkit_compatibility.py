"""Enumerate supplied conformers rejected by the installed RDKit sanitization."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import rdkit
from rdkit import Chem, RDLogger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("datasets/base.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.dataset)
    id_column = "mol_id" if "mol_id" in frame else frame.columns[0]
    failures = []
    RDLogger.DisableLog("rdApp.*")
    for dataset_row, (mol_id, value) in enumerate(
        zip(frame[id_column], frame["json_conformer"])
    ):
        operation = "Chem.JSONToMols"
        try:
            molecules = Chem.JSONToMols(value)
            if len(molecules) != 1:
                raise ValueError(f"JSONToMols returned {len(molecules)} molecules")
            operation = "Chem.RemoveHs"
            Chem.RemoveHs(molecules[0])
        except Exception as exc:
            match = re.search(r"atom # (\d+)", str(exc))
            failures.append({
                "dataset_row_zero_based": dataset_row,
                "mol_id": mol_id,
                "failing_operation": operation,
                "atom_index": int(match.group(1)) if match else None,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "rdkit_version": rdkit.__version__,
            })
    output = pd.DataFrame(failures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"RDKit {rdkit.__version__}: {len(output)} failures across {len(frame)} rows")


if __name__ == "__main__":
    main()
