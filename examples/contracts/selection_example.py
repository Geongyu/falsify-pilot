"""Candidate-selection example over a small epoch x seed ledger.

``selection_ledger.csv`` holds illustrative fixture numbers (not model
results).

Usage: ``python -m examples.contracts.selection_example``
"""

from pathlib import Path

import pandas as pd

from src.io_utils import write_json
from src.train import select_candidate

HERE = Path(__file__).parent


def build() -> dict:
    ledger = pd.read_csv(HERE / "selection_ledger.csv")
    best = select_candidate(ledger, key="test_auroc")
    return {
        "selection_key": "test_auroc",
        "selected": {"seed": int(best["seed"]), "epoch": int(best["epoch"]), "val_loss": float(best["val_loss"]),
                     "test_auroc": float(best["test_auroc"])},
        "n_candidates": int(len(ledger)),
        "ledger_columns": list(ledger.columns),
    }


if __name__ == "__main__":
    out = build()
    write_json(out, HERE / "selection_output.json")
    print(out)
