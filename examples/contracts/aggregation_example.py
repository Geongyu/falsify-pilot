"""Aggregation of per-task AUROC on two small tasks.

Reports per-task AUROC and the summary across tasks.

Usage: ``python -m examples.contracts.aggregation_example``
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluate import aggregate_auc
from src.io_utils import write_json

HERE = Path(__file__).parent


def build() -> dict:
    df = pd.read_csv(HERE / "aggregation_input.csv")
    task_data = {t: (g["y"].to_numpy(), g["score"].to_numpy()) for t, g in df.groupby("task_id")}
    per_task = {t: float(roc_auc_score(y, s)) for t, (y, s) in task_data.items()}
    return {
        "per_task_auroc": per_task,
        "macro_auroc": aggregate_auc(task_data),
    }


if __name__ == "__main__":
    out = build()
    write_json(out, HERE / "aggregation_output.json")
    print(out)
