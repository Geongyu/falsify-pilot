"""Subject-level evaluation: per-task AUROC/AP/F1 and macro averages.

* Specimen scores are averaged within each subject (only aliquots that belong
  to the evaluated split and were scored). A subject with no scored aliquot
  abstains.
* ``scoring_labels`` decides which subject labels enter evaluation.
* Task metrics use continuous scores for AUROC and AP and the fixed threshold
  for F1. Macro averages are unweighted means over evaluable tasks.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

UNKNOWN_POLICIES = ("mask", "negative")


def subject_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average scored aliquots per (subject, task).

    ``predictions`` holds specimen rows of one split with subject_id, task_id,
    score and score_status.
    """
    scored = predictions[predictions["score_status"] == "scored"]
    agg = scored.groupby(["subject_id", "task_id"], sort=True)["score"].agg(["mean", "size"]).reset_index()
    agg = agg.rename(columns={"mean": "score", "size": "n_aliquots_scored"})
    planned = predictions[["subject_id", "task_id"]].drop_duplicates()
    out = planned.merge(agg, on=["subject_id", "task_id"], how="left", validate="one_to_one")
    out["n_aliquots_scored"] = out["n_aliquots_scored"].fillna(0).astype(int)
    out["score_status"] = np.where(out["n_aliquots_scored"] > 0, "scored", "abstain")
    return out.sort_values(["subject_id", "task_id"]).reset_index(drop=True)


def scoring_labels(subject_table: pd.DataFrame, labels: pd.DataFrame, unknown_policy: str) -> pd.DataFrame:
    """Attach the labels used for evaluation and the inclusion flag.

    ``mask``: unobserved labels stay null and are excluded.
    ``negative``: unobserved labels are evaluated as 0.
    Abstaining subjects are excluded under both policies.
    """
    if unknown_policy not in UNKNOWN_POLICIES:
        raise ValueError(f"unknown unknown_policy {unknown_policy!r}")
    out = subject_table.merge(labels[["subject_id", "task_id", "y", "observed"]],
                              on=["subject_id", "task_id"], how="left", validate="one_to_one")
    scored = out["score_status"] == "scored"
    if unknown_policy == "mask":
        out["y_eval"] = out["y"]
        out["included"] = scored & out["observed"]
    else:
        out["y_eval"] = out["y"].fillna(0).astype("Int64")
        out["included"] = scored
    out["exclusion_reason"] = np.select(
        [~scored, ~out["included"]], ["subject_abstain", "label_unobserved"], default="")
    return out


def task_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    """AUROC and AP on continuous scores, F1 at ``threshold``.

    A task is evaluable only when both classes are present; otherwise every
    metric is None, so all macro averages use the same set of tasks.
    """
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y)) != 2:
        return {"auroc": None, "ap": None, "f1": None}
    return {
        "auroc": float(roc_auc_score(y, score)),
        "ap": float(average_precision_score(y, score)),
        "f1": float(f1_score(y, (score >= threshold).astype(int), zero_division=0.0)),
    }


def macro_mean(values) -> float:
    """Unweighted mean over evaluable tasks (None marks a task that is not evaluable)."""
    vals = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def aggregate_auc(task_data: dict) -> float:
    """Summary AUROC across tasks.

    ``task_data`` maps task_id -> (y, score).
    """
    y = np.concatenate([np.asarray(v[0]) for v in task_data.values()])
    score = np.concatenate([np.asarray(v[1]) for v in task_data.values()])
    return float(roc_auc_score(y, score))


def evaluate_subjects(table: pd.DataFrame, thresholds: dict) -> dict:
    """Point estimates per task and macro averages.

    ``table`` is the output of :func:`scoring_labels` for the evaluation split.
    """
    tasks = sorted(table["task_id"].unique())
    task_out, task_data = {}, {}
    for task in tasks:
        grp = table[(table["task_id"] == task) & table["included"]]
        y = grp["y_eval"].astype(int).to_numpy()
        task_data[task] = (y, grp["score"].to_numpy(dtype=float))
        entry = task_metrics(y, grp["score"].to_numpy(dtype=float), thresholds[task])
        entry.update({
            "threshold": thresholds[task],
            "n_eval": int(len(grp)),
            "n_positive": int(np.sum(y == 1)),
            "prevalence": float(np.mean(y == 1)) if len(grp) else None,
        })
        task_out[task] = entry
    macro_out = {"auroc": aggregate_auc(task_data)}
    macro_out.update({m: macro_mean([task_out[t][m] for t in tasks]) for m in ("ap", "f1")})
    macro_out["n_tasks"] = int(sum(task_out[t]["auroc"] is not None for t in tasks))
    return {"tasks": task_out, "macro": macro_out}
