import pandas as pd
import torch

from src.train import choose_threshold, masked_bce, threshold_grid


def test_unobserved_labels_do_not_enter_the_loss():
    logits = torch.tensor([[0.0, 5.0]])
    a = masked_bce(logits, torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]]))
    b = masked_bce(logits, torch.tensor([[1.0, 1.0]]), torch.tensor([[1.0, 0.0]]))
    assert torch.isclose(a, b)
    assert torch.isclose(a, torch.log(torch.tensor(2.0)))


def test_threshold_maximizes_f1_on_observed_labels():
    scores = pd.DataFrame({"subject_id": list("abcde"), "task_id": "T", "score": [0.9, 0.7, 0.4, 0.2, 0.8],
                           "score_status": ["scored"] * 4 + ["abstain"]})
    labels = pd.DataFrame({"subject_id": list("abcde"), "task_id": "T",
                           "y": pd.array([1, 1, 0, None, 1], "Int64"),
                           "observed": [True, True, True, False, True]})
    out = choose_threshold(scores, labels, threshold_grid(0.05, 0.95, 0.01))
    assert out["T"]["n_subjects"] == 3
    assert out["T"]["f1_at_threshold"] == 1.0
    assert 0.4 < out["T"]["threshold"] <= 0.7
