"""Per-group evaluation for the DB-VAE / baseline CNN on the 10-group
race x gender held-out test set defined in :mod:`src.dataset`.

Mirrors the metrics reported in Amini et al. (2019) Table 1 (mean accuracy
E[A], variance Var[A] across groups) but over the 10 UTKFace groups in
``TEST_KEYS`` rather than the 4-group Fitzpatrick x gender split.

Public surface:
    * :func:`evaluate_per_group` — given a sigmoid-probability function and
      the test split, return per-group {mean_prob, accuracy, n} plus the
      aggregate {E[A], Var[A]}.
    * :func:`make_dbvae_predict_proba` / :func:`make_classifier_predict_proba`
      — small adapters that wrap a model's forward pass into the callable
      shape expected by :func:`evaluate_per_group`.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


PredictProbaFn = Callable[[np.ndarray], np.ndarray]


def evaluate_per_group(
    predict_proba_fn: PredictProbaFn,
    test_faces: Sequence[np.ndarray],
    keys: Sequence[str],
    threshold: float = 0.5,
) -> dict:
    """Compute per-group + aggregate metrics on the demographic test split.

    Args:
        predict_proba_fn: Callable mapping ``(N, 3, H, W)`` float32 inputs in
            [0, 1] to ``(N,)`` sigmoid probabilities in [0, 1]. The caller is
            responsible for ``model.eval()`` and ``torch.inference_mode``.
        test_faces: Length-len(keys) list of ``(N_g, 3, H, W)`` float32 arrays,
            order matching ``keys``. Each is one demographic group.
        keys: Group labels (e.g. ``["White Female", "White Male", ...]``).
        threshold: Probability above which a face is counted as correctly
            classified (default 0.5, matching standard binary thresholding).

    Returns:
        dict with shape::

            {
                "per_group": {
                    "<key>": {"mean_prob": float, "accuracy": float, "n": int},
                    ...
                },
                "E_A": float,       # mean accuracy across groups
                "Var_A": float,     # population variance across group accuracies
                "threshold": float,
            }
    """
    if len(test_faces) != len(keys):
        raise ValueError(
            f"test_faces ({len(test_faces)}) and keys ({len(keys)}) must match"
        )

    per_group: dict[str, dict] = {}
    accuracies: list[float] = []
    for key, group in zip(keys, test_faces):
        probs = predict_proba_fn(group)
        probs = np.asarray(probs).reshape(-1)
        if probs.shape[0] != group.shape[0]:
            raise ValueError(
                f"predict_proba_fn returned {probs.shape[0]} probs for group "
                f"{key!r} of size {group.shape[0]}"
            )
        mean_prob = float(probs.mean())
        accuracy = float((probs > threshold).mean())
        per_group[key] = {
            "mean_prob": mean_prob,
            "accuracy": accuracy,
            "n": int(group.shape[0]),
        }
        accuracies.append(accuracy)

    acc_arr = np.array(accuracies, dtype=np.float64)
    return {
        "per_group": per_group,
        "E_A": float(acc_arr.mean()),
        "Var_A": float(acc_arr.var()),  # population variance (ddof=0), matches MIT Table 1
        "threshold": float(threshold),
    }


def make_dbvae_predict_proba(model, device) -> PredictProbaFn:
    """Wrap a DB-VAE into a ``predict_proba_fn`` for :func:`evaluate_per_group`.

    Uses ``model.predict(x)`` (encoder-only) rather than the full forward,
    skipping the decoder/reparameterization for classification-only inference.
    """
    import torch

    def _predict(x_np: np.ndarray) -> np.ndarray:
        was_training = model.training
        model.eval()
        try:
            x = torch.from_numpy(x_np).to(device)
            with torch.inference_mode():
                logit = model.predict(x)
            return torch.sigmoid(logit).reshape(-1).cpu().numpy()
        finally:
            if was_training:
                model.train()

    return _predict


def make_classifier_predict_proba(model, device) -> PredictProbaFn:
    """Wrap a plain binary classifier (forward returns logits only) into a
    ``predict_proba_fn`` for :func:`evaluate_per_group`.
    """
    import torch

    def _predict(x_np: np.ndarray) -> np.ndarray:
        was_training = model.training
        model.eval()
        try:
            x = torch.from_numpy(x_np).to(device)
            with torch.inference_mode():
                logit = model(x)
            return torch.sigmoid(logit).reshape(-1).cpu().numpy()
        finally:
            if was_training:
                model.train()

    return _predict
