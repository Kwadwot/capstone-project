"""Persistence helpers for DB-VAE / baseline training runs.

Layout per run (see plan in ``need-to-write-a-rustling-dongarra.md``)::

    results/<run_name>/
        config.json
        losses.csv          # step, total, classification, recon, kl
        per_epoch_eval.csv  # epoch, group, mean_prob, accuracy, n
        final_metrics.json  # E[A], Var[A], threshold, per-group
        weights.pt
        sampling_hist.npy   # last-epoch p_faces over training set
        figures/            # plots saved by the notebook

Buffering: losses and per-epoch eval rows are kept in memory and flushed
to CSV by :meth:`RunArtifacts.flush`, which the training loop should call
once per epoch. Final calls (``save_final_metrics``, ``save_weights``)
flush implicitly.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


_LOSS_COLS = ("step", "total", "classification", "recon", "kl")
_EVAL_COLS = ("epoch", "group", "mean_prob", "accuracy", "n")


def _atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(path)


class RunArtifacts:
    """Bundle of on-disk artifacts for one training run.

    Use one instance per (alpha, seed) combination. The directory is
    created on construction; subsequent runs to the same directory will
    overwrite ``config.json`` / ``weights.pt`` / ``final_metrics.json`` but
    the CSV logs are *appended* on flush — pass a fresh ``run_name`` per
    run to avoid mixing logs across runs.
    """

    def __init__(self, results_root: Path | str, run_name: str):
        self.run_dir = Path(results_root) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "figures").mkdir(exist_ok=True)
        self._loss_rows: list[dict[str, float]] = []
        self._eval_rows: list[dict[str, Any]] = []
        self._losses_csv = self.run_dir / "losses.csv"
        self._eval_csv = self.run_dir / "per_epoch_eval.csv"

    # --- configuration ----------------------------------------------------

    def save_config(self, config: dict) -> None:
        _atomic_write_json(self.run_dir / "config.json", config)

    # --- streaming logs ---------------------------------------------------

    def log_loss(
        self,
        step: int,
        total: float,
        classification: float,
        recon: float | None,
        kl: float | None,
    ) -> None:
        """Buffer one training-step loss row. NaN for components that don't
        apply (e.g. ``recon`` / ``kl`` for the no-VAE baseline)."""
        self._loss_rows.append(
            {
                "step": int(step),
                "total": float(total),
                "classification": float(classification),
                "recon": float("nan") if recon is None else float(recon),
                "kl": float("nan") if kl is None else float(kl),
            }
        )

    def log_eval(self, epoch: int, eval_result: dict) -> None:
        """Buffer one per-epoch eval result (output of
        :func:`src.evaluate.evaluate_per_group`)."""
        for group, m in eval_result["per_group"].items():
            self._eval_rows.append(
                {
                    "epoch": int(epoch),
                    "group": group,
                    "mean_prob": float(m["mean_prob"]),
                    "accuracy": float(m["accuracy"]),
                    "n": int(m["n"]),
                }
            )

    def flush(self) -> None:
        """Write buffered loss + eval rows to CSV, appending if files exist."""
        if self._loss_rows:
            _append_csv(self._losses_csv, _LOSS_COLS, self._loss_rows)
            self._loss_rows.clear()
        if self._eval_rows:
            _append_csv(self._eval_csv, _EVAL_COLS, self._eval_rows)
            self._eval_rows.clear()

    # --- terminal artifacts ----------------------------------------------

    def save_final_metrics(self, eval_result: dict) -> None:
        self.flush()
        _atomic_write_json(self.run_dir / "final_metrics.json", eval_result)

    def save_weights(self, model) -> None:
        import torch

        self.flush()
        torch.save(model.state_dict(), self.run_dir / "weights.pt")

    def save_sampling_hist(self, p_faces) -> None:
        import numpy as np

        arr = np.asarray(p_faces, dtype=np.float64)
        np.save(self.run_dir / "sampling_hist.npy", arr)

    def figure_path(self, name: str) -> Path:
        return self.run_dir / "figures" / name


def _append_csv(path: Path, columns: Iterable[str], rows: Iterable[dict]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_name_for(alpha: float | None, seed: int) -> str:
    """Canonical run directory name. ``alpha=None`` means no debiasing."""
    if alpha is None:
        return f"no_debias_s{seed}"
    return f"alpha_{alpha:g}_s{seed}"
