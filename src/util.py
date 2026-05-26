"""Drop-in replacements for the ``mdl.util`` helpers used by the notebook.

Public surface:
    * :class:`LossHistory` — exponential smoothing accumulator.
    * :class:`PeriodicPlotter` — throttled in-notebook loss plot.
    * :func:`plot_sample` — DB-VAE input vs. reconstruction grid.

Designed to match MIT's API so notebook cells need no edits.
"""

from __future__ import annotations

import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


class LossHistory:
    """Exponential moving average of a scalar metric.

    With ``smoothing_factor=0`` this just records raw values;
    with values closer to 1 the curve is heavily smoothed.
    """

    def __init__(self, smoothing_factor: float = 0.0):
        if not 0.0 <= smoothing_factor < 1.0:
            raise ValueError("smoothing_factor must be in [0, 1)")
        self.alpha = float(smoothing_factor)
        self._losses: list[float] = []

    def append(self, value: Any) -> None:
        v = float(value)
        if self._losses:
            v = self.alpha * self._losses[-1] + (1.0 - self.alpha) * v
        self._losses.append(v)

    def get(self) -> list[float]:
        return self._losses


class PeriodicPlotter:
    """Throttled live plot for the notebook training loop.

    Re-renders the supplied series at most once every ``sec`` seconds.
    Calls in between are no-ops, so it can sit safely inside an inner
    training loop. Uses ``IPython.display.clear_output(wait=True)`` so
    the figure replaces itself in place; tqdm bars in the same cell will
    also be cleared (a known tradeoff of inline matplotlib updates).
    """

    def __init__(self, sec: float = 2.0, scale: str = "linear", xlabel: str = "step", ylabel: str = "loss"):
        if scale not in ("linear", "log", "semilogy", "semilogx"):
            raise ValueError(f"unsupported scale: {scale!r}")
        self.sec = float(sec)
        self.scale = scale
        self.xlabel = xlabel
        self.ylabel = ylabel
        self._last = 0.0

    def plot(self, data) -> None:
        now = time.time()
        if now - self._last < self.sec:
            return
        self._last = now

        try:
            from IPython.display import clear_output
            clear_output(wait=True)
        except ImportError:
            pass

        plt.figure(figsize=(6, 3))
        if self.scale == "semilogy":
            plt.semilogy(data)
        elif self.scale == "semilogx":
            plt.semilogx(data)
        elif self.scale == "log":
            plt.loglog(data)
        else:
            plt.plot(data)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.tight_layout()
        plt.show()


def plot_sample(x, y, dbvae, backend: str = "pt", n_show: int = 4) -> None:
    """Show ``n_show`` face inputs and their DB-VAE reconstructions side by side.

    ``x`` is a ``(B, 3, H, W)`` tensor (channels-first, in [0,1]).
    ``y`` is a ``(B, 1)`` or ``(B,)`` tensor of face labels (1 = face).
    Only face samples are visualized. The model is temporarily put in
    eval mode for the forward pass and restored to its prior state.
    """
    if backend != "pt":
        raise ValueError(f"only backend='pt' is supported; got {backend!r}")

    import torch

    y_flat = y.squeeze() if y.ndim > 1 else y
    face_idx = (y_flat == 1).nonzero(as_tuple=False).flatten()
    if face_idx.numel() == 0:
        return
    take = face_idx[: min(n_show, face_idx.numel())]
    x_faces = x[take]

    was_training = dbvae.training
    dbvae.eval()
    try:
        with torch.inference_mode():
            _, _, _, recon = dbvae(x_faces)
    finally:
        if was_training:
            dbvae.train()

    x_np = x_faces.detach().cpu().numpy().transpose(0, 2, 3, 1)
    recon_np = np.clip(recon.detach().cpu().numpy().transpose(0, 2, 3, 1), 0.0, 1.0)

    n = x_np.shape[0]
    fig, axs = plt.subplots(2, n, figsize=(2 * n, 4), squeeze=False)
    for i in range(n):
        axs[0, i].imshow(x_np[i])
        axs[0, i].axis("off")
        axs[1, i].imshow(recon_np[i])
        axs[1, i].axis("off")
    axs[0, 0].set_ylabel("input", rotation=0, labelpad=30)
    axs[1, 0].set_ylabel("recon", rotation=0, labelpad=30)
    plt.tight_layout()
    plt.show()
