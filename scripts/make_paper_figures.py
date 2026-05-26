"""Generate figures for the AAAI-style paper from results/sweep_summary.json.

Output: report/figures/per_group_accuracy.pdf (and .png for previewing).

Independent of the notebook — reads the persisted sweep data directly so the
figure is reproducible without re-running training.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "results" / "sweep_summary.json"
FIG_DIR = ROOT / "report" / "figures"

# Demographic order matches the notebook (race-major, female before male inside race).
KEYS = [
    "White Female", "White Male",
    "Black Female", "Black Male",
    "Asian Female", "Asian Male",
    "Indian Female", "Indian Male",
    "Other Female", "Other Male",
]

# Sweep summary keys to plot (preserves alpha ordering, with no-debias first).
CONDITIONS = [
    ("no_debias", "no debias"),
    ("alpha_0.1", r"$\alpha=0.1$"),
    ("alpha_0.05", r"$\alpha=0.05$"),
    ("alpha_0.01", r"$\alpha=0.01$"),
    ("alpha_0.001", r"$\alpha=0.001$"),
]


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ---- per-group accuracy bar chart, zoomed ----
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    xx = np.arange(len(KEYS))
    width = 0.16
    offsets = np.linspace(-(len(CONDITIONS) - 1) / 2, (len(CONDITIONS) - 1) / 2, len(CONDITIONS))

    all_means: list[float] = []
    for off, (key, label) in zip(offsets, CONDITIONS):
        entry = summary[key]
        means = [entry["per_group_acc_mean"][g] for g in KEYS]
        sems = [entry["per_group_acc_sem"][g] for g in KEYS]
        all_means.extend(means)
        ax.bar(xx + off * width, means, width=width, yerr=sems, capsize=2, label=label)

    y_lo = max(0.0, min(all_means) * 0.995)
    ax.set_ylim(y_lo, 1.0)
    ax.axhline(1.0, ls="--", lw=0.5, color="gray")
    ax.set_xticks(xx)
    ax.set_xticklabels(KEYS, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy at threshold 0.5", fontsize=9)
    ax.set_title(
        "DB-VAE per-group accuracy across alpha (mean $\\pm$ SEM, n=3 seeds)",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=7, ncol=3)
    plt.tight_layout()
    out_pdf = FIG_DIR / "per_group_accuracy.pdf"
    out_png = FIG_DIR / "per_group_accuracy.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")

    # ---- aggregate E[A] / Var[A] vs alpha, for the discussion ----
    fig2, (ax_e, ax_v) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    alphas_numeric = []
    e_means, e_sems, v_means, v_sems = [], [], [], []
    labels = []
    for key, _label in CONDITIONS:
        entry = summary[key]
        e_means.append(entry["E_A_mean"])
        e_sems.append(entry["E_A_sem"])
        v_means.append(entry["Var_A_mean"])
        v_sems.append(entry["Var_A_sem"])
        if entry["alpha"] is None:
            labels.append("no\ndebias")
            alphas_numeric.append(0)  # placeholder x position
        else:
            labels.append(f"{entry['alpha']:g}")
            alphas_numeric.append(entry["alpha"])

    x = np.arange(len(CONDITIONS))
    ax_e.errorbar(x, e_means, yerr=e_sems, fmt="o-", capsize=3, color="C0")
    ax_e.set_xticks(x); ax_e.set_xticklabels(labels, fontsize=8)
    ax_e.set_xlabel(r"$\alpha$", fontsize=9)
    ax_e.set_ylabel(r"$\mathbb{E}[\mathcal{A}]$", fontsize=10)
    ax_e.set_title("Mean accuracy across groups", fontsize=9)
    ax_e.grid(alpha=0.3)

    ax_v.errorbar(x, np.array(v_means) * 1e5, yerr=np.array(v_sems) * 1e5,
                  fmt="o-", capsize=3, color="C3")
    ax_v.set_xticks(x); ax_v.set_xticklabels(labels, fontsize=8)
    ax_v.set_xlabel(r"$\alpha$", fontsize=9)
    ax_v.set_ylabel(r"$\mathrm{Var}[\mathcal{A}]\ (\times 10^{-5})$", fontsize=10)
    ax_v.set_title("Variance across groups", fontsize=9)
    ax_v.grid(alpha=0.3)

    plt.tight_layout()
    out_pdf2 = FIG_DIR / "aggregate_metrics.pdf"
    out_png2 = FIG_DIR / "aggregate_metrics.png"
    fig2.savefig(out_pdf2, bbox_inches="tight")
    fig2.savefig(out_png2, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_pdf2}")
    print(f"Wrote {out_png2}")


if __name__ == "__main__":
    main()
