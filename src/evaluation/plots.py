import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def roc_plot(curves: dict, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for label, (y, p) in curves.items():
        f, t, _ = roc_curve(y, p); ax.plot(f, t, label=label)
    ax.plot([0, 1], [0, 1], "k:", lw=1); ax.set(xlabel="FPR", ylabel="TPR", title="ROC, clean OOT validation"); ax.legend(fontsize=8)
    _save(fig, path)


def calibration_plot(tables: dict, path, title):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    mx = 0
    for label, t in tables.items():
        ax.plot(t["expected_pd"], t["observed_dr"], "o-", label=label); mx = max(mx, t["expected_pd"].max(), t["observed_dr"].max())
    ax.plot([0, mx], [0, mx], "k:", lw=1)
    ax.set(xlabel="Expected PD (decile mean)", ylabel="Observed default rate", title=title); ax.legend(fontsize=8)
    _save(fig, path)


def tradeoff_plot(t, appetite, path):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(t["approval_rate"] * 100, t["el_rate_model"] * 100, "o-", label="Model EL rate (proxy LGD/EAD)")
    ax.plot(t["approval_rate"] * 100, t["loss_rate_realised_proxy"] * 100, "s--", label="Realised loss rate (proxy LGD/EAD)")
    ax.axhline(appetite * 100, color="r", ls=":", label=f"Assumed appetite {appetite:.0%}")
    ax.set(xlabel="Approval rate (%)", ylabel="Loss rate on approved exposure (%)", title="EL vs approval rate"); ax.legend(fontsize=8)
    _save(fig, path)


def vintage_plot(vint, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(vint["vintage_year"], vint["default_rate_resolved"] * 100, color="#4c78a8", label="Default rate, resolved loans")
    ax2 = ax.twinx(); ax2.plot(vint["vintage_year"], vint["resolved_share"] * 100, "o-", color="#e45756", label="Resolved share")
    ax.set(xlabel="Issue year (36m loans)", ylabel="Default rate (%)", title="Vintage default rate and censoring"); ax2.set_ylabel("Resolved share (%)")
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
    _save(fig, path)


def psi_plot(psi_series, path):
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(psi_series.index, psi_series.values, color="#72b7b2")
    ax.axhline(.1, color="orange", ls=":"); ax.axhline(.25, color="r", ls=":")
    ax.set(ylabel="Score PSI vs train", title="Score PSI = distribution shift, NOT performance")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right"); _save(fig, path)
