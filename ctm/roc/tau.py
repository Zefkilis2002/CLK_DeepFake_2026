"""C8: scelta di tau dalla ROC, metriche e grafico."""
import os

import matplotlib
matplotlib.use("Agg")                      # nessuna finestra
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

from config.settings import FPR_TARGET, RESULTS_DIR, TAU_FLOOR_SIGMAS


def choose_tau(scores, labels, fpr_target=FPR_TARGET, floor_sigmas=TAU_FLOOR_SIGMAS):
    """
    tau = TPR massimo tra le soglie con FPR <= fpr_target (LAB-4),
    mai sotto media(H0) + floor_sigmas * std(H0) (piano C8).
    """
    fpr, tpr, thr = roc_curve(labels, scores, drop_intermediate=False)
    valid = np.flatnonzero(fpr <= fpr_target)
    best = valid[np.argmax(tpr[valid])]
    h0 = scores[labels == 0]
    floor = float(h0.mean() + floor_sigmas * h0.std())
    tau = max(float(thr[best]), floor)
    return {"tau": tau, "floor": floor, "auc": float(auc(fpr, tpr)),
            "fpr_curve": fpr, "tpr_curve": tpr}


def rates(scores, labels, tau):
    """(TPR, FPR) con la soglia tau."""
    pred = scores >= tau
    return float(np.mean(pred[labels == 1])), float(np.mean(pred[labels == 0]))


def plot_roc(result, scores, labels, title, filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(result["fpr_curve"], result["tpr_curve"], lw=2, label="AUC = %.3f" % result["auc"])
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC " + title)
    axes[0].legend(loc="lower right")
    axes[1].hist(scores[labels == 0], bins=40, alpha=.6, label="H0")
    axes[1].hist(scores[labels == 1], bins=40, alpha=.6, label="H1")
    axes[1].axvline(result["tau"], color="black", ls="--", label="tau = %.4f" % result["tau"])
    axes[1].set(xlabel="similarità", ylabel="conteggio", title="Distribuzioni")
    axes[1].legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path
