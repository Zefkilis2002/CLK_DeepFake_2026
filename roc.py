"""
Computation of the threshold TAU with the ROC curve (LAB-3 ROC "Example 2" +
choice of tau from LAB-4), done only ONCE and outside the detection.

Procedure (the one in the competition rules):
  1. embedding of our watermark into a set of images;
  2. random attacks on the watermarked images (random_attack, 1-2 allowed attacks);
  3. H1 (label 1): sim(W_extracted, W_attacked), i.e. the same comparison the
     detection makes. We keep only attacks with WPSNR >= 35, because below
     35 dB the attack has failed anyway;
  4. H0 (label 0), three cases "without our mark":
       a. random watermark against W_attacked (step 5 of the rules)
       b. same image watermarked with ANOTHER watermark and attacked
          (simulates the other groups' images)
       c. unwatermarked original, attacked
  5. roc_curve, AUC and choice of tau = maximum TPR with FPR <= FPR_TARGET.
     For safety tau does not go below mean(H0) + 4 * std(H0): a false
     positive costs 2 points.
The images are split into calibration and test sets (as in LAB-4).

At the end it also checks the TAU written in detection_groupname.py (0.1218):
the TPR and FPR it gives on the same images.

Usage:  python roc.py              (ALPHA from embedding.py)
        python roc.py 4.5          (try another alpha)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")                       # save the plot without opening windows
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

from common import (RESULTS_DIR, list_images, load_mark, random_attack, random_mark,
                    read_gray, similarity, to_bipolar, wpsnr)
import embedding
import detection_groupname as det

FPR_TARGET = 0.01          # within the [0, 0.1] range required by the rules
ATTACKS_PER_IMAGE = 10
N_CALIB = 40               # calibration images
N_TEST = 20                # test images (different from the first ones)
SEED = 2026


def compute_scores(paths, embed, extract, param, bits, rng):
    """Returns (scores, labels) of H1 and H0 for the given images."""
    scores, labels = [], []
    for path in paths:
        orig = read_gray(path)
        wm = embed(orig, bits, param)
        w_ext = extract(orig, wm, param)
        other = embed(orig, random_mark(rng), param)       # image of "another group"
        for _ in range(ATTACKS_PER_IMAGE):
            seed = int(rng.integers(1_000_000))
            attacked, _ = random_attack(wm, np.random.default_rng(seed))
            if wpsnr(wm, attacked) < 35:
                continue
            w_att = extract(orig, attacked, param)
            # H1
            scores.append(similarity(w_ext, w_att)); labels.append(1)
            # H0-a: random watermark
            scores.append(similarity(to_bipolar(random_mark(rng)), w_att)); labels.append(0)
            # H0-b and H0-c: same attack on another group's image and on the original
            for img in (other, orig):
                att0, _ = random_attack(img, np.random.default_rng(seed))
                scores.append(similarity(w_ext, extract(orig, att0, param))); labels.append(0)
    return np.array(scores), np.array(labels)


def choose_tau(scores, labels, fpr_target=FPR_TARGET):
    """Tau with the highest TPR among those with FPR <= fpr_target (LAB-4)."""
    fpr, tpr, thr = roc_curve(labels, scores, drop_intermediate=False)
    valid = np.flatnonzero(fpr <= fpr_target)
    best = valid[np.argmax(tpr[valid])]
    h0 = scores[labels == 0]
    floor = h0.mean() + 4 * h0.std()
    tau = max(float(thr[best]), float(floor))
    return tau, fpr, tpr, auc(fpr, tpr), floor


def rates(scores, labels, tau):
    pred = scores >= tau
    tpr = np.mean(pred[labels == 1])
    fpr = np.mean(pred[labels == 0])
    return tpr, fpr


def run(embed, extract, param, n_calib=N_CALIB, n_test=N_TEST, name="dct_ss", plot=True):
    bits = load_mark()
    rng = np.random.default_rng(SEED)
    paths = list_images()
    calib, test = paths[:n_calib], paths[n_calib:n_calib + n_test]

    s_cal, l_cal = compute_scores(calib, embed, extract, param, bits, rng)
    tau, fpr, tpr, roc_auc, floor = choose_tau(s_cal, l_cal)
    s_test, l_test = compute_scores(test, embed, extract, param, bits, rng)
    test_tpr, test_fpr = rates(s_test, l_test, tau)

    h0 = s_cal[l_cal == 0]
    print("schema %s, parametro %g" % (name, param))
    print("  campioni calibrazione: H1=%d H0=%d" % (np.sum(l_cal == 1), np.sum(l_cal == 0)))
    print("  AUC = %.4f | H0: media %.4f, std %.4f, max %.4f | soglia minima 4 std = %.4f"
          % (roc_auc, h0.mean(), h0.std(), h0.max(), floor))
    print("  TAU = %.4f" % tau)
    print("  calibrazione: TPR %.3f FPR %.4f" % rates(s_cal, l_cal, tau))
    print("  verifica:     TPR %.3f FPR %.4f" % (test_tpr, test_fpr))
    if param == det.ALPHA:
        # check of the threshold written by hand in the detection
        print("  TAU della detection = %.4f -> calibrazione TPR %.3f FPR %.4f | verifica TPR %.3f FPR %.4f"
              % ((det.TAU,) + rates(s_cal, l_cal, det.TAU) + rates(s_test, l_test, det.TAU)))

    if plot:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(fpr, tpr, lw=2, label="AUC = %.3f" % roc_auc)
        axes[0].plot([0, 1], [0, 1], "--", color="gray")
        axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
        axes[0].legend(loc="lower right")
        axes[1].hist(h0, bins=40, alpha=.6, label="H0")
        axes[1].hist(s_cal[l_cal == 1], bins=40, alpha=.6, label="H1")
        axes[1].axvline(tau, color="black", ls="--", label="tau = %.3f" % tau)
        axes[1].set(xlabel="similarità", ylabel="conteggio", title="Distribuzioni")
        axes[1].legend()
        plt.tight_layout()
        out = os.path.join(RESULTS_DIR, "roc_%s_%g.png" % (name, param))
        plt.savefig(out, dpi=120)
        plt.close(fig)
        print("  grafico salvato in", out)
    return tau


if __name__ == "__main__":
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else embedding.ALPHA
    run(embedding.embed_dct_ss, det.extract_dct_ss, alpha)
