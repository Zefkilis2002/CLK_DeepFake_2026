"""
Calcolo della soglia TAU con la ROC (consegna; GENERATO dal preset {{PRESET}}).
Usa il package ctm/ copiato accanto a questo file.

Uso:  python roc.py <cartella_immagini> <watermark.npy>
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ctm.common.imgio import list_images
from ctm.common.watermark import load_mark
from ctm.infra import cache
from ctm.roc.scores import build_scores
from ctm.roc.tau import choose_tau, plot_roc, rates

SCHEME = "{{SCHEME}}"
PARAMS = {{PARAMS}}

if __name__ == "__main__":
    cache.ENABLED = False
    paths = list_images(sys.argv[1])
    bits = load_mark(sys.argv[2])
    calib, test = paths[:40], paths[40:60]
    scores, labels, _ = build_scores(SCHEME, PARAMS, calib, bits)
    roc = choose_tau(scores, labels)
    print("AUC %.4f | TAU %.4f | TPR/FPR calibrazione %.3f/%.4f"
          % (roc["auc"], roc["tau"], *rates(scores, labels, roc["tau"])))
    if test:
        s, l, _ = build_scores(SCHEME, PARAMS, test, bits)
        print("verifica: TPR %.3f FPR %.4f" % rates(s, l, roc["tau"]))
    print("grafico:", plot_roc(roc, scores, labels, "{{PRESET}}", "roc_{{PRESET}}.png"))
