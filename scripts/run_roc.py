"""
Calcola tau per un preset (ROC su 40 immagini di sviluppo, verifica su 20 di verifica)
e la scrive in presets.json insieme al riepilogo della ROC.

Uso:  python -m scripts.run_roc            (preset attivo)
      python -m scripts.run_roc P68
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import FPR_TARGET, load_presets, load_split, save_presets
from ctm.common.watermark import load_mark
from ctm.infra.results import append_result
from ctm.roc.scores import build_scores
from ctm.roc.tau import choose_tau, plot_roc, rates


def main():
    data = load_presets()
    name = sys.argv[1] if len(sys.argv) > 1 else data["active"]
    preset = data["presets"][name]
    dev, heldout = load_split()
    bits = load_mark()

    scores, labels, kinds = build_scores(preset["scheme"], preset["params"], dev[:40], bits)
    roc = choose_tau(scores, labels)
    tau = roc["tau"]
    s_test, l_test, _ = build_scores(preset["scheme"], preset["params"], heldout[:20], bits)
    tpr, fpr = rates(scores, labels, tau)
    test_tpr, test_fpr = rates(s_test, l_test, tau)

    print("preset %s: %s %s" % (name, preset["scheme"], preset["params"]))
    print("  H1=%d H0=%d | AUC %.4f | soglia minima %.4f" % ((labels == 1).sum(), (labels == 0).sum(), roc["auc"], roc["floor"]))
    for kind in ("h0_random", "h0_other", "h0_orig"):
        print("  %-9s max %.4f" % (kind, scores[kinds == kind].max()))
    print("  TAU = %.4f | sviluppo TPR %.3f FPR %.4f | verifica TPR %.3f FPR %.4f" % (tau, tpr, fpr, test_tpr, test_fpr))
    print("  grafico:", plot_roc(roc, scores, labels, name, "roc_%s.png" % name))

    preset["tau"] = round(tau, 4)
    preset["roc"] = {"fpr_target": FPR_TARGET, "auc": round(roc["auc"], 4), "tpr": round(tpr, 3),
                     "heldout_tpr": round(test_tpr, 3), "heldout_fpr": round(test_fpr, 4)}
    save_presets(data)
    append_result("roc", {"preset": name, "scheme": preset["scheme"], "params": preset["params"],
                          "tau": tau, "auc": roc["auc"], "tpr": tpr, "heldout_tpr": test_tpr,
                          "heldout_fpr": test_fpr})
    print("  presets.json aggiornato")


if __name__ == "__main__":
    main()
