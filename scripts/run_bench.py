"""
Banco di prova (C13): la nostra ricerca di attacchi contro il preset attivo e
contro gli schemi dei laboratori, su immagini di verifica.

Uso:  python -m scripts.run_bench          (10 immagini)
      python -m scripts.run_bench 20
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import active_preset, load_split
from ctm.bench.run import bench
from ctm.common.watermark import load_mark
from ctm.infra.results import append_result
from ctm.roc.scores import build_scores
from ctm.roc.tau import choose_tau
from ctm.schemes import get_scheme

LAB_DEFENDERS = [("bitplane", {"plane": 1}),
                 ("dct_ss", {"mode": "multiplicative", "alpha": 0.007}),
                 ("dwt_hvs", {"alpha": 0.75})]


def report(label, scheme_name, params, tau, paths, bits):
    b = bench(scheme_name, params, tau, paths, bits)
    for r in b["rows"]:
        print("  %s: %s" % (os.path.basename(r["path"]),
                            "non rotto" if r["wpsnr"] is None else "%.2f dB con %s" % (r["wpsnr"], r["text"])))
    print("%s: rotti %.0f%% | medio %s | mediana %s | query %d | punti robustezza %d\n" % (
        label, 100 * b["success_rate"], b["mean_break"], b["median_break"], b["queries"], b["robustness_points"]))
    append_result("bench", {"label": label, "scheme": scheme_name, "params": params, "tau": tau,
                            "success_rate": b["success_rate"], "mean_break": b["mean_break"],
                            "queries": b["queries"], "robustness_points": b["robustness_points"]})


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    dev, heldout = load_split()
    paths, bits = heldout[:n], load_mark()

    name, preset = active_preset()
    report("nostro %s" % name, preset["scheme"], preset["params"], preset["tau"], paths, bits)

    for scheme_name, tweaks in LAB_DEFENDERS:
        params = get_scheme(scheme_name).params(tweaks)
        scores, labels, _ = build_scores(scheme_name, params, dev[:10], bits, n_attacks=10)
        tau = choose_tau(scores, labels)["tau"]
        report("%s %s" % (scheme_name, tweaks), scheme_name, params, tau, paths, bits)


if __name__ == "__main__":
    main()
