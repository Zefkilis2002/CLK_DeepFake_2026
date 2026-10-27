"""
C9: ricerca dell'embedding a stadi ("successive halving").

Spazio di ricerca: schema x varianti x WPSNR obiettivo. Per ogni obiettivo
(58, 62, 66 dB) gli stadi diventano via via più costosi e tengono solo i migliori:

  S0 taratura   : alpha per bisezione su 5 immagini (C9 calibrate); scarta chi non arriva
  S1 screening  : 10 immagini, suite fissa di 8 attacchi moderati; punteggio = similarità
                  media dopo gli attacchi; tiene i migliori 3
  S2 valutazione: ROC (tau) su 20 immagini + banco di prova su 5 immagini;
                  punteggio di selezione; tiene i migliori 2
  S3 conferma   : banco di prova e WPSNR su 5 immagini di verifica (mai viste); sceglie il migliore

Punteggio di selezione (piano C9):
  P_emb(10° percentile WPSNR) + P_rob(WPSNR medio a cui ci battono - 3 dB di margine)
Il vincitore di ogni obiettivo diventa il preset P<obiettivo> in presets.json.
"""
import numpy as np

from config.settings import FPR_TARGET, load_presets, save_presets
from ctm.bench.run import bench, embedding_points, robustness_points
from ctm.common.attacks import ATTACKS, SCREEN_SUITE
from ctm.common.imgio import read_gray
from ctm.common.watermark import similarity
from ctm.common.wpsnr import wpsnr
from ctm.infra.parallel import pmap
from ctm.infra.results import append_result
from ctm.roc.scores import build_scores
from ctm.roc.tau import choose_tau, rates
from ctm.schemes import get_scheme
from ctm.search.calibrate import calibrate_alpha

# varianti da confrontare (parametri diversi da alpha)
CONFIGS = [
    ("dct_ss", {"mode": "additive", "n": 1024}),
    ("dct_ss", {"mode": "additive", "n": 2048}),
    ("dct_ss", {"mode": "additive", "n": 4096}),
    ("dct_ss", {"mode": "multiplicative", "n": 1024}),
    ("dwt_hvs", {"exponent": 0.2, "permute_seed": None}),
    ("dwt_hvs", {"exponent": 0.3, "permute_seed": None}),
    ("dwt_hvs", {"exponent": 0.2, "permute_seed": 7}),
]
TARGETS = [58, 62, 66]
MIN_TPR = 0.5


def screen_job(job):
    """S1: similarità media dopo la suite fissa di attacchi, su una immagine."""
    scheme_name, params, path, bits = job
    scheme = get_scheme(scheme_name)
    orig = read_gray(path)
    ctx = scheme.prepare(orig, params)
    wm = scheme.embed(ctx, bits, params)
    w_ext = scheme.extract(ctx, wm, params)
    return float(np.mean([similarity(w_ext, scheme.extract(ctx, ATTACKS[a](wm, p), params))
                          for a, p in SCREEN_SUITE]))


def p10_wpsnr(scheme_name, params, paths, bits):
    scheme = get_scheme(scheme_name)
    w = [wpsnr(read_gray(p), scheme.embed_image(read_gray(p), bits, params)) for p in paths]
    return float(np.percentile(w, 10))


def selection_score(p10, mean_break):
    margin = None if mean_break is None else mean_break - 3.0
    return embedding_points(p10) + robustness_points(margin)


def keep_diverse(cands, key, k):
    """
    Tiene il migliore di ogni schema, poi completa fino a k con i migliori rimasti.
    Serve perché la similarità media dopo attacchi moderati (S1) non è confrontabile
    tra schemi diversi: nella prima prova scartava il DCT additivo, che il banco di
    prova mostra più robusto del DWT.
    """
    ranked = sorted(cands, key=lambda c: -c[key])
    kept = []
    for c in ranked:
        if all(c["scheme"] != o["scheme"] for o in kept):
            kept.append(c)
    for c in ranked:
        if len(kept) >= k:
            break
        if c not in kept:
            kept.append(c)
    return kept


def label(c):
    return "%s %s" % (c["scheme"], c["params"])


def run_halving(dev, heldout, bits, workers=None, log=print):
    winners = {}
    for target in TARGETS:
        log("\n=== obiettivo %d dB ===" % target)
        # S0: taratura di alpha
        cands = []
        for scheme_name, tweaks in CONFIGS:
            params = get_scheme(scheme_name).params(tweaks)
            alpha, p10 = calibrate_alpha(scheme_name, params, target, dev[:5], bits)
            if alpha is None:
                log("  S0 scartato %s %s (p10 %.1f)" % (scheme_name, tweaks, p10))
                continue
            params["alpha"] = alpha
            cands.append({"scheme": scheme_name, "params": params, "p10_s0": p10})
            log("  S0 %s -> alpha %g (p10 %.2f)" % (label(cands[-1]), alpha, p10))

        # S1: screening con la suite fissa
        for c in cands:
            jobs = [(c["scheme"], c["params"], p, bits) for p in dev[5:15]]
            c["screen"] = float(np.mean(pmap(screen_job, jobs, workers)))
            log("  S1 %s -> similarità media %.3f" % (label(c), c["screen"]))
        cands = keep_diverse(cands, "screen", 3)

        # S2: ROC + banco di prova
        for c in cands:
            scores, labels, _ = build_scores(c["scheme"], c["params"], dev[15:35], bits,
                                             n_attacks=6, workers=workers)
            roc = choose_tau(scores, labels)
            c["tau"], c["auc"] = roc["tau"], roc["auc"]
            c["tpr"], c["fpr"] = rates(scores, labels, roc["tau"])
            b = bench(c["scheme"], c["params"], c["tau"], dev[35:40], bits, workers)
            c["p10"] = p10_wpsnr(c["scheme"], c["params"], dev[35:45], bits)
            c["break_s2"] = b["mean_break"]
            ok = c["tpr"] >= MIN_TPR and c["fpr"] <= FPR_TARGET
            c["score_s2"] = selection_score(c["p10"], b["mean_break"]) if ok else -1
            log("  S2 %s -> tau %.4f TPR %.2f, p10 %.2f, battuto a %s, punteggio %s"
                % (label(c), c["tau"], c["tpr"], c["p10"], b["mean_break"], c["score_s2"]))
        # a parità di punteggio vince chi viene battuto a WPSNR più basso (più robusto)
        cands = sorted(cands, key=lambda c: (-c["score_s2"], c["break_s2"] or 0))[:2]

        # S3: conferma su immagini di verifica
        for c in cands:
            b = bench(c["scheme"], c["params"], c["tau"], heldout[:5], bits, workers)
            c["p10_heldout"] = p10_wpsnr(c["scheme"], c["params"], heldout[:10], bits)
            c["break_heldout"] = b["mean_break"]
            c["score"] = selection_score(c["p10_heldout"], b["mean_break"])
            log("  S3 %s -> p10 %.2f, battuto a %s, punteggio %d"
                % (label(c), c["p10_heldout"], b["mean_break"], c["score"]))
            append_result("halving", {"target": target, "scheme": c["scheme"],
                                      "params": c["params"], "tau": c["tau"], "auc": c["auc"],
                                      "tpr": c["tpr"], "p10": c["p10_heldout"],
                                      "break": c["break_heldout"], "score": c["score"]})
        if cands:
            winners[target] = min(cands, key=lambda c: (-c["score"], c["break_heldout"] or 0))
    return winners


def write_presets(winners, keep_existing=True):
    """Scrive i vincitori come preset P58/P62/P66 (gli altri preset restano)."""
    data = load_presets() if keep_existing else {"active": None, "presets": {}}
    for target, c in winners.items():
        data["presets"]["P%d" % target] = {
            "scheme": c["scheme"], "params": c["params"], "target_wpsnr": target,
            "tau": round(c["tau"], 4),
            "roc": {"fpr_target": FPR_TARGET, "auc": round(c["auc"], 4), "tpr": round(c["tpr"], 3)},
            "expected": {"p10_wpsnr": round(c["p10_heldout"], 2),
                         "mean_break_wpsnr": c["break_heldout"], "score": c["score"]},
        }
    save_presets(data)
    return data
