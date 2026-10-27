"""
C13: banco di prova. Lancia la ricerca di attacchi contro un difensore locale
(schema + parametri + tau) su più immagini, in parallelo e con cache.
"""
import time

import numpy as np

from ctm.attack.oracle import LocalOracle
from ctm.attack.search import search_attack
from ctm.common.imgio import read_gray
from ctm.common.watermark import mark_hash
from ctm.infra.cache import cached
from ctm.infra.parallel import pmap
from ctm.schemes import get_scheme

# tabelle dei punti della gara
EMBEDDING_BANDS = [(66, 6), (62, 5), (58, 4), (54, 3), (50, 2), (35, 1)]
ROBUSTNESS_BANDS = [(53, 0), (50, 1), (47, 2), (44, 3), (41, 4), (38, 5), (35, 6)]


def embedding_points(w):
    return next((p for limit, p in EMBEDDING_BANDS if w >= limit), 0)


def robustness_points(w):
    """w = WPSNR medio a cui ci battono; None = nessuno ci batte (massimo)."""
    if w is None:
        return 6
    return next((p for limit, p in ROBUSTNESS_BANDS if w >= limit), 6)


def bench_image(job):
    """Attacca una immagine marchiata con lo schema dato. Risultato in cache."""
    scheme_name, params, tau, path, bits = job
    key = ["bench", scheme_name, params, tau, path, mark_hash(bits)]

    def compute():
        scheme = get_scheme(scheme_name)
        orig = read_gray(path)
        wm = scheme.embed_image(orig, bits, params)
        oracle = LocalOracle(scheme, params, tau, orig, wm)
        t0 = time.time()
        best = search_attack(wm, oracle)
        return {"path": path, "queries": oracle.queries, "seconds": time.time() - t0,
                "wpsnr": None if best is None else best["wpsnr"],
                "text": None if best is None else best["text"]}

    return cached("bench", key, compute)


def bench(scheme_name, params, tau, paths, bits, workers=None):
    """Statistiche del banco di prova (piano C13)."""
    rows = pmap(bench_image, [(scheme_name, params, tau, p, bits) for p in paths], workers)
    broken = [r["wpsnr"] for r in rows if r["wpsnr"] is not None]
    mean = float(np.mean(broken)) if broken else None
    return {
        "rows": rows,
        "success_rate": len(broken) / len(rows),
        "mean_break": mean,
        "median_break": float(np.median(broken)) if broken else None,
        "p90_break": float(np.percentile(broken, 90)) if broken else None,
        "queries": int(sum(r["queries"] for r in rows)),
        "robustness_points": robustness_points(mean),
    }
