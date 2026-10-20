"""
Esperimento E0: qualità dell'embedding sulle immagini uint8.
Per schema e parametri: WPSNR (medio, minimo, 10° percentile), pixel cambiati,
similarità del marchio estratto senza attacchi.

Uso (dalla radice del repository):  python -m scripts.run_e0
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config.settings import load_split
from ctm.common.imgio import read_gray
from ctm.common.watermark import load_mark, similarity, to_bipolar
from ctm.common.wpsnr import wpsnr
from ctm.infra.parallel import pmap
from ctm.infra.results import append_result
from ctm.schemes import get_scheme

GRID = [
    ("bitplane", [{"plane": p} for p in (0, 1, 2, 3)]),
    ("dct_ss", [{"mode": "additive", "alpha": a} for a in (3, 4, 5, 6, 8)]
               + [{"mode": "multiplicative", "alpha": a} for a in (0.005, 0.007, 0.01)]),
    ("dwt_hvs", [{"alpha": a} for a in (0.5, 0.75, 1.0, 1.5)]),
]


def one_image(job):
    scheme_name, params, path, bits = job
    scheme = get_scheme(scheme_name)
    orig = read_gray(path)
    ctx = scheme.prepare(orig, params)
    wm = scheme.embed(ctx, bits, params)
    return wpsnr(orig, wm), float(np.mean(wm != orig)), similarity(to_bipolar(bits), scheme.extract(ctx, wm, params))


def main():
    bits = load_mark()
    paths = load_split()[0][:10]
    print("%-9s %-40s | %7s %7s %7s | %7s | %6s" % ("schema", "parametri", "medio", "min", "p10", "pixel", "sim"))
    for scheme_name, variants in GRID:
        for tweaks in variants:
            params = get_scheme(scheme_name).params(tweaks)
            rows = pmap(one_image, [(scheme_name, params, p, bits) for p in paths])
            w, changed, sims = (np.array(c) for c in zip(*rows))
            print("%-9s %-40s | %7.2f %7.2f %7.2f | %6.1f%% | %6.3f" % (
                scheme_name, tweaks, w.mean(), w.min(), np.percentile(w, 10), 100 * changed.mean(), sims.mean()))
            append_result("e0", {"scheme": scheme_name, "params": params, "wpsnr_mean": w.mean(),
                                 "wpsnr_min": w.min(), "changed": changed.mean(), "sim": sims.mean()})


if __name__ == "__main__":
    main()
