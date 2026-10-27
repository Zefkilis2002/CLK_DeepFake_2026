"""
C9: taratura di alpha per bisezione.

Con schema e altri parametri fissi, il WPSNR (sulle immagini uint8) scende
quando alpha cresce. Invece di provare una griglia di alpha, si fissa un
WPSNR obiettivo e si cerca l'alpha più grande per cui il 10° percentile del
WPSNR sulle immagini di sviluppo resta sopra l'obiettivo.
"""
import numpy as np

from ctm.common.imgio import read_gray
from ctm.common.wpsnr import wpsnr
from ctm.schemes import get_scheme

ALPHA_RANGE = {"dct_ss:additive": (0.5, 30.0), "dct_ss:multiplicative": (0.001, 0.05),
               "dwt_hvs": (0.05, 5.0)}


def alpha_range(scheme_name, params):
    key = scheme_name + (":" + params["mode"] if scheme_name == "dct_ss" else "")
    return ALPHA_RANGE[key]


def wpsnr_percentile(scheme, params, originals, bits, q=10):
    w = [wpsnr(o, scheme.embed_image(o, bits, params)) for o in originals]
    return float(np.percentile(w, q))


def calibrate_alpha(scheme_name, params, target, paths, bits, iters=10, tol=0.2):
    """Restituisce (alpha, p10 WPSNR) oppure (None, p10) se l'obiettivo non è raggiungibile."""
    scheme = get_scheme(scheme_name)
    originals = [read_gray(p) for p in paths]
    lo, hi = alpha_range(scheme_name, params)
    p_lo = wpsnr_percentile(scheme, dict(params, alpha=lo), originals, bits)
    if p_lo < target:
        return None, p_lo                      # anche l'alpha minimo è troppo visibile
    best = (lo, p_lo)
    for _ in range(iters):
        mid = (lo + hi) / 2
        p = wpsnr_percentile(scheme, dict(params, alpha=mid), originals, bits)
        if p >= target:
            lo, best = mid, (mid, p)
            if p - target < tol:
                break
        else:
            hi = mid
    return round(best[0], 6), best[1]
