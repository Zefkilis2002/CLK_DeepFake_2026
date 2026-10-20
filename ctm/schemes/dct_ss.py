"""
M2: Spread Spectrum in DCT (LAB-3), il nostro schema.

Parametri:
  alpha : forza
  mode  : "additive"       |C| + alpha*x      (scelta attuale)
          "multiplicative" |C| (1 + alpha*x)  (versione di ss_embed del LAB-3)
  n     : numero di coefficienti usati (1024, 2048, 4096): con n > 1024 ogni bit
          è ripetuto n/1024 volte e in estrazione si fa la media delle copie.

Le funzioni tra ">>> inline" e "<<< inline" vengono copiate nei file consegnati.
"""
import numpy as np
from scipy.fft import dct, idct

from ctm.schemes.base import Scheme

# >>> inline
MARK_SIZE = 1024


def dct2(img):
    return dct(dct(img, axis=0, norm="ortho"), axis=1, norm="ortho")


def idct2(coeff):
    return idct(idct(coeff, axis=1, norm="ortho"), axis=0, norm="ortho")


def select_locations(magnitudes, n):
    """Posizioni degli n coefficienti più grandi, saltando il primo (DC)."""
    return np.argsort(-magnitudes, axis=None)[1:n + 1]


def embed_dct_ss(original, bits, alpha, mode, n):
    x = np.resize(2.0 * np.asarray(bits, dtype=float) - 1.0, n)   # +-1, ripetuto se n > 1024
    coeff = dct2(original.astype(float))
    signs, magnitudes = np.sign(coeff), np.abs(coeff)
    locations = select_locations(magnitudes, n)
    edited = magnitudes.ravel().copy()
    if mode == "additive":
        edited[locations] += alpha * x
    else:
        edited[locations] *= 1.0 + alpha * x
    result = idct2(edited.reshape(coeff.shape) * signs)
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def extract_dct_ss(original, image, alpha, mode, n):
    c0 = np.abs(dct2(original.astype(float)))
    c1 = np.abs(dct2(image.astype(float)))
    locations = select_locations(c0, n)
    base, changed = c0.ravel()[locations], c1.ravel()[locations]
    if mode == "additive":
        estimate = (changed - base) / alpha
    else:
        estimate = (changed - base) / (alpha * base)
    return estimate.reshape(-1, MARK_SIZE).mean(axis=0)            # media delle copie
# <<< inline


class DctSS(Scheme):
    name = "dct_ss"
    default_params = {"alpha": 5.0, "mode": "additive", "n": 1024}

    def embed(self, ctx, bits, params):
        return embed_dct_ss(ctx["original"], bits, params["alpha"], params["mode"], params["n"])

    def extract(self, ctx, image, params):
        return extract_dct_ss(ctx["original"], image, params["alpha"], params["mode"], params["n"])
