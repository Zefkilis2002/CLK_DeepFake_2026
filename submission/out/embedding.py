"""
Embedding del gruppo groupname (file GENERATO da submission/build_submission.py,
preset P66: non modificare a mano, rigenerare dal preset).

Schema: Spread Spectrum in DCT (LAB-3), modo additive, sui 1024 coefficienti
DCT più grandi (DC escluso); con N > 1024 ogni bit è ripetuto N/1024 volte.
"""
import cv2
import numpy as np
from scipy.fft import dct, idct

ALPHA = 4.360352
MODE = "additive"
N = 1024

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


def embedding(input1, input2):
    """input1: percorso originale, input2: percorso watermark .npy -> immagine marchiata uint8."""
    original = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    bits = np.load(input2)
    return embed_dct_ss(original, bits, ALPHA, MODE, N)
