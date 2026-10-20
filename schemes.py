"""
I tre schemi di watermarking dei laboratori, con la stessa interfaccia:

    embed(original, bits, param)    -> immagine marchiata uint8
    extract(original, image, param) -> vettore di 1024 valori (stima di x = +-1)

  M1 bitplane : LSB / bit-plane del LAB-3 (param = indice del bit-plane)
  M2 dct_ss   : Spread Spectrum additivo in DCT del LAB-3 (param = alpha)  <- il nostro
  M2 dct_mult : Spread Spectrum moltiplicativo in DCT del LAB-3 (param = alpha)
  M3 dwt_hvs  : DWT db6 a 4 livelli con maschera HVS del LAB-4 (param = alpha)

Servono per l'esperimento E0 (confronto qualità) e come "difensori locali"
su cui provare la nostra strategia di attacco (bench.py).
"""
import numpy as np
import pywt
from scipy.ndimage import uniform_filter

from embedding import dct2, embed_dct_ss, idct2
from detection_groupname import extract_dct_ss

MARK_SIZE = 1024


# ---------------------------------------------------------------------------
# M1: bit-plane (LAB-3, EXERCISEs_SOL esercizio 1)
# ---------------------------------------------------------------------------
def embed_bitplane(original, bits, plane=0):
    """Scrive i 1024 bit, ripetuti su tutta l'immagine, nel bit-plane 'plane'."""
    tiled = np.resize(np.asarray(bits, dtype=np.uint8), original.size)   # 256 copie
    clear_mask = np.uint8(~(1 << plane) & 0xFF)
    flat = original.ravel()
    return ((flat & clear_mask) | (tiled << plane)).reshape(original.shape).astype(np.uint8)


def extract_bitplane(original, image, plane=0):
    """Legge il bit-plane e fa la media delle 256 copie; restituisce valori in [-1, 1]."""
    plane_bits = (image.ravel() >> plane) & 1
    copies = plane_bits.reshape(-1, MARK_SIZE)          # 256 righe x 1024 bit
    return 2.0 * copies.mean(axis=0) - 1.0


# ---------------------------------------------------------------------------
# M2 moltiplicativo (LAB-3, ss_embed / ss_detect): |C|(1 + alpha*x)
# Era la nostra prima scelta; ora serve come difensore finto nel banco di prova.
# ---------------------------------------------------------------------------
def embed_dct_mult(original, bits, alpha=0.007):
    x = 2.0 * np.asarray(bits, dtype=float) - 1.0
    coeff = dct2(original.astype(float))
    magnitudes = np.abs(coeff)
    locations = np.argsort(-magnitudes, axis=None)[1:MARK_SIZE + 1]
    edited = magnitudes.ravel().copy()
    edited[locations] *= 1.0 + alpha * x
    result = idct2(edited.reshape(coeff.shape) * np.sign(coeff))
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def extract_dct_mult(original, image, alpha=0.007):
    c0 = np.abs(dct2(original.astype(float)))
    c1 = np.abs(dct2(image.astype(float)))
    locations = np.argsort(-c0, axis=None)[1:MARK_SIZE + 1]
    base = c0.ravel()[locations]
    return (c1.ravel()[locations] - base) / (alpha * base)


# ---------------------------------------------------------------------------
# M3: DWT con maschera percettiva (LAB-4, Barni-Bartolini-Piva)
# ---------------------------------------------------------------------------
WAVELET = "db6"
LEVELS = 4


def wavedec4(img):
    return pywt.wavedec2(img.astype(float), WAVELET, mode="periodization", level=LEVELS)


def upsample_to(array, shape):
    ry = int(np.ceil(shape[0] / array.shape[0]))
    rx = int(np.ceil(shape[1] / array.shape[1]))
    return np.repeat(np.repeat(array, ry, axis=0), rx, axis=1)[:shape[0], :shape[1]]


def build_hvs_masks(coeffs):
    """Maschera pixel-wise del LAB-4 (luminanza x attività x varianza)."""
    finest_shape = coeffs[-1][0].shape
    activity = np.zeros(finest_shape)
    for k in range(LEVELS):
        energy = sum(4.0 * uniform_filter(b * b, size=2, mode="reflect") for b in coeffs[-1 - k])
        activity += upsample_to(energy / (16.0 ** k), finest_shape)
    lowpass = coeffs[0] / (2.0 ** LEVELS)
    lum = np.clip(lowpass / 255.0, 0.0, 1.0)
    lum_factor = np.where(lum < 0.5, 1.0 - lum, lum)
    mean = uniform_filter(lowpass, size=2, mode="reflect")
    var = np.maximum(uniform_filter(lowpass * lowpass, size=2, mode="reflect") - mean * mean, 0.0)
    xi = activity * upsample_to(var, finest_shape)
    common = upsample_to(lum_factor, finest_shape) * np.power(xi + 1e-12, 0.2) / 2.0
    return common, common, np.sqrt(2.0) * common


def embed_dwt_hvs(original, bits, alpha=0.75):
    """I 1024 bit (+-1) ripetuti 192 volte coprono H1, V1, D1 (3 x 256 x 256)."""
    coeffs = wavedec4(original)
    masks = build_hvs_masks(coeffs)
    shape = coeffs[-1][0].shape
    x = 2.0 * np.asarray(bits, dtype=float) - 1.0
    tiled = np.resize(x, 3 * shape[0] * shape[1])
    pieces = [p.reshape(shape) for p in np.split(tiled, 3)]
    coeffs[-1] = tuple(b + alpha * m * p for b, m, p in zip(coeffs[-1], masks, pieces))
    result = pywt.waverec2(coeffs, WAVELET, mode="periodization")
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def extract_dwt_hvs(original, image, alpha=0.75):
    """
    Differenza dei coefficienti H1/V1/D1 rispetto all'originale, poi per ogni bit
    una media pesata con la maschera sulle sue 192 copie (minimi quadrati).
    """
    c0 = wavedec4(original)
    masks = build_hvs_masks(c0)
    c1 = wavedec4(image)
    diff = np.concatenate([(a - o).ravel() for a, o in zip(c1[-1], c0[-1])])
    m = np.concatenate([mk.ravel() for mk in masks])
    num = (diff * m).reshape(-1, MARK_SIZE).sum(axis=0)
    den = (alpha * m * m).reshape(-1, MARK_SIZE).sum(axis=0)
    return num / den


# ---------------------------------------------------------------------------
# Tabella degli schemi: nome -> (embed, extract)
# ---------------------------------------------------------------------------
SCHEMES = {
    "bitplane": (embed_bitplane, extract_bitplane),
    "dct_ss": (embed_dct_ss, extract_dct_ss),
    "dct_mult": (embed_dct_mult, extract_dct_mult),
    "dwt_hvs": (embed_dwt_hvs, extract_dwt_hvs),
}
