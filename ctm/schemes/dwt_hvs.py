"""
M3: DWT db6 a 4 livelli con maschera percettiva pixel-wise (LAB-4, Barni-Bartolini-Piva).

Parametri:
  alpha        : forza
  exponent     : esponente della maschera (0.2 nel LAB-4; il piano prova 0.1-0.5)
  permute_seed : None, oppure un seed per mescolare le posizioni delle 192 copie
"""
import numpy as np
import pywt
from scipy.ndimage import uniform_filter

from ctm.schemes.base import Scheme

MARK_SIZE = 1024
WAVELET, LEVELS = "db6", 4


def wavedec4(img):
    return pywt.wavedec2(img.astype(float), WAVELET, mode="periodization", level=LEVELS)


def upsample_to(array, shape):
    ry = int(np.ceil(shape[0] / array.shape[0]))
    rx = int(np.ceil(shape[1] / array.shape[1]))
    return np.repeat(np.repeat(array, ry, axis=0), rx, axis=1)[:shape[0], :shape[1]]


def build_hvs_masks(coeffs, exponent):
    finest = coeffs[-1][0].shape
    activity = np.zeros(finest)
    for k in range(LEVELS):
        energy = sum(4.0 * uniform_filter(b * b, size=2, mode="reflect") for b in coeffs[-1 - k])
        activity += upsample_to(energy / (16.0 ** k), finest)
    lowpass = coeffs[0] / (2.0 ** LEVELS)
    lum = np.clip(lowpass / 255.0, 0.0, 1.0)
    lum_factor = np.where(lum < 0.5, 1.0 - lum, lum)
    mean = uniform_filter(lowpass, size=2, mode="reflect")
    var = np.maximum(uniform_filter(lowpass * lowpass, size=2, mode="reflect") - mean * mean, 0.0)
    common = upsample_to(lum_factor, finest) * np.power(activity * upsample_to(var, finest) + 1e-12,
                                                        exponent) / 2.0
    return np.concatenate([common.ravel(), common.ravel(), np.sqrt(2.0) * common.ravel()])


class DwtHvs(Scheme):
    name = "dwt_hvs"
    default_params = {"alpha": 0.75, "exponent": 0.2, "permute_seed": None}

    def prepare(self, original, params):
        coeffs = wavedec4(original)
        mask = build_hvs_masks(coeffs, params["exponent"])
        order = np.arange(mask.size)
        if params.get("permute_seed") is not None:
            order = np.random.default_rng(params["permute_seed"]).permutation(mask.size)
        return {"original": original, "coeffs": coeffs, "mask": mask, "order": order}

    def embed(self, ctx, bits, params):
        coeffs = [c if i == 0 else tuple(b.copy() for b in c) for i, c in enumerate(ctx["coeffs"])]
        shape = coeffs[-1][0].shape
        x = 2.0 * np.asarray(bits, dtype=float) - 1.0
        signal = np.empty(ctx["mask"].size)
        signal[ctx["order"]] = np.resize(x, ctx["mask"].size)        # 192 copie
        delta = params["alpha"] * ctx["mask"] * signal
        pieces = [p.reshape(shape) for p in np.split(delta, 3)]
        coeffs[-1] = tuple(b + p for b, p in zip(coeffs[-1], pieces))
        result = pywt.waverec2(coeffs, WAVELET, mode="periodization")
        return np.clip(np.rint(result), 0, 255).astype(np.uint8)

    def extract(self, ctx, image, params):
        """Media pesata con la maschera delle 192 copie di ogni bit (minimi quadrati)."""
        c1 = wavedec4(image)
        diff = np.concatenate([(a - o).ravel() for a, o in zip(c1[-1], ctx["coeffs"][-1])])
        m = ctx["mask"]
        num = (diff * m)[ctx["order"]].reshape(-1, MARK_SIZE).sum(axis=0)
        den = (params["alpha"] * m * m)[ctx["order"]].reshape(-1, MARK_SIZE).sum(axis=0)
        return num / den
