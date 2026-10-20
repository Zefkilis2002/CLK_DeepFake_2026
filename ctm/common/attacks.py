"""C5: i 6 attacchi ammessi. Tutti uint8 in ingresso e in uscita, funzioni pure."""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

from ctm.common.imgio import to_uint8


def awgn(img, sigma, seed=0):
    noise = np.random.default_rng(seed).normal(0, sigma, img.shape)
    return to_uint8(img.astype(float) + noise)


def blur(img, sigma):
    return to_uint8(gaussian_filter(img.astype(float), sigma))


def sharpen(img, alpha, sigma=1.0):
    x = img.astype(float)
    return to_uint8(x + alpha * (x - gaussian_filter(x, sigma)))


def median(img, k):
    return median_filter(img, size=int(k)).astype(np.uint8)


def jpeg(img, qf):
    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(qf)])
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)


def resize(img, scale):
    h, w = img.shape
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


ATTACKS = {"awgn": awgn, "blur": blur, "sharp": sharpen,
           "median": median, "jpeg": jpeg, "resize": resize}


def random_attack(img, rng):
    """
    Famiglia di attacchi casuali della ROC (piano C8): 1 o 2 attacchi con
    parametri negli intervalli del piano. Restituisce (immagine, descrizione).
    """
    out, steps = img, []
    for _ in range(rng.integers(1, 3)):
        name = rng.choice(list(ATTACKS))
        if name == "awgn":
            p = rng.uniform(1, 20)
            out = awgn(out, p, seed=int(rng.integers(1_000_000)))
        elif name == "blur":
            p = rng.uniform(0.3, 2.0)
            out = blur(out, p)
        elif name == "sharp":
            p = rng.uniform(0.2, 2.0)
            out = sharpen(out, p, sigma=rng.choice([0.5, 1.0]))
        elif name == "median":
            p = rng.choice([3, 5])
            out = median(out, p)
        elif name == "jpeg":
            p = rng.integers(20, 96)
            out = jpeg(out, p)
        else:
            p = rng.uniform(0.5, 0.95)
            out = resize(out, p)
        steps.append("%s(%.2f)" % (name, p))
    return out, " + ".join(steps)


# suite fissa di attacchi moderati per lo screening dell'halving (piano C9, S1)
SCREEN_SUITE = [("jpeg", 50), ("jpeg", 30), ("blur", 1.0), ("median", 3),
                ("awgn", 8), ("resize", 0.6), ("sharp", 1.5), ("median", 5)]
