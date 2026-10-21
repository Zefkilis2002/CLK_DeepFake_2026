"""
Common functions used by roc.py, attacks.py, demo.py and check_rules.py.

Contains:
  - image reading and rounding to uint8 (LAB-1/LAB-2)
  - WPSNR (LAB-3 code, CSF filter computed only once)
  - similarity and watermark utilities (LAB-3)
  - the 6 attacks allowed by the competition (LAB-3 ROC + LAB-4)
  - random_attack for the ROC (LAB-3 ROC, Example 2)

NB: the detection file does NOT import this module, because the competition
requires a single file: there WPSNR and similarity are copied identically.
"""
import os

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import convolve2d

# Project folders (everything relative to this file, not to the current directory)
CTM_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CTM_DIR)
IMAGES_DIR = os.path.join(ROOT_DIR, "images")
MARK_PATH = os.path.join(ROOT_DIR, "clk.npy")
RESULTS_DIR = os.path.join(CTM_DIR, "results")

MARK_SIZE = 1024


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
def read_gray(path):
    """Reads a grayscale image with OpenCV (uint8, 512x512)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def to_uint8(x):
    """The project's single rounding rule: clip to [0,255], then rint."""
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def list_images(folder=IMAGES_DIR):
    """Sorted paths of all the .bmp files in a folder."""
    names = sorted(n for n in os.listdir(folder) if n.lower().endswith(".bmp"))
    return [os.path.join(folder, n) for n in names]


# ---------------------------------------------------------------------------
# WPSNR (LAB-3, identical to the competition one)
# ---------------------------------------------------------------------------
def _csffun(u, v):
    f = np.sqrt(u ** 2 + v ** 2)
    w = 2 * np.pi * f / 60
    sigma = 2
    Sw = 1.5 * np.exp(-sigma ** 2 * w ** 2 / 2) - np.exp(-2 * sigma ** 2 * w ** 2 / 2)
    sita = np.arctan2(v, u)
    bita = 8
    f0 = 11.13
    w0 = 2 * np.pi * f0 / 60
    exp_term = np.exp(bita * (w - w0))
    Ow = (1 + exp_term * (np.cos(2 * sita)) ** 4) / (1 + exp_term)
    return Sw * Ow


def _get_csf_filter():
    freq_range = np.arange(-20, 21, 1)
    u, v = np.meshgrid(freq_range, freq_range, indexing="xy")
    Fmat = _csffun(u, v)
    return np.real(np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(Fmat))))


CSF_FILTER = _get_csf_filter()   # computed only once


def wpsnr_lab(image_a, image_b):
    """WPSNR exactly as in LAB-3 (convolve2d). Slow: about 3 s per call."""
    A = image_a.astype(np.float64) / 255.0
    B = image_b.astype(np.float64) / 255.0
    if np.array_equal(A, B):
        return 9999999.0
    weighted_error = convolve2d(A - B, CSF_FILTER, mode="same", boundary="wrap")
    wmse = np.mean(weighted_error ** 2)
    return 20 * np.log10(1.0 / np.sqrt(wmse))


def wpsnr(image_a, image_b):
    """
    Same WPSNR as LAB-3, but the convolution with boundary='wrap' is done with the
    FFT (circular convolution, LAB-2): identical result (difference ~1e-14 dB,
    verified in check_rules.py) and about 30 times faster (~0.1 s).
    """
    A = image_a.astype(np.float64) / 255.0
    B = image_b.astype(np.float64) / 255.0
    if np.array_equal(A, B):
        return 9999999.0
    error = A - B
    kernel = np.zeros_like(error)
    kernel[:41, :41] = CSF_FILTER                         # 41x41 filter ...
    kernel = np.roll(kernel, (-20, -20), axis=(0, 1))     # ... centered at the origin
    weighted_error = np.real(np.fft.ifft2(np.fft.fft2(error) * np.fft.fft2(kernel)))
    wmse = np.mean(weighted_error ** 2)
    return 20 * np.log10(1.0 / np.sqrt(wmse))


# ---------------------------------------------------------------------------
# Watermark and similarity (LAB-3)
# ---------------------------------------------------------------------------
def load_mark(path=MARK_PATH):
    """The competition watermark: 1024 bits 0/1."""
    return np.load(path).astype(np.int64)


def random_mark(rng):
    """Random 0/1 watermark of the same length (for H0 of the ROC)."""
    return rng.integers(0, 2, MARK_SIZE)


def to_bipolar(bits):
    """0/1 -> -1/+1, so that the similarity between random marks is about 0."""
    return 2.0 * np.asarray(bits, dtype=float) - 1.0


def similarity(a, b):
    """Cosine similarity (LAB-3, EXERCISEs_SOL). It is 0 if a vector is null."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


# ---------------------------------------------------------------------------
# The 6 allowed attacks. All of them: uint8 512x512 as input and output.
# ---------------------------------------------------------------------------
def awgn(img, sigma, seed=0):
    """Additive Gaussian noise with standard deviation sigma."""
    noise = np.random.default_rng(seed).normal(0, sigma, img.shape)
    return to_uint8(img.astype(float) + noise)


def blur(img, sigma):
    """Gaussian low-pass filter."""
    return to_uint8(gaussian_filter(img.astype(float), sigma))


def sharpen(img, alpha, sigma=1.0):
    """Unsharp masking: img + alpha * (img - blur(img))."""
    x = img.astype(float)
    return to_uint8(x + alpha * (x - gaussian_filter(x, sigma)))


def median(img, k):
    """k x k median filter (k odd)."""
    return median_filter(img, size=int(k)).astype(np.uint8)


def jpeg(img, qf):
    """In-memory JPEG compression (cv2.imencode, no temporary file)."""
    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(qf)])
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)


def resize(img, scale):
    """Shrinks by 'scale' and brings it back to the original size."""
    h, w = img.shape
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


ATTACKS = {
    "awgn": awgn,
    "blur": blur,
    "sharp": sharpen,
    "median": median,
    "jpeg": jpeg,
    "resize": resize,
}


def random_attack(img, rng):
    """
    Random attack for the ROC (idea from LAB-3, Example 2), with parameters
    drawn from the plan's ranges. One or two operations in sequence.
    Returns the attacked image and a text description.
    """
    out = img
    steps = []
    for _ in range(rng.integers(1, 3)):          # 1 or 2 attacks
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
