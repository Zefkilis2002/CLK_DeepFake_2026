"""
The group's detection (single file, as the competition requires).

Scheme: additive Spread Spectrum in DCT (LAB-3), non-blind detection.
  - W_extracted = mark extracted from the watermarked image (using the original)
  - W_attacked  = mark extracted from the attacked image    (using the original)
  - present if sim(W_extracted, W_attacked) >= TAU
The watermark file is NOT read: the mark is extracted from the images.
TAU was computed only once with roc.py and written here by hand.
No prints, no windows, no relative paths.
"""
import cv2
import numpy as np
from scipy.fft import dct

# --- parameters: must match embedding.py ------------------------------------
ALPHA = 5.0
MARK_SIZE = 1024
TAU = 0.1218         # threshold from the ROC (ALPHA = 5): TPR 0.87, FPR 0 on 1119 H0 cases


# ---------------------------------------------------------------------------
# WPSNR (lab code)
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


_CSF_FILTER = _get_csf_filter()


def wpsnr(image_a, image_b):
    """
    Lab WPSNR. The original line was
        convolve2d(A - B, _CSF_FILTER, mode="same", boundary="wrap")
    and takes about 3 s (too close to the 5 s limit). Here the same
    circular convolution is done with the FFT: identical result, ~0.1 s.
    """
    A = image_a.astype(np.float64) / 255.0
    B = image_b.astype(np.float64) / 255.0
    if np.array_equal(A, B):
        return 9999999.0
    error = A - B
    kernel = np.zeros_like(error)
    kernel[:41, :41] = _CSF_FILTER
    kernel = np.roll(kernel, (-20, -20), axis=(0, 1))
    weighted_error = np.real(np.fft.ifft2(np.fft.fft2(error) * np.fft.fft2(kernel)))
    wmse = np.mean(weighted_error ** 2)
    return 20 * np.log10(1.0 / np.sqrt(wmse))


# ---------------------------------------------------------------------------
# Extraction and similarity (LAB-3: detection with v='additive' and similarity)
# ---------------------------------------------------------------------------
def extract_dct_ss(original, image, alpha=ALPHA):
    """Estimate of x (+-1) at the 1024 locations chosen on the ORIGINAL."""
    c0 = np.abs(dct(dct(original.astype(float), axis=0, norm="ortho"), axis=1, norm="ortho"))
    c1 = np.abs(dct(dct(image.astype(float), axis=0, norm="ortho"), axis=1, norm="ortho"))
    locations = np.argsort(-c0, axis=None)[1:MARK_SIZE + 1]
    base = c0.ravel()[locations]
    changed = c1.ravel()[locations]
    return (changed - base) / alpha


def similarity(a, b):
    """Cosine similarity; 0 if either vector is null."""
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def detection(input1, input2, input3):
    """
    input1: path of the original
    input2: path of the watermarked image
    input3: path of the attacked image
    output1: 1 if the mark is present in the attacked image, otherwise 0
    output2: WPSNR(watermarked, attacked)
    """
    original = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    watermarked = cv2.imread(input2, cv2.IMREAD_GRAYSCALE)
    attacked = cv2.imread(input3, cv2.IMREAD_GRAYSCALE)
    if attacked.shape != original.shape:          # safety: same size
        attacked = cv2.resize(attacked, (original.shape[1], original.shape[0]))

    # detection is ALWAYS performed, whatever the WPSNR
    w_extracted = extract_dct_ss(original, watermarked)
    w_attacked = extract_dct_ss(original, attacked)
    sim = similarity(w_extracted, w_attacked)

    output1 = 1 if sim >= TAU else 0
    output2 = float(wpsnr(watermarked, attacked))
    return output1, output2
