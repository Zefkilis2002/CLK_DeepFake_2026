"""Detection of group orca."""
import cv2
import numpy as np
import pywt
from scipy.ndimage import uniform_filter
from scipy.signal import convolve2d

ALPHA = 1.62
TAU = 0.0922

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
    Sa = Sw * Ow
    return Sa


def _csfmat():
    min_f, max_f, step_f = -20, 20, 1
    freq_range = np.arange(min_f, max_f + step_f, step_f)
    u, v = np.meshgrid(freq_range, freq_range, indexing='xy')
    Fmat = _csffun(u, v)
    return Fmat


def _get_csf_filter():
    Fmat = _csfmat()
    filter_coeffs = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(Fmat)))
    return np.real(filter_coeffs)


def wpsnr(image_a, image_b):
    if image_a.shape != image_b.shape:
        raise ValueError("Input images must have the same dimensions.")
    if image_a.dtype != np.float64 and image_a.dtype != np.float32:
        A = image_a.astype(np.float64) / 255.0
    else:
        A = image_a.copy()
    if image_b.dtype != np.float64 and image_b.dtype != np.float32:
        B = image_b.astype(np.float64) / 255.0
    else:
        B = image_b.copy()
    if A.max() > 1.0 or A.min() < 0.0 or B.max() > 1.0 or B.min() < 0.0:
        raise ValueError("Input image values must be in the interval [0, 1] for floats or [0, 255] for integers.")
    if np.array_equal(A, B):
        return 9999999.0
    error_image = A - B
    csf_filter = _get_csf_filter()
    weighted_error = convolve2d(error_image, csf_filter, mode='same', boundary='wrap')
    wmse = np.mean(weighted_error**2)
    if wmse == 0:
        return 9999999.0
    decibels = 20 * np.log10(1.0 / np.sqrt(wmse))
    return decibels


def _up(a, shape):
    r = int(np.ceil(shape[0] / a.shape[0]))
    return np.repeat(np.repeat(a, r, axis=0), r, axis=1)[:shape[0], :shape[1]]


def _mask(coeffs):
    fine = coeffs[-1][0].shape
    act = np.zeros(fine)
    for k in range(4):
        act += _up(sum(4.0 * uniform_filter(b * b, size=2, mode="reflect") for b in coeffs[-1 - k]) / 16.0 ** k, fine)
    low = coeffs[0] / 16.0
    lum = np.clip(low / 255.0, 0, 1)
    lf = np.where(lum < 0.5, 1 - lum, lum)
    m = uniform_filter(low, size=2, mode="reflect")
    var = np.maximum(uniform_filter(low * low, size=2, mode="reflect") - m * m, 0)
    c = _up(lf, fine) * np.power(act * _up(var, fine) + 1e-12, 0.2) / 2.0
    return np.concatenate([c.ravel(), c.ravel(), np.sqrt(2.0) * c.ravel()])


def extract(original, image):
    c0 = pywt.wavedec2(original.astype(float), "db6", mode="periodization", level=4)
    c1 = pywt.wavedec2(image.astype(float), "db6", mode="periodization", level=4)
    d = np.concatenate([(a - o).ravel() for a, o in zip(c1[-1], c0[-1])])
    m = _mask(c0)
    return (d * m).reshape(-1, 1024).sum(axis=0) / (ALPHA * (m * m).reshape(-1, 1024).sum(axis=0))


def similarity(a, b):
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if n == 0 else float(np.dot(a, b) / n)


def detection(input1, input2, input3):
    original = cv2.imread(input1, 0)
    watermarked = cv2.imread(input2, 0)
    attacked = cv2.imread(input3, 0)
    sim = similarity(extract(original, watermarked), extract(original, attacked))
    return (1 if sim >= TAU else 0), wpsnr(watermarked, attacked)
