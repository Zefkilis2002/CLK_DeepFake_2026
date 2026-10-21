"""WPSNR code provided for the competition (same as the labs)."""
import numpy as np
from scipy.signal import convolve2d

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
