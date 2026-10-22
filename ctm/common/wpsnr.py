"""
C3: WPSNR del laboratorio.

Versione veloce (FFT) usata dagli esperimenti del package. Il file di detection
consegnato usa invece il codice LAB-3 letterale di wpsnr_lab.py (stesso risultato).
"""
import numpy as np
from scipy.signal import convolve2d

# >>> inline
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
    WPSNR del laboratorio. La riga originale era
        convolve2d(A - B, _CSF_FILTER, mode="same", boundary="wrap")
    (circa 3 s); qui la stessa convoluzione circolare è fatta con la FFT:
    risultato identico, circa 0.1 s.
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
    return float(20 * np.log10(1.0 / np.sqrt(wmse)))
# <<< inline


def wpsnr_lab(image_a, image_b):
    """Versione letterale del LAB-3 (convolve2d), usata solo nei test."""
    A = image_a.astype(np.float64) / 255.0
    B = image_b.astype(np.float64) / 255.0
    if np.array_equal(A, B):
        return 9999999.0
    weighted_error = convolve2d(A - B, _CSF_FILTER, mode="same", boundary="wrap")
    return float(20 * np.log10(1.0 / np.sqrt(np.mean(weighted_error ** 2))))
