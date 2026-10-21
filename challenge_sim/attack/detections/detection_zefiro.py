"""Detection of group zefiro."""
import cv2
import numpy as np
from scipy.fft import dct

ALPHA = 5.68
MODE = 'add'
N = 1024
TAU = 0.0938

def _csf():
    f = np.arange(-20, 21)
    u, v = np.meshgrid(f, f, indexing="xy")
    w = 2 * np.pi * np.sqrt(u * u + v * v) / 60
    sw = 1.5 * np.exp(-2 * w ** 2) - np.exp(-4 * w ** 2)
    e = np.exp(8 * (w - 2 * np.pi * 11.13 / 60))
    ow = (1 + e * np.cos(2 * np.arctan2(v, u)) ** 4) / (1 + e)
    return np.real(np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(sw * ow))))


_K = _csf()


def wpsnr(a, b):
    err = a.astype(float) / 255.0 - b.astype(float) / 255.0
    if not np.any(err):
        return 9999999.0
    k = np.zeros_like(err)
    k[:41, :41] = _K
    k = np.roll(k, (-20, -20), axis=(0, 1))
    we = np.real(np.fft.ifft2(np.fft.fft2(err) * np.fft.fft2(k)))
    return float(20 * np.log10(1.0 / np.sqrt(np.mean(we ** 2))))


def extract(original, image):
    c0 = np.abs(dct(dct(original.astype(float), axis=0, norm="ortho"), axis=1, norm="ortho"))
    c1 = np.abs(dct(dct(image.astype(float), axis=0, norm="ortho"), axis=1, norm="ortho"))
    loc = np.argsort(-c0, axis=None)[1:N + 1]
    if MODE == "add":
        w = (c1.ravel()[loc] - c0.ravel()[loc]) / ALPHA
    else:
        w = (c1.ravel()[loc] - c0.ravel()[loc]) / (ALPHA * c0.ravel()[loc])
    return w.reshape(-1, 1024).mean(axis=0)


def similarity(a, b):
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if n == 0 else float(np.dot(a, b) / n)


def detection(input1, input2, input3):
    original = cv2.imread(input1, 0)
    watermarked = cv2.imread(input2, 0)
    attacked = cv2.imread(input3, 0)
    sim = similarity(extract(original, watermarked), extract(original, attacked))
    return (1 if sim >= TAU else 0), wpsnr(watermarked, attacked)
