"""
Funzioni comuni usate da ROC, attacchi, benchmark e test.

Contiene:
  - lettura delle immagini e arrotondamento a uint8 (LAB-1/LAB-2)
  - WPSNR (codice del LAB-3, filtro CSF calcolato una volta sola)
  - similarità e utilità per il watermark (LAB-3)
  - i 6 attacchi ammessi dalla gara (LAB-3 ROC + LAB-4)
  - random_attack per la ROC (LAB-3 ROC, Example 2)

NB: il file di detection NON importa questo modulo, perché la gara vuole un
file unico: lì WPSNR e similarità sono ricopiati identici.
"""
import os

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import convolve2d

# Cartelle del progetto (tutto relativo a questo file, non alla cartella corrente)
CTM_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CTM_DIR)
IMAGES_DIR = os.path.join(ROOT_DIR, "images")
MARK_PATH = os.path.join(ROOT_DIR, "clk.npy")
RESULTS_DIR = os.path.join(CTM_DIR, "results")

MARK_SIZE = 1024


# ---------------------------------------------------------------------------
# Immagini
# ---------------------------------------------------------------------------
def read_gray(path):
    """Legge un'immagine in scala di grigi con OpenCV (uint8, 512x512)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def to_uint8(x):
    """Unica regola di arrotondamento del progetto: clip in [0,255], poi rint."""
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def list_images(folder=IMAGES_DIR):
    """Percorsi ordinati di tutte le .bmp di una cartella."""
    names = sorted(n for n in os.listdir(folder) if n.lower().endswith(".bmp"))
    return [os.path.join(folder, n) for n in names]


# ---------------------------------------------------------------------------
# WPSNR (LAB-3, identico a quello della gara)
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


CSF_FILTER = _get_csf_filter()   # calcolato una volta sola


def wpsnr_lab(image_a, image_b):
    """WPSNR esattamente come nel LAB-3 (convolve2d). Lento: circa 3 s a chiamata."""
    A = image_a.astype(np.float64) / 255.0
    B = image_b.astype(np.float64) / 255.0
    if np.array_equal(A, B):
        return 9999999.0
    weighted_error = convolve2d(A - B, CSF_FILTER, mode="same", boundary="wrap")
    wmse = np.mean(weighted_error ** 2)
    return 20 * np.log10(1.0 / np.sqrt(wmse))


def wpsnr(image_a, image_b):
    """
    Stesso WPSNR del LAB-3, ma la convoluzione con boundary='wrap' è fatta con la
    FFT (convoluzione circolare, LAB-2): risultato identico (differenza ~1e-14 dB,
    verificata in check_rules.py) e circa 30 volte più veloce (~0.1 s).
    """
    A = image_a.astype(np.float64) / 255.0
    B = image_b.astype(np.float64) / 255.0
    if np.array_equal(A, B):
        return 9999999.0
    error = A - B
    kernel = np.zeros_like(error)
    kernel[:41, :41] = CSF_FILTER                         # filtro 41x41 ...
    kernel = np.roll(kernel, (-20, -20), axis=(0, 1))     # ... centrato nell'origine
    weighted_error = np.real(np.fft.ifft2(np.fft.fft2(error) * np.fft.fft2(kernel)))
    wmse = np.mean(weighted_error ** 2)
    return 20 * np.log10(1.0 / np.sqrt(wmse))


# ---------------------------------------------------------------------------
# Watermark e similarità (LAB-3)
# ---------------------------------------------------------------------------
def load_mark(path=MARK_PATH):
    """Il watermark della gara: 1024 bit 0/1."""
    return np.load(path).astype(np.int64)


def random_mark(rng):
    """Watermark casuale 0/1 della stessa lunghezza (per H0 della ROC)."""
    return rng.integers(0, 2, MARK_SIZE)


def to_bipolar(bits):
    """0/1 -> -1/+1, così la similarità tra marchi casuali è circa 0."""
    return 2.0 * np.asarray(bits, dtype=float) - 1.0


def similarity(a, b):
    """Similarità coseno (LAB-3, EXERCISEs_SOL). Vale 0 se un vettore è nullo."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


# ---------------------------------------------------------------------------
# I 6 attacchi ammessi. Tutti: uint8 512x512 in ingresso e in uscita.
# ---------------------------------------------------------------------------
def awgn(img, sigma, seed=0):
    """Rumore gaussiano additivo con deviazione standard sigma."""
    noise = np.random.default_rng(seed).normal(0, sigma, img.shape)
    return to_uint8(img.astype(float) + noise)


def blur(img, sigma):
    """Filtro gaussiano passa-basso."""
    return to_uint8(gaussian_filter(img.astype(float), sigma))


def sharpen(img, alpha, sigma=1.0):
    """Unsharp masking: img + alpha * (img - blur(img))."""
    x = img.astype(float)
    return to_uint8(x + alpha * (x - gaussian_filter(x, sigma)))


def median(img, k):
    """Filtro mediano k x k (k dispari)."""
    return median_filter(img, size=int(k)).astype(np.uint8)


def jpeg(img, qf):
    """Compressione JPEG in memoria (cv2.imencode, nessun file temporaneo)."""
    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(qf)])
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)


def resize(img, scale):
    """Rimpicciolisce di 'scale' e riporta alla dimensione originale."""
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
    Attacco casuale per la ROC (idea del LAB-3, Example 2), con parametri
    estratti negli intervalli del piano. Una o due operazioni in sequenza.
    Restituisce l'immagine attaccata e una descrizione testuale.
    """
    out = img
    steps = []
    for _ in range(rng.integers(1, 3)):          # 1 o 2 attacchi
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
