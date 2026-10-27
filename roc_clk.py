"""
Computes the detection threshold TAU using the ROC curve.
Steps:
  1. Embed our watermark in images and apply random attacks.
  2. Score similarity for images WITH our mark (H1) and WITHOUT (H0).
  3. Pick TAU: highest detection rate with false alarms below FPR_TARGET.
  4. Verify TAU against the value in detection_clk.py.
Usage:  python roc_clk.py          (uses ALPHA from embedding_clk.py)
        python roc_clk.py 4.5      (try a different alpha)
"""
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")                       # save the plot without opening windows
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

from scipy.fft import dct, idct
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import convolve2d

# ---------------------------------------------------------------------------
# Functions copied from common.py
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Copied from embedding_clk.py (embedding parameters and embed_dct_ss)
# ---------------------------------------------------------------------------
ALPHA = 5.0          # mark strength: WPSNR about 64 dB (see README)


def dct2(img):
    return dct(dct(img, axis=0, norm="ortho"), axis=1, norm="ortho")


def idct2(coeff):
    return idct(idct(coeff, axis=1, norm="ortho"), axis=0, norm="ortho")


def embed_dct_ss(original, bits, alpha=ALPHA):
    """Embeds the 0/1 bits into the uint8 image 'original'. Returns uint8."""
    x = 2.0 * np.asarray(bits, dtype=float) - 1.0        # 0/1 -> -1/+1
    coeff = dct2(original.astype(float))
    signs = np.sign(coeff)
    magnitudes = np.abs(coeff)

    # locations of the largest coefficients, skipping the first one (the DC)
    locations = np.argsort(-magnitudes, axis=None)[1:MARK_SIZE + 1]

    edited = magnitudes.ravel().copy()
    edited[locations] += alpha * x                       # additive formula
    edited = edited.reshape(coeff.shape) * signs

    result = idct2(edited)
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Copied from detection_clk.py (threshold written there and extraction)
# ---------------------------------------------------------------------------
DET_ALPHA = 5.0
DET_TAU = 0.1253     # threshold from the ROC (ALPHA = 5): TPR 0.875, FPR 0 on 1101 H0 cases


def extract_dct_ss(original, image, alpha=DET_ALPHA):
    """Estimate of x (+-1) at the 1024 locations chosen on the ORIGINAL."""
    c0 = np.abs(dct(dct(original.astype(float), axis=0, norm="ortho"), axis=1, norm="ortho"))
    c1 = np.abs(dct(dct(image.astype(float), axis=0, norm="ortho"), axis=1, norm="ortho"))
    locations = np.argsort(-c0, axis=None)[1:MARK_SIZE + 1]
    base = c0.ravel()[locations]
    changed = c1.ravel()[locations]
    return (changed - base) / alpha


FPR_TARGET = 0.01          # within the [0, 0.1] range required by the rules
ATTACKS_PER_IMAGE = 10
N_CALIB = 40               # calibration images
N_TEST = 20                # test images (different from the first ones)
SEED = 2026


def compute_scores(paths, embed, extract, param, bits, rng):
    """Returns (scores, labels) of H1 and H0 for the given images."""
    scores, labels = [], []
    for path in paths:
        orig = read_gray(path)
        wm = embed(orig, bits, param)
        w_ext = extract(orig, wm, param)
        other = embed(orig, random_mark(rng), param)       # image of "another group"
        for _ in range(ATTACKS_PER_IMAGE):
            seed = int(rng.integers(1_000_000))
            attacked, _ = random_attack(wm, np.random.default_rng(seed))
            if wpsnr(wm, attacked) < 35:
                continue
            w_att = extract(orig, attacked, param)
            # H1
            scores.append(similarity(w_ext, w_att)); labels.append(1)
            # H0-a: random watermark
            scores.append(similarity(to_bipolar(random_mark(rng)), w_att)); labels.append(0)
            # H0-b and H0-c: same attack on another group's image and on the original
            for img in (other, orig):
                att0, _ = random_attack(img, np.random.default_rng(seed))
                scores.append(similarity(w_ext, extract(orig, att0, param))); labels.append(0)
    return np.array(scores), np.array(labels)


def choose_tau(scores, labels, fpr_target=FPR_TARGET):
    """Tau with the highest TPR among those with FPR <= fpr_target (LAB-4)."""
    fpr, tpr, thr = roc_curve(labels, scores, drop_intermediate=False)
    valid = np.flatnonzero(fpr <= fpr_target)
    best = valid[np.argmax(tpr[valid])]
    h0 = scores[labels == 0]
    floor = h0.mean() + 4 * h0.std()
    tau = max(float(thr[best]), float(floor))
    return tau, fpr, tpr, auc(fpr, tpr), floor


def rates(scores, labels, tau):
    pred = scores >= tau
    tpr = np.mean(pred[labels == 1])
    fpr = np.mean(pred[labels == 0])
    return tpr, fpr


def run(embed, extract, param, n_calib=N_CALIB, n_test=N_TEST, name="dct_ss", plot=True):
    bits = load_mark()
    rng = np.random.default_rng(SEED)
    paths = list_images()
    calib, test = paths[:n_calib], paths[n_calib:n_calib + n_test]

    s_cal, l_cal = compute_scores(calib, embed, extract, param, bits, rng)
    tau, fpr, tpr, roc_auc, floor = choose_tau(s_cal, l_cal)
    s_test, l_test = compute_scores(test, embed, extract, param, bits, rng)
    test_tpr, test_fpr = rates(s_test, l_test, tau)

    h0 = s_cal[l_cal == 0]
    print("schema %s, parametro %g" % (name, param))
    print("  campioni calibrazione: H1=%d H0=%d" % (np.sum(l_cal == 1), np.sum(l_cal == 0)))
    print("  AUC = %.4f | H0: media %.4f, std %.4f, max %.4f | soglia minima 4 std = %.4f"
          % (roc_auc, h0.mean(), h0.std(), h0.max(), floor))
    print("  TAU = %.4f" % tau)
    print("  calibrazione: TPR %.3f FPR %.4f" % rates(s_cal, l_cal, tau))
    print("  verifica:     TPR %.3f FPR %.4f" % (test_tpr, test_fpr))
    if param == DET_ALPHA:
        # check of the threshold written by hand in the detection
        print("  TAU della detection = %.4f -> calibrazione TPR %.3f FPR %.4f | verifica TPR %.3f FPR %.4f"
              % ((DET_TAU,) + rates(s_cal, l_cal, DET_TAU) + rates(s_test, l_test, DET_TAU)))

    if plot:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(fpr, tpr, lw=2, label="AUC = %.3f" % roc_auc)
        axes[0].plot([0, 1], [0, 1], "--", color="gray")
        axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
        axes[0].legend(loc="lower right")
        axes[1].hist(h0, bins=40, alpha=.6, label="H0")
        axes[1].hist(s_cal[l_cal == 1], bins=40, alpha=.6, label="H1")
        axes[1].axvline(tau, color="black", ls="--", label="tau = %.3f" % tau)
        axes[1].set(xlabel="similarità", ylabel="conteggio", title="Distribuzioni")
        axes[1].legend()
        plt.tight_layout()
        out = os.path.join(RESULTS_DIR, "roc_%s_%g.png" % (name, param))
        plt.savefig(out, dpi=120)
        plt.close(fig)
        print("  grafico salvato in", out)
    return tau


if __name__ == "__main__":
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else ALPHA
    run(embed_dct_ss, extract_dct_ss, alpha)
