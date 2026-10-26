"""
How we attack other groups watermarks:

1. attacks(): applies one or more of the 6 allowed attacks to an image.
   Available attacks: 'awgn', 'blur', 'sharp', 'jpeg', 'resize', 'median'.

2. search_attack(): automatically finds the lightest attack that removes
   another group's watermark while keeping the image quality above 35 dB.
   - Tries different attack strengths (0 = very light, 1 = very strong).
   - Uses bisection to find the minimum strength that works.

RULE: we only use the watermarked image, never the original.
"""

import argparse
import contextlib
import csv
import importlib.util
import io
import os

import cv2
import numpy as np

from scipy.ndimage import gaussian_filter, median_filter


def read_gray(path):
    """Loads an image from disk as grayscale (black and white, 512x512 pixels)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def to_uint8(x):
    """Rounds pixel values and clamps them to the valid range [0, 255], then converts to uint8."""
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


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


# ---------------------------------------------------------------------------
# 1. Function required by the rules
# ---------------------------------------------------------------------------
def apply_attacks(img, names, params):
    """Applies the attacks 'names' in sequence with the parameters 'params'."""
    out = img
    for name, p in zip(names, params):
        out = ATTACKS[name](out, p)
    return out


def attacks(input1, attack_name, param_array):
    """
    input1: path of the watermarked image
    attack_name: string or list of strings ('awgn', 'blur', 'sharp', 'jpeg', 'resize', 'median')
    param_array: list of parameters, one per attack
    output1: attacked image (uint8)
    """
    names = [attack_name] if isinstance(attack_name, str) else list(attack_name)
    params = list(np.atleast_1d(param_array))
    return apply_attacks(read_gray(input1), names, params)


# ---------------------------------------------------------------------------
# 2. Chain library: strength s in [0,1] -> parameters
# ---------------------------------------------------------------------------
def k_median(s):
    return [3, 5, 7, 9][min(int(s * 4), 3)]


CHAINS = {
    "jpeg":             [("jpeg", lambda s: int(round(95 - 90 * s)))],
    "blur":             [("blur", lambda s: 0.2 + 2.8 * s)],
    "median":           [("median", k_median)],
    "awgn":             [("awgn", lambda s: 0.5 + 29.5 * s)],
    "resize":           [("resize", lambda s: 0.95 - 0.8 * s)],
    "sharp":            [("sharp", lambda s: 0.2 + 4.8 * s)],
    "blur+jpeg":        [("blur", lambda s: 0.2 + 1.8 * s), ("jpeg", lambda s: int(round(95 - 60 * s)))],
    "median+jpeg":      [("median", lambda s: 3 if s < 0.5 else 5), ("jpeg", lambda s: int(round(95 - 70 * s)))],
    "awgn+jpeg":        [("awgn", lambda s: 0.5 + 15 * s), ("jpeg", lambda s: int(round(95 - 60 * s)))],
    "resize+jpeg":      [("resize", lambda s: 0.95 - 0.6 * s), ("jpeg", lambda s: int(round(95 - 60 * s)))],
    "awgn+median":      [("awgn", lambda s: 0.5 + 15 * s), ("median", lambda s: 3 if s < 0.5 else 5)],
    "blur+awgn":        [("blur", lambda s: 0.2 + 1.8 * s), ("awgn", lambda s: 0.5 + 15 * s)],
    "median+blur+jpeg": [("median", lambda s: 3), ("blur", lambda s: 0.2 + 1.3 * s),
                         ("jpeg", lambda s: int(round(95 - 60 * s)))],
}


def chain_params(chain, s):
    names = [name for name, _ in CHAINS[chain]]
    params = [f(s) for _, f in CHAINS[chain]]
    return names, params


def describe(names, params):
    """Text for the Excel log, e.g. 'blur 1.20 + jpeg 70'."""
    return " + ".join("%s %s" % (n, ("%.2f" % p) if isinstance(p, float) else p)
                      for n, p in zip(names, params))


def strongest_allowed(wm, chain, min_wpsnr=35.5, steps=6):
    """Maximum strength with WPSNR >= min_wpsnr, found with LOCAL bisection (no detection)."""
    names, params = chain_params(chain, 1.0)
    attacked = apply_attacks(wm, names, params)
    if wpsnr(wm, attacked) >= min_wpsnr:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(steps):
        mid = (lo + hi) / 2
        names, params = chain_params(chain, mid)
        if wpsnr(wm, apply_attacks(wm, names, params)) >= min_wpsnr:
            lo = mid
        else:
            hi = mid
    return lo


def search_attack(wm, oracle, chains=None, bisect_steps=4):
    """
    Searches for the successful attack with the highest WPSNR.
    wm: watermarked image (uint8). oracle(img) -> (present, wpsnr).
    Returns (image, wpsnr, description) of the best one, or None.
    """
    best = None
    for chain in (chains or CHAINS):
        # maximum allowed strength: if the mark survives here, the chain is useless
        s_hi = strongest_allowed(wm, chain)
        names, params = chain_params(chain, s_hi)
        attacked = apply_attacks(wm, names, params)
        w = wpsnr(wm, attacked)
        if best is not None and w <= best[1]:
            continue                       # cannot beat the best one: no query
        present, w = oracle(attacked)
        if present or w < 35:
            continue
        found = (attacked, w, describe(names, params))
        # bisection: lightest attack that still removes the mark
        s_lo = 0.0
        for _ in range(bisect_steps):
            mid = (s_lo + s_hi) / 2
            names, params = chain_params(chain, mid)
            attacked = apply_attacks(wm, names, params)
            if wpsnr(wm, attacked) <= found[1]:
                s_lo = mid                 # would not improve: try lighter
                continue
            present, w = oracle(attacked)
            if not present and w >= 35:
                s_hi, found = mid, (attacked, w, describe(names, params))
            else:
                s_lo = mid
        if best is None or found[1] > best[1]:
            best = found
    return best


# ---------------------------------------------------------------------------
# Oracle: another group's detection (the only place that sees the original)
# ---------------------------------------------------------------------------
def load_detection(path):
    """Loads detection_<group>.py or .pyc (like the labs' .pyc grader)."""
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.detection


def make_oracle(detection, original_path, watermarked_path, tmp_path):
    """Returns oracle(img) -> (present, wpsnr) using the given detection."""
    def oracle(attacked):
        cv2.imwrite(tmp_path, attacked)
        with contextlib.redirect_stdout(io.StringIO()):     # ignore any prints
            present, w = detection(original_path, watermarked_path, tmp_path)
        return int(present), float(w)
    return oracle


# ---------------------------------------------------------------------------
# Competition day: attack all the downloaded images
# ---------------------------------------------------------------------------
def find_detection(folder, group):
    for ext in (".py", ".pyc"):
        path = os.path.join(folder, "detection_%s%s" % (group, ext))
        if os.path.exists(path):
            return path
    return None


def attack_folder(our, targets, detectors, originals, out_dir, log_path):
    """
    targets: folder with the other groups' images, named groupB_image.bmp
    detectors: folder with detection_groupB.py / .pyc
    originals: folder with the 3 original competition images (image.bmp)
    """
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = os.path.abspath(os.path.join(out_dir, "_tmp_attacked.bmp"))
    new_log = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        log = csv.writer(f, delimiter=";")
        if new_log:
            log.writerow(["Image", "Group", "WPSNR", "Attack(s) with parameters"])
            f.flush()
        for fname in sorted(os.listdir(targets)):
            if not fname.lower().endswith(".bmp") or "_" not in fname:
                continue
            group, image = fname[:-4].split("_", 1)
            if group == our:
                continue
            det_path = find_detection(detectors, group)
            if det_path is None:
                print("%s: detection di %s non trovata" % (fname, group))
                continue
            wm_path = os.path.abspath(os.path.join(targets, fname))
            orig_path = os.path.abspath(os.path.join(originals, image + ".bmp"))
            oracle = make_oracle(load_detection(det_path), orig_path, wm_path, tmp_path)

            result = search_attack(read_gray(wm_path), oracle)
            if result is None:
                print("%s: nessun attacco riuscito" % fname)
                continue
            attacked, w, text = result
            out_path = os.path.join(out_dir, "%s_%s_%s.bmp" % (our, group, image))
            cv2.imwrite(out_path, attacked)
            # check on the saved file, exactly the one we will upload
            present, w_check = oracle(read_gray(out_path))
            ok = (present == 0 and w_check >= 35)
            print("%s: WPSNR %.2f | %s | verificato: %s" % (fname, w_check, text, ok))
            if ok:
                log.writerow([image, group, "%.2f" % w_check, text])
                f.flush()
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Attacca le immagini degli altri gruppi")
    ap.add_argument("--our", required=True, help="nome del nostro gruppo")
    ap.add_argument("--targets", required=True, help="cartella gruppoB_immagine.bmp")
    ap.add_argument("--detectors", required=True, help="cartella detection_gruppoB.py/.pyc")
    ap.add_argument("--originals", required=True, help="cartella con le immagini originali")
    ap.add_argument("--out", default="attacked", help="cartella di uscita")
    ap.add_argument("--log", default="attack_log.csv", help="log per l'Excel")
    a = ap.parse_args()
    attack_folder(a.our, a.targets, a.detectors, a.originals, a.out, a.log)
