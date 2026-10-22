"""
Detection del gruppo {{GROUP}} (file unico GENERATO da submission/build_submission.py,
preset {{PRESET}}: non modificare a mano, rigenerare dal preset).

Schema: Spread Spectrum in DCT (LAB-3), modo {{MODE}}, detection non-blind:
  W_extracted dalla marchiata, W_attacked dall'attaccata (posizioni dall'originale),
  presente se sim(W_extracted, W_attacked) >= TAU.
Il watermark non viene letto; TAU viene dalla ROC (scripts/run_roc.py).
Nessuna stampa, nessuna finestra, nessun percorso relativo.
"""
import cv2
import numpy as np
from scipy.fft import dct, idct
from scipy.signal import convolve2d

ALPHA = {{ALPHA}}
MODE = "{{MODE}}"
N = {{N}}
TAU = {{TAU}}

{{WPSNR_CODE}}

{{SCHEME_CODE}}

{{SIMILARITY_CODE}}


def detection(input1, input2, input3):
    """
    input1: originale, input2: marchiata, input3: attaccata (percorsi)
    output1: 1 se il marchio è presente, altrimenti 0
    output2: WPSNR(marchiata, attaccata)
    """
    original = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    watermarked = cv2.imread(input2, cv2.IMREAD_GRAYSCALE)
    attacked = cv2.imread(input3, cv2.IMREAD_GRAYSCALE)
    if attacked.shape != original.shape:
        attacked = cv2.resize(attacked, (original.shape[1], original.shape[0]))

    # la detection viene SEMPRE eseguita, qualunque sia il WPSNR
    w_extracted = extract_dct_ss(original, watermarked, ALPHA, MODE, N)
    w_attacked = extract_dct_ss(original, attacked, ALPHA, MODE, N)
    sim = similarity(w_extracted, w_attacked)

    output1 = 1 if sim >= TAU else 0
    output2 = float(wpsnr(watermarked, attacked))
    return output1, output2
