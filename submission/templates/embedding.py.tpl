"""
Embedding del gruppo {{GROUP}} (file GENERATO da submission/build_submission.py,
preset {{PRESET}}: non modificare a mano, rigenerare dal preset).

Schema: Spread Spectrum in DCT (LAB-3), modo {{MODE}}, sui {{N}} coefficienti
DCT più grandi (DC escluso); con N > 1024 ogni bit è ripetuto N/1024 volte.
"""
import cv2
import numpy as np
from scipy.fft import dct, idct

ALPHA = {{ALPHA}}
MODE = "{{MODE}}"
N = {{N}}

{{SCHEME_CODE}}


def embedding(input1, input2):
    """input1: percorso originale, input2: percorso watermark .npy -> immagine marchiata uint8."""
    original = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    bits = np.load(input2)
    return embed_dct_ss(original, bits, ALPHA, MODE, N)
