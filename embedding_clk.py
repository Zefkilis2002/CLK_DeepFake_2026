"""
How we hide a watermark inside an image:

  1. Convert the image into frequency data (DCT).
  2. Pick the 1024 strongest frequencies (skip the first one - brightness).
  3. For each bit of the watermark: add ALPHA if the bit is 1, subtract ALPHA if it is 0.
  4. Convert back to a normal image.

Why this way? Because every pixel changes by the same small amount,
so the image always looks almost identical to the original.

ALPHA controls the trade-off: higher = easier to detect, lower = better quality.
Must be the same value used in detection_clk.py.
"""

import cv2
import numpy as np
from scipy.fft import dct, idct

# --- parameters (allowed fine-tuning: ALPHA and locations) ------------------
ALPHA = 5.0          # mark strength: WPSNR about 64 dB (see README)
MARK_SIZE = 1024     # length of the competition watermark


def dct2(img):      # converts an image into 2D Discrete Cosine Transforms (DCT) coefficients
    return dct(dct(img, axis=0, norm="ortho"), axis=1, norm="ortho")


def idct2(coeff):   # converts 2D DCT coefficients back into an image's spatial representation
    return idct(idct(coeff, axis=1, norm="ortho"), axis=0, norm="ortho")

# Additive Spread Spectrum in DCT
def embed_dct_ss(original, bits, alpha=None):
    """Embeds the 0/1 bits into the uint8 image 'original'. Returns uint8."""
    alpha = ALPHA if alpha is None else alpha    # read ALPHA at call time, so changing embedding_clk.ALPHA works
    x =2.0 * np.asarray(bits, dtype=float) - 1.0        # 0/1 -> -1/+1     +1 -> increase coef, -1 -> decrease coef, with 0/1, 0 would do nothing
    coeff = dct2(original.astype(float))   # converts image to decimals, DCT uses decimals & negatives
    signs = np.sign(coeff) # it saves only the sign for each coef
    magnitudes = np.abs(coeff) # it saves every magnituded with its absolute value (no sign)

    # locations of the largest coefficients, skipping the first one (the DC)
    locations = np.argsort(-magnitudes, axis=None)[1:MARK_SIZE + 1]

    edited = magnitudes.ravel().copy() # Flatten the 2D intensity matrix into a 1D vector and make a copy of it
    edited[locations] += alpha * x                       # additive formula (change only 1024 coefficients)
    edited = edited.reshape(coeff.shape) * signs  # Restore the 2D matrix and reapply the original signs

    result = idct2(edited)      # converts frequencies back to pixels
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)    # caps values to a valid pixel range and converts back to normal 8-bit image


def embedding(input1, input2):
    """
    input1: path of the original image
    input2: path of the watermark (.npy with 1024 bits 0/1)
    output1: watermarked image (np.uint8, 512x512)
    """
    original = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    bits = np.load(input2)
    output1 = embed_dct_ss(original, bits)
    return output1
