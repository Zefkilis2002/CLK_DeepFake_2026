"""
The group's embedding: ADDITIVE Spread Spectrum in DCT (LAB-3).

Idea (WatermarkingSpreadSpectrum2022.pdf; LAB3-ROC.ipynb, embedding with
v='additive'):
  1. 2D DCT of the whole image;
  2. the 1024 coefficients with the largest magnitude are chosen (DC excluded):
     they are the perceptually most important components, so to remove the
     mark the attacker has to ruin the image;
  3. ALPHA * x is added to the magnitude of each one, with x = +1 / -1;
  4. the signs are restored, IDCT, rounding to uint8.

Why additive and not multiplicative (|C|(1 + alpha*x))? After rounding to
uint8 the multiplicative scheme gives very different results from image to
image: some images stay identical (mark lost), others drop to 56-57 dB,
and on some images different marks become correlated (risk of false
positives). With the additive scheme every coefficient changes by the same
amount: almost the same WPSNR on all images and none of these problems
(measurements in the README).

ALPHA sets the quality (WPSNR) / robustness trade-off.
It must be identical to the one written in detection_groupname.py.
"""
import cv2
import numpy as np
from scipy.fft import dct, idct

# --- parameters (allowed fine-tuning: ALPHA and locations) ------------------
ALPHA = 5.0          # sets the mark strength: WPSNR about 64 dB (see README)
MARK_SIZE = 1024     # length of the competition watermark


def dct2(img):      # Converts an image into 2D Discrete Cosine Transform (DCT) coefficients
    return dct(dct(img, axis=0, norm="ortho"), axis=1, norm="ortho")


def idct2(coeff):   # Converts the 2D DCT coefficients back into the image’s spatial representation
    return idct(idct(coeff, axis=1, norm="ortho"), axis=0, norm="ortho")


def embed_dct_ss(original, bits, alpha=ALPHA):
    """Embeds the 0/1 bits into the uint8 image 'original'. Returns uint8."""
    x = 2.0 * np.asarray(bits, dtype=float) - 1.0        # 0/1 -> -1/+1     +1 ->increase coef, -1 -> decrease coef, with 0/1, 0 would do nothing
    coeff = dct2(original.astype(float))    # converts image to decimals, DCT uses decimals & negatives
    signs = np.sign(coeff)      # each coef is split into its sign(+/1) and its magnitude 6880.5 (E.g -6880.5)
    magnitudes = np.abs(coeff)

    # locations of the largest coefficients, skipping the first one (the DC)
    locations = np.argsort(-magnitudes, axis=None)[1:MARK_SIZE + 1]     # choose where to hide the bits, takes positions 1 to 1024 and skips 0

    edited = magnitudes.ravel().copy()
    edited[locations] += alpha * x                       # additive formula
    edited = edited.reshape(coeff.shape) * signs

    result = idct2(edited)      # converting frequencies back to pixels
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)    # caps values to a valid pixel range and converts back to normal 8-bit image


def embedding(input1, input2):      # simply handles file loading and calling
    """
    input1: path of the original image
    input2: path of the watermark (.npy with 1024 bits 0/1)
    output1: watermarked image (np.uint8, 512x512)
    """
    original = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    bits = np.load(input2)
    output1 = embed_dct_ss(original, bits)
    return output1
