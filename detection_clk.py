"""
How we check if our watermark survived an attack:

  1. Extract the watermark from the watermarked image (using the original -> WATERMARKED_IMG - ORIGINAL_IMG).
  2. Extract the watermark from the attacked image (using the original -> ATTACKED_IMG - ORIGINAL_IMG).
  3. Compare the two: if they are similar enough (>= TAU), the mark is still there.

We never read the watermark file directly: we extract it from the images.
TAU is a fixed threshold computed once with roc_clk.py.
"""
import cv2
import numpy as np
from scipy.fft import dct
from scipy.signal import convolve2d

# --- parameters: must match embedding.py ------------------------------------
ALPHA = 5.0
MARK_SIZE = 1024
TAU = 0.1253         # threshold from the ROC (ALPHA = 5): TPR 0.875, FPR 0 on 1101 H0 cases


# ---------------------------------------------------------------------------
# WPSNR (lab code, LAB3, unchanged)
# ---------------------------------------------------------------------------
def _csffun(u, v):
    """
    Computes the Contrast Sensitivity Function (CSF) value for given spatial frequencies.
    This is a Python translation of csffun.m.
    """
    # Calculate radial frequency
    f = np.sqrt(u**2 + v**2)
    w = 2 * np.pi * f / 60

    # Intermediate spatial frequency response
    sigma = 2
    Sw = 1.5 * np.exp(-sigma**2 * w**2 / 2) - np.exp(-2 * sigma**2 * w**2 / 2)

    # High-frequency modification
    sita = np.arctan2(v, u) # Use arctan2 for quadrant correctness
    bita = 8
    f0 = 11.13
    w0 = 2 * np.pi * f0 / 60
    
    # Avoid division by zero or overflow in exp
    exp_term = np.exp(bita * (w - w0))
    Ow = (1 + exp_term * (np.cos(2 * sita))**4) / (1 + exp_term)
    
    # Final response
    Sa = Sw * Ow
    return Sa

def _csfmat():
    """
    Computes the CSF frequency response matrix.
    This is a Python translation of csfmat.m.
    """
    # Define frequency range
    min_f, max_f, step_f = -20, 20, 1
    freq_range = np.arange(min_f, max_f + step_f, step_f)
    n = len(freq_range)
    
    # Create frequency grids
    u, v = np.meshgrid(freq_range, freq_range, indexing='xy')
    
    # Compute the frequency response matrix by calling _csffun
    Fmat = _csffun(u, v)
    
    return Fmat

def _get_csf_filter():
    """
    Computes the 2D filter coefficients for the CSF.
    This is a Python translation of csf.m, which uses fsamp2.
    The fsamp2 function is implemented using an inverse Fourier transform.
    """
    # 1. Get the frequency response matrix
    Fmat = _csfmat()
    
    # 2. Compute the 2D filter coefficients using the frequency sampling method
    # This is equivalent to MATLAB's fsamp2(Fmat)
    # The shifts are necessary to handle the centered frequency response
    filter_coeffs = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(Fmat)))
    
    # The filter coefficients should be real
    return np.real(filter_coeffs)


def wpsnr(image_a, image_b):
    """
    Computes the Weighted Peak Signal-to-Noise Ratio (WPSNR) between two images.

    This function is a Python translation of the provided WPSNR.m script. It uses a 
    Contrast Sensitivity Function (CSF) to weigh the spatial frequencies of the error image.

    Args:
        image_a (np.ndarray): The original image, as a NumPy array. 
                              Values can be uint8 (0-255) or float (0.0-1.0).
        image_b (np.ndarray): The distorted image, as a NumPy array. 
                              Must have the same dimensions and type as image_a.

    Returns:
        float: The WPSNR value in decibels (dB).
    """
    

    if image_a.shape != image_b.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Normalize images to the [0, 1] range if they are not already floats
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

    # --- WPSNR Calculation ---
    # Handle identical images case
    if np.array_equal(A, B):
        return 9999999.0  # Return a large number for infinite PSNR, as in the original code

    # 1. Calculate the error image
    error_image = A - B
    
    # 2. Get the Contrast Sensitivity Function (CSF) filter
    csf_filter = _get_csf_filter()
    
    # 3. Filter the error image with the CSF filter (2D convolution)
    # This is equivalent to MATLAB's filter2(fc, e)
    weighted_error = convolve2d(error_image, csf_filter, mode='same', boundary='wrap')
    
    # 4. Calculate the weighted mean squared error (WMSE)
    wmse = np.mean(weighted_error**2)
    
    # 5. Calculate WPSNR
    # The peak signal value is 1.0 because the images are normalized
    if wmse == 0:
        return 9999999.0 # Should be caught by the identity check, but included for safety
    
    decibels = 20 * np.log10(1.0 / np.sqrt(wmse))
    
    return decibels


# ---------------------------------------------------------------------------
# Extraction and similarity (LAB-3: detection with v='additive' and similarity)
# ---------------------------------------------------------------------------
def extract_dct_ss(original, image, alpha=ALPHA):
    """Recovers the 1024 hidden watermark bits (+1 or -1) by comparing with the original image."""
    c0 = np.abs(dct(dct(original.astype(float), axis=0, norm="ortho"), axis=1, norm="ortho"))
    c1 = np.abs(dct(dct(image.astype(float), axis=0, norm="ortho"), axis=1, norm="ortho"))
    locations = np.argsort(-c0, axis=None)[1:MARK_SIZE + 1]
    base = c0.ravel()[locations]
    changed = c1.ravel()[locations]
    return (changed - base) / alpha


def similarity(a, b):
    """Measures how similar two lists of numbers are (from -1 to 1; 1 means identical)."""
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def detection(input1, input2, input3):
    """
    input1: path of the original
    input2: path of the watermarked image
    input3: path of the attacked image
    output1: 1 if the mark is present in the attacked image, otherwise 0
    output2: WPSNR(watermarked, attacked)
    """
    original = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    watermarked = cv2.imread(input2, cv2.IMREAD_GRAYSCALE)
    attacked = cv2.imread(input3, cv2.IMREAD_GRAYSCALE)
    if attacked.shape != original.shape:          # safety: same size
        attacked = cv2.resize(attacked, (original.shape[1], original.shape[0]))     # it resizes the shape of the attacked image to match original

    # detection is ALWAYS performed, whatever the WPSNR
    w_extracted = extract_dct_ss(original, watermarked)
    w_attacked = extract_dct_ss(original, attacked)
    sim = similarity(w_extracted, w_attacked)   # compares watermark estimate of the watermark/attack images

    output1 = 1 if sim >= TAU else 0
    output2 = float(wpsnr(watermarked, attacked))
    return output1, output2         # detection result 1 >= threashold, 0 if below threshold
