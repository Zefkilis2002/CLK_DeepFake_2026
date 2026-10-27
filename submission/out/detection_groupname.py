"""
Detection del gruppo groupname (file unico GENERATO da submission/build_submission.py,
preset P66: non modificare a mano, rigenerare dal preset).

Schema: Spread Spectrum in DCT (LAB-3), modo additive, detection non-blind:
  W_extracted dalla marchiata, W_attacked dall'attaccata (posizioni dall'originale),
  presente se sim(W_extracted, W_attacked) >= TAU.
Il watermark non viene letto; TAU viene dalla ROC (scripts/run_roc.py).
Nessuna stampa, nessuna finestra, nessun percorso relativo.
"""
import cv2
import numpy as np
from scipy.fft import dct, idct
from scipy.signal import convolve2d

ALPHA = 4.360352
MODE = "additive"
N = 1024
TAU = 0.1214

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

MARK_SIZE = 1024


def dct2(img):
    return dct(dct(img, axis=0, norm="ortho"), axis=1, norm="ortho")


def idct2(coeff):
    return idct(idct(coeff, axis=1, norm="ortho"), axis=0, norm="ortho")


def select_locations(magnitudes, n):
    """Posizioni degli n coefficienti più grandi, saltando il primo (DC)."""
    return np.argsort(-magnitudes, axis=None)[1:n + 1]


def embed_dct_ss(original, bits, alpha, mode, n):
    x = np.resize(2.0 * np.asarray(bits, dtype=float) - 1.0, n)   # +-1, ripetuto se n > 1024
    coeff = dct2(original.astype(float))
    signs, magnitudes = np.sign(coeff), np.abs(coeff)
    locations = select_locations(magnitudes, n)
    edited = magnitudes.ravel().copy()
    if mode == "additive":
        edited[locations] += alpha * x
    else:
        edited[locations] *= 1.0 + alpha * x
    result = idct2(edited.reshape(coeff.shape) * signs)
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def extract_dct_ss(original, image, alpha, mode, n):
    c0 = np.abs(dct2(original.astype(float)))
    c1 = np.abs(dct2(image.astype(float)))
    locations = select_locations(c0, n)
    base, changed = c0.ravel()[locations], c1.ravel()[locations]
    if mode == "additive":
        estimate = (changed - base) / alpha
    else:
        estimate = (changed - base) / (alpha * base)
    return estimate.reshape(-1, MARK_SIZE).mean(axis=0)            # media delle copie

def similarity(a, b):
    """Similarità coseno; 0 se uno dei due vettori è nullo."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


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
