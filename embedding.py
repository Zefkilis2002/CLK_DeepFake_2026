"""
Embedding del gruppo: Spread Spectrum ADDITIVO in DCT (LAB-3).

Idea (WatermarkingSpreadSpectrum2022.pdf; LAB3-ROC.ipynb, embedding con
v='additive'):
  1. DCT 2D dell'intera immagine;
  2. si scelgono i 1024 coefficienti di modulo più grande (DC escluso):
     sono le componenti percettivamente più importanti, quindi per cancellare
     il marchio l'attaccante deve rovinare l'immagine;
  3. al modulo di ciascuno si somma ALPHA * x, con x = +1 / -1;
  4. si rimettono i segni, IDCT, arrotondamento a uint8.

Perché additivo e non moltiplicativo (|C|(1 + alpha*x))? Dopo l'arrotondamento
a uint8 il moltiplicativo dà risultati molto diversi da immagine a immagine:
alcune immagini restano identiche (marchio perso), altre scendono a 56-57 dB,
e su alcune immagini marchi diversi diventano correlati (rischio di falsi
positivi). Con l'additivo ogni coefficiente cambia della stessa quantità:
WPSNR quasi uguale su tutte le immagini e nessuno di questi problemi
(misure nel README).

ALPHA regola il compromesso qualità (WPSNR) / robustezza.
Deve essere identico a quello scritto in detection_groupname.py.
"""
import cv2
import numpy as np
from scipy.fft import dct, idct

# --- parametri (fine-tuning ammesso: ALPHA e posizioni) ---------------------
ALPHA = 5.0          # forza del marchio (vedi tabella dei preset nel README)
MARK_SIZE = 1024     # lunghezza del watermark della gara


def dct2(img):
    return dct(dct(img, axis=0, norm="ortho"), axis=1, norm="ortho")


def idct2(coeff):
    return idct(idct(coeff, axis=1, norm="ortho"), axis=0, norm="ortho")


def embed_dct_ss(original, bits, alpha=ALPHA):
    """Inserisce i bit 0/1 nell'immagine uint8 'original'. Restituisce uint8."""
    x = 2.0 * np.asarray(bits, dtype=float) - 1.0        # 0/1 -> -1/+1
    coeff = dct2(original.astype(float))
    signs = np.sign(coeff)
    magnitudes = np.abs(coeff)

    # posizioni dei coefficienti più grandi, saltando il primo (il DC)
    locations = np.argsort(-magnitudes, axis=None)[1:MARK_SIZE + 1]

    edited = magnitudes.ravel().copy()
    edited[locations] += alpha * x                       # formula additiva
    edited = edited.reshape(coeff.shape) * signs

    result = idct2(edited)
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def embedding(input1, input2):
    """
    input1: percorso dell'immagine originale
    input2: percorso del watermark (.npy con 1024 bit 0/1)
    output1: immagine marchiata (np.uint8, 512x512)
    """
    original = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    bits = np.load(input2)
    output1 = embed_dct_ss(original, bits)
    return output1
