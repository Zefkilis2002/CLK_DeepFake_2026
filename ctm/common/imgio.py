"""C2: lettura/scrittura immagini e unica regola di quantizzazione."""
import os

import cv2
import numpy as np

from config.settings import IMAGES_DIR


def read_gray(path):
    """Legge un'immagine in scala di grigi con OpenCV (uint8)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def write_bmp(path, img):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, img)


def to_uint8(x):
    """L'UNICA regola di arrotondamento del progetto: clip in [0,255], poi rint."""
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def match_shape(img, ref_shape):
    """Riporta un'immagine alla dimensione di riferimento (es. 512x512)."""
    if img.shape != tuple(ref_shape):
        img = cv2.resize(img, (ref_shape[1], ref_shape[0]))
    return img


def list_images(folder=IMAGES_DIR):
    names = sorted(n for n in os.listdir(folder) if n.lower().endswith(".bmp"))
    return [os.path.join(folder, n) for n in names]
