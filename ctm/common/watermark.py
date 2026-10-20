"""C4: watermark e similarità."""
import hashlib

import numpy as np

from config.settings import MARK_PATH, MARK_SIZE


def load_mark(path=MARK_PATH):
    """Watermark della gara: 1024 bit 0/1 (.npy)."""
    bits = np.load(path).astype(np.int8).ravel()
    assert bits.size == MARK_SIZE and set(np.unique(bits)) <= {0, 1}
    return bits


def random_mark(seed):
    """Watermark casuale riproducibile (seed intero)."""
    return np.random.default_rng(seed).integers(0, 2, MARK_SIZE).astype(np.int8)


def to_bipolar(bits):
    """0/1 -> -1/+1."""
    return 2.0 * np.asarray(bits, dtype=float) - 1.0


def mark_hash(bits):
    """Impronta del watermark (per le chiavi della cache), non il watermark."""
    return hashlib.sha256(np.asarray(bits, dtype=np.int8).tobytes()).hexdigest()[:12]


# >>> inline
def similarity(a, b):
    """Similarità coseno; 0 se uno dei due vettori è nullo."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)
# <<< inline
