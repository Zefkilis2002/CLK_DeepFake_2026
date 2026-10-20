"""M1: bit-plane (LAB-3, esercizio 1). Parametro: plane (0..3)."""
import numpy as np

from ctm.schemes.base import Scheme

MARK_SIZE = 1024


class BitPlane(Scheme):
    name = "bitplane"
    default_params = {"plane": 0}

    def embed(self, ctx, bits, params):
        original, plane = ctx["original"], int(params["plane"])
        tiled = np.resize(np.asarray(bits, dtype=np.uint8), original.size)    # 256 copie
        clear = np.uint8(~(1 << plane) & 0xFF)
        flat = original.ravel()
        return ((flat & clear) | (tiled << plane)).reshape(original.shape).astype(np.uint8)

    def extract(self, ctx, image, params):
        bits = (image.ravel() >> int(params["plane"])) & 1
        return 2.0 * bits.reshape(-1, MARK_SIZE).mean(axis=0) - 1.0           # media delle copie
