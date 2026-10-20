"""
Esperimento E0 del piano: qualità dell'embedding dopo l'arrotondamento a uint8.

Per ogni schema e parametro, su alcune immagini di prova misura:
  - WPSNR(originale, marchiata): media, minimo, 10° percentile
  - frazione di pixel cambiati
  - similarità del marchio estratto dalla marchiata pulita con quello vero

Uso:  python e0_quality.py
"""
import numpy as np

from common import list_images, load_mark, read_gray, similarity, to_bipolar, wpsnr
from schemes import SCHEMES

N_IMAGES = 10
GRID = {
    "bitplane": [0, 1, 2, 3],
    "dct_ss": [3, 4, 5, 6, 8],
    "dct_mult": [0.005, 0.007, 0.01, 0.015],
    "dwt_hvs": [0.5, 0.75, 1.0, 1.5, 2.0],
}


def main():
    bits = load_mark()
    paths = list_images()[:N_IMAGES]
    originals = [read_gray(p) for p in paths]

    print("%-9s %7s | %7s %7s %7s | %8s | %6s" %
          ("schema", "param", "WPSNRm", "min", "p10", "pix camb", "sim"))
    for name, params in GRID.items():
        embed, extract = SCHEMES[name]
        for p in params:
            w, changed, sims = [], [], []
            for orig in originals:
                wm = embed(orig, bits, p)
                w.append(wpsnr(orig, wm))
                changed.append(np.mean(wm != orig))
                sims.append(similarity(to_bipolar(bits), extract(orig, wm, p)))
            print("%-9s %7g | %7.2f %7.2f %7.2f | %7.1f%% | %6.3f" %
                  (name, p, np.mean(w), np.min(w), np.percentile(w, 10),
                   100 * np.mean(changed), np.mean(sims)))


if __name__ == "__main__":
    main()
