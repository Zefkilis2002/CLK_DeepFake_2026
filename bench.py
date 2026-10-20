"""
Banco di prova locale (C13 del piano, semplificato).

Lancia la nostra ricerca di attacchi (attacks.search_attack) contro:
  - il NOSTRO detector (detection_groupname.py, con ALPHA e TAU veri):
    stima di quanto siamo robusti, cioè con che WPSNR ci battono;
  - due "difensori finti" costruiti con gli schemi dei laboratori
    (bit-plane e SS moltiplicativo del LAB-3, DWT-HVS del LAB-4), ognuno con la sua tau:
    servono a controllare che la ricerca di attacchi funzioni.
Atteso: bit-plane si rompe a WPSNR alto, il nostro schema a WPSNR basso.

Uso:  python bench.py            (5 immagini di verifica)
      python bench.py 10         (10 immagini)
"""
import os
import sys
import time

import cv2
import numpy as np

from attacks import make_oracle, search_attack
from common import CTM_DIR, list_images, load_mark, read_gray, similarity, wpsnr
import detection_groupname as det
import embedding
import roc
from schemes import SCHEMES

FIRST_IMAGE = 80          # immagini non usate dalla ROC (0-59)

# punti robustezza della gara: WPSNR medio delle immagini rotte -> punti
ROBUSTNESS = [(53, 0), (50, 1), (47, 2), (44, 3), (41, 4), (38, 5), (35, 6)]


def robustness_points(w):
    for limit, points in ROBUSTNESS:
        if w >= limit:
            return points
    return 6


def local_oracle(name, param, tau, orig, wm):
    """Detector in memoria per uno schema dei laboratori."""
    _, extract = SCHEMES[name]
    w_ext = extract(orig, wm, param)

    def oracle(attacked):
        present = int(similarity(w_ext, extract(orig, attacked, param)) >= tau)
        return present, wpsnr(wm, attacked)
    return oracle


def counted(oracle):
    """Conta quante volte viene interrogato l'oracolo."""
    def wrapper(img):
        wrapper.queries += 1
        return oracle(img)
    wrapper.queries = 0
    return wrapper


def bench(label, paths, make):
    bits = load_mark()
    results, queries, t0 = [], 0, time.time()
    for path in paths:
        wm, oracle = make(path, bits)
        oracle = counted(oracle)
        best = search_attack(wm, oracle)
        queries += oracle.queries
        if best is None:
            print("  %s: non rotto sopra 35 dB" % os.path.basename(path))
        else:
            print("  %s: rotto a %.2f dB con %s" % (os.path.basename(path), best[1], best[2]))
            results.append(best[1])
    print("%s: rotti %d/%d | WPSNR medio %.2f | migliore %.2f | query %d | %.0f s" %
          (label, len(results), len(paths), np.mean(results) if results else 0,
           max(results) if results else 0, queries, time.time() - t0))
    if results:
        print("  -> punti robustezza stimati: %d" % robustness_points(np.mean(results)))
    print()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    paths = list_images()[FIRST_IMAGE:FIRST_IMAGE + n]
    tmp_wm = os.path.join(CTM_DIR, "results", "_bench_wm.bmp")
    tmp_att = os.path.join(CTM_DIR, "results", "_bench_att.bmp")
    os.makedirs(os.path.dirname(tmp_wm), exist_ok=True)

    # 1. il nostro detector vero, passando dai file come il giorno della gara
    def make_ours(path, bits):
        wm = embedding.embedding(path, os.path.join(os.path.dirname(CTM_DIR), "clk.npy"))
        cv2.imwrite(tmp_wm, wm)
        oracle = make_oracle(det.detection, path, tmp_wm, tmp_att)
        return wm, oracle

    print("Nostro schema: DCT-SS alpha=%g tau=%.4f" % (embedding.ALPHA, det.TAU))
    bench("nostro", paths, make_ours)

    # 2. difensori finti dai laboratori
    for name, param in (("bitplane", 1), ("dct_mult", 0.007), ("dwt_hvs", 0.75)):
        embed, extract = SCHEMES[name]
        print("Calcolo tau per %s(%g) ..." % (name, param))
        tau = roc.run(embed, extract, param, n_calib=10, n_test=5, name=name, plot=False)

        def make_local(path, bits, name=name, param=param, tau=tau, embed=embed):
            orig = read_gray(path)
            wm = embed(orig, bits, param)
            oracle = local_oracle(name, param, tau, orig, wm)
            return wm, oracle
        bench("%s(%g)" % (name, param), paths, make_local)


if __name__ == "__main__":
    main()
