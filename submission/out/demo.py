"""
Demo (consegna, GENERATA dal preset P66): embedding, detection, attacchi, ricerca.

Uso:  python demo.py <immagine_originale.bmp> <watermark.npy>
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2

from attacks import attacks
from ctm.attack.oracle import FileOracle
from ctm.attack.search import search_attack
from ctm.common.wpsnr import wpsnr
from detection_groupname import detection
from embedding import embedding

if __name__ == "__main__":
    original_path, mark_path = os.path.abspath(sys.argv[1]), sys.argv[2]
    name = os.path.splitext(os.path.basename(original_path))[0]
    wm = embedding(original_path, mark_path)
    wm_path = os.path.abspath("groupname_%s.bmp" % name)
    cv2.imwrite(wm_path, wm)
    print("WPSNR embedding: %.2f dB -> %s" % (wpsnr(cv2.imread(original_path, 0), wm), wm_path))
    print("marchiata:", detection(original_path, wm_path, wm_path))
    print("originale:", detection(original_path, wm_path, original_path))
    tmp = os.path.abspath("_demo_attacked.bmp")
    for names, params in [("jpeg", [50]), ("blur", [1.0]), ("awgn", [10]), ("median", [5]),
                          (["blur", "jpeg"], [1.0, 70])]:
        cv2.imwrite(tmp, attacks(wm_path, names, params))
        print("  %-25s -> %s" % ("%s %s" % (names, params), detection(original_path, wm_path, tmp)))
    best = search_attack(wm, FileOracle(detection, original_path, wm_path, tmp))
    print("ricerca:", "nessun attacco riuscito" if best is None else "%.2f dB con %s" % (best["wpsnr"], best["text"]))
    os.remove(tmp)
