"""
Demo: shows how to use embedding, detection and attacks (required by the rules).

  1. embedding of the clk.npy watermark into a test image
  2. saving with the competition name: groupname_<image>.bmp
  3. detection on the watermarked image (expected 1) and on the original (expected 0)
  4. some single attacks with attacks() and detection on each one
  5. automatic search for the best attack against our detector

Usage:  python demo.py            (image 0000.bmp)
        python demo.py 0042       (another image from images/)
"""
import os
import sys

import cv2

from attacks import attacks, make_oracle, search_attack
from common import IMAGES_DIR, MARK_PATH, read_gray, wpsnr
from detection_groupname import detection
from embedding import embedding

GROUP = "groupname"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def main():
    image = sys.argv[1] if len(sys.argv) > 1 else "0000"
    original_path = os.path.join(IMAGES_DIR, image + ".bmp")
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1-2. embedding and saving
    watermarked = embedding(original_path, MARK_PATH)
    wm_path = os.path.join(OUT_DIR, "%s_%s.bmp" % (GROUP, image))
    cv2.imwrite(wm_path, watermarked)
    print("Embedding: WPSNR(originale, marchiata) = %.2f dB -> %s"
          % (wpsnr(read_gray(original_path), watermarked), wm_path))

    # 3. detection without attack
    print("Detection marchiata: ", detection(original_path, wm_path, wm_path))
    print("Detection originale: ", detection(original_path, wm_path, original_path))

    # 4. single and combined attacks
    att_path = os.path.join(OUT_DIR, "_demo_attacked.bmp")
    tests = [("jpeg", [50]), ("blur", [1.0]), ("awgn", [10]), ("median", [5]),
             ("resize", [0.5]), ("sharp", [2.0]), (["blur", "jpeg"], [1.0, 70])]
    for names, params in tests:
        attacked = attacks(wm_path, names, params)
        cv2.imwrite(att_path, attacked)
        present, w = detection(original_path, wm_path, att_path)
        print("  %-20s presente=%d  WPSNR=%.2f" % ("%s %s" % (names, params), present, w))

    # 5. automatic search (as we will do against the other groups)
    oracle = make_oracle(detection, original_path, wm_path, att_path)
    best = search_attack(watermarked, oracle)
    if best is None:
        print("Ricerca: nessun attacco cancella il marchio con WPSNR >= 35")
    else:
        print("Ricerca: attacco migliore %s con WPSNR %.2f dB" % (best[2], best[1]))
    os.remove(att_path)


if __name__ == "__main__":
    main()
