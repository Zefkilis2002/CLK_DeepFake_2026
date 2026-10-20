"""
Attacchi del gruppo {{GROUP}} (consegna; usa il package ctm/ copiato accanto).

    from attacks import attacks
    img = attacks("gruppob_lena.bmp", ["blur", "jpeg"], [1.0, 70])

Nomi: 'awgn' (sigma), 'blur' (sigma), 'sharp' (alpha), 'jpeg' (QF),
'resize' (scala), 'median' (k). Solo l'immagine marchiata viene usata.

Giorno della gara:
    python attacks.py --targets T --detectors D --originals O
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ctm.attack.chains import attacks          # noqa: F401  (funzione delle regole)
from ctm.attack.day import attack_day

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--our", default="{{GROUP}}")
    ap.add_argument("--targets", required=True)
    ap.add_argument("--detectors", required=True)
    ap.add_argument("--originals", required=True)
    ap.add_argument("--out", default="attacked")
    ap.add_argument("--log", default="attack_log.csv")
    a = ap.parse_args()
    attack_day(a.our, a.targets, a.detectors, a.originals, a.out, a.log)
