"""
Fase di attacco (C15).

Uso:  python -m scripts.attack_day --targets T --detectors D --originals O [--our NOME]
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import GROUP
from ctm.attack.day import attack_day

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--our", default=GROUP)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--detectors", required=True)
    ap.add_argument("--originals", required=True)
    ap.add_argument("--out", default="attacked")
    ap.add_argument("--log", default="attack_log.csv")
    a = ap.parse_args()
    attack_day(a.our, a.targets, a.detectors, a.originals, a.out, a.log)
