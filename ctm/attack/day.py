"""
C15: fase di attacco del giorno della gara.

Cartelle:
  targets    immagini degli altri gruppi, nome gruppoB_immagine.bmp
  detectors  detection_gruppoB.py o .pyc
  originals  le immagini originali della gara (immagine.bmp): usate SOLO dall'oracolo

Fase A (ampiezza): una passata veloce su tutti (2 passi di bisezione) per assicurare
i punti attività. Fase B (profondità): ricerca completa, prima sulle immagini con il
WPSNR più basso. Per ogni immagine si tiene solo il miglior attacco verificato.
"""
import os

from ctm.attack.log import log_attack
from ctm.attack.oracle import FileOracle, load_detection
from ctm.attack.search import search_attack
from ctm.common.imgio import read_gray, write_bmp


def find_detection(folder, group):
    for ext in (".py", ".pyc"):
        path = os.path.join(folder, "detection_%s%s" % (group, ext))
        if os.path.exists(path):
            return path
    return None


def list_targets(our, targets, detectors, originals):
    """Lista di (gruppo, immagine, percorso marchiata, percorso originale, detection)."""
    found = []
    for fname in sorted(os.listdir(targets)):
        if not fname.lower().endswith(".bmp") or "_" not in fname:
            continue
        group, image = fname[:-4].split("_", 1)
        det = find_detection(detectors, group)
        if group == our or det is None:
            continue
        found.append((group, image, os.path.abspath(os.path.join(targets, fname)),
                      os.path.abspath(os.path.join(originals, image + ".bmp")), det))
    return found


def attack_one(our, target, out_dir, bisect_steps):
    group, image, wm_path, orig_path, det_path = target
    tmp = os.path.abspath(os.path.join(out_dir, "_tmp_%s_%s.bmp" % (group, image)))
    oracle = FileOracle(load_detection(det_path), orig_path, wm_path, tmp)
    best = search_attack(read_gray(wm_path), oracle, bisect_steps=bisect_steps)
    result = None
    if best is not None:
        out_path = os.path.join(out_dir, "%s_%s_%s.bmp" % (our, group, image))
        write_bmp(out_path, best["image"])
        present, w = oracle.query(read_gray(out_path))          # verifica sul file salvato
        if present == 0 and w >= 35:
            result = dict(best, wpsnr=w, path=out_path)
    if os.path.exists(tmp):
        os.remove(tmp)
    return result


def attack_day(our, targets, detectors, originals, out_dir="attacked", log_path="attack_log.csv"):
    os.makedirs(out_dir, exist_ok=True)
    todo = list_targets(our, targets, detectors, originals)
    best = {}
    for phase, steps in (("A", 2), ("B", 4)):
        order = sorted(todo, key=lambda t: best.get(t[:2], {}).get("wpsnr", 0))  # peggiori prima
        for t in order:
            r = attack_one(our, t, out_dir, steps)
            old = best.get(t[:2])
            if r is not None and (old is None or r["wpsnr"] > old["wpsnr"]):
                best[t[:2]] = r
                print("fase %s %s_%s: %.2f dB con %s" % (phase, t[0], t[1], r["wpsnr"], r["text"]))
            elif r is None and old is None:
                print("fase %s %s_%s: nessun attacco riuscito" % (phase, t[0], t[1]))
            if old is not None and (r is None or r["wpsnr"] <= old["wpsnr"]):
                write_bmp(old["path"], old["image"])            # ripristina il migliore
    for (group, image), r in best.items():
        log_attack(log_path, image, group, r["wpsnr"], r["text"], r["chain"], r["s"])
    return best
