"""
Controlli di conformità alle regole della gara (C14 del piano, semplificato).
Da lanciare ogni volta che si cambiano ALPHA o TAU.

Controlla:
  1. WPSNR veloce identico a quello del laboratorio
  2. file di detection: niente lettura del watermark, niente percorsi relativi,
     niente import di file nostri
  3. embedding: uint8 512x512, nessuna stampa
  4. detection, su immagini mai usate dalla ROC:
       marchiata -> 1, originale -> 0, altri watermark -> 0, altre immagini -> 0,
       immagini distrutte (WPSNR <= 25) -> 0, attacchi leggeri -> 1,
       nessuna stampa, tempo < 5 s
  5. le funzioni di attacco non ricevono l'originale

Uso:  python check_rules.py
"""
import contextlib
import inspect
import io
import os
import time

import cv2
import numpy as np

import attacks
from common import (CTM_DIR, MARK_PATH, awgn, blur, jpeg, list_images, random_mark, sharpen,
                    read_gray, wpsnr, wpsnr_lab)
import embedding

DETECTION_FILE = os.path.join(CTM_DIR, "detection_groupname.py")
TEST_IMAGES = list_images()[60:80]        # non usate dalla ROC
OTHER_MARKS = 5
TMP = os.path.join(CTM_DIR, "results")

failures = []


def check(ok, text):
    print(("  OK    " if ok else "  ERRORE ") + text)
    if not ok:
        failures.append(text)


def silent_call(fn, *args):
    """Esegue fn catturando stdout e stderr; restituisce (risultato, testo stampato)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        result = fn(*args)
    return result, out.getvalue() + err.getvalue()


def main():
    os.makedirs(TMP, exist_ok=True)
    p_wm, p_att = os.path.join(TMP, "_chk_wm.bmp"), os.path.join(TMP, "_chk_att.bmp")
    detection = attacks.load_detection(DETECTION_FILE)     # caricata come la caricano gli altri

    print("1. WPSNR")
    a = read_gray(TEST_IMAGES[0])
    for b in (awgn(a, 3), jpeg(a, 50), blur(a, 1.0)):
        check(abs(wpsnr(a, b) - wpsnr_lab(a, b)) < 1e-9, "WPSNR veloce = WPSNR del lab (%.4f dB)" % wpsnr(a, b))

    print("2. File di detection")
    src = open(DETECTION_FILE, encoding="utf-8").read()
    check("np.load" not in src and ".npy" not in src, "non legge il file del watermark")
    check("./" not in src and "..\\" not in src and "open(" not in src, "nessun percorso relativo o file aperto")
    imports = [l.split()[1] for l in src.splitlines() if l.startswith(("import ", "from "))]
    check(all(m.split(".")[0] in ("cv2", "numpy", "scipy") for m in imports),
          "importa solo cv2, numpy, scipy (%s)" % ", ".join(imports))

    print("3. Embedding")
    wm, printed = silent_call(embedding.embedding, TEST_IMAGES[0], MARK_PATH)
    check(wm.dtype == np.uint8 and wm.shape == (512, 512), "restituisce uint8 512x512")
    check(printed == "", "non stampa nulla")

    print("4. Detection su %d immagini" % len(TEST_IMAGES))
    rng = np.random.default_rng(7)
    counts = {k: [0, 0] for k in ("marchiata", "originale", "altro watermark", "altra immagine",
                                  "distrutta", "attacco leggero")}
    times, printed_any, emb_w, destroyed_w = [], "", [], []

    def run(name, expected, attacked):
        cv2.imwrite(p_att, attacked)
        t = time.time()
        (present, w), printed = silent_call(detection, path, p_wm, p_att)
        times.append(time.time() - t)
        counts[name][0] += int(present == expected)
        counts[name][1] += 1
        return w, printed

    for i, path in enumerate(TEST_IMAGES):
        orig = read_gray(path)
        wm = embedding.embedding(path, MARK_PATH)
        emb_w.append(wpsnr(orig, wm))
        cv2.imwrite(p_wm, wm)
        printed_any += run("marchiata", 1, wm)[1]
        run("originale", 0, orig)
        for _ in range(OTHER_MARKS):
            run("altro watermark", 0, embedding.embed_dct_ss(orig, random_mark(rng)))
        run("altra immagine", 0, read_gray(TEST_IMAGES[(i + 1) % len(TEST_IMAGES)]))
        # attacchi che portano davvero il WPSNR sotto 25 dB (regola delle penalità)
        for destroyed in (awgn(wm, 150), sharpen(wm, 20, 2), awgn(sharpen(wm, 10, 1), 80)):
            w, _ = run("distrutta", 0, destroyed)
            destroyed_w.append(w)
        for light in (jpeg(wm, 90), awgn(wm, 2), blur(wm, 0.5)):
            run("attacco leggero", 1, light)

    for name, (good, total) in counts.items():
        check(good == total, "%-16s risultato atteso %d/%d" % (name, good, total))
    check(printed_any == "", "la detection non stampa nulla")
    check(max(times) < 5, "tempo massimo %.2f s (medio %.2f s), limite 5 s" % (max(times), np.mean(times)))
    print("    WPSNR embedding: medio %.2f, minimo %.2f dB" % (np.mean(emb_w), np.min(emb_w)))
    print("    WPSNR immagini distrutte: massimo %.2f dB" % np.max(destroyed_w))

    print("5. Attacchi")
    for fn in (attacks.attacks, attacks.search_attack, attacks.apply_attacks):
        args = list(inspect.signature(fn).parameters)
        check(not any("orig" in a for a in args), "%s%s non riceve l'originale" % (fn.__name__, tuple(args)))

    for p in (p_wm, p_att):
        if os.path.exists(p):
            os.remove(p)
    print("\nTUTTO OK" if not failures else "\n%d CONTROLLI FALLITI" % len(failures))


if __name__ == "__main__":
    main()
