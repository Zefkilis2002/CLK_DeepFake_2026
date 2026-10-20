"""
C14: controlli di conformità sui file GENERATI in submission/out/.

Uso:  python -m submission.checks [--group nome] [--n 20]
"""
import argparse
import contextlib
import inspect
import io
import os
import subprocess
import sys
import tempfile
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from config.settings import GROUP, MARK_PATH, RESULTS_DIR, SUBMISSION_OUT, load_split
from ctm.attack import chains, search
from ctm.attack.oracle import load_detection
from ctm.common.attacks import awgn, blur, jpeg, sharpen
from ctm.common.imgio import read_gray
from ctm.common.watermark import random_mark
from ctm.common.wpsnr import wpsnr, wpsnr_lab
from ctm.schemes import get_scheme

failures = []


def check(ok, text):
    print(("  OK     " if ok else "  ERRORE ") + text)
    if not ok:
        failures.append(text)


def silent(fn, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        result = fn(*args)
    return result, out.getvalue()


def load_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(group, n):
    det_path = os.path.join(SUBMISSION_OUT, "detection_%s.py" % group)
    emb_path = os.path.join(SUBMISSION_OUT, "embedding.py")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    p_wm, p_att = os.path.join(RESULTS_DIR, "_chk_wm.bmp"), os.path.join(RESULTS_DIR, "_chk_att.bmp")
    paths = load_split()[1][:n]

    print("1. Nomi e file")
    check(os.path.exists(det_path), "esiste detection_%s.py" % group)
    check(group == group.lower() and group.isalnum(), "nome del gruppo senza spazi, simboli o maiuscole")

    print("2. Detection come file unico")
    src = open(det_path, encoding="utf-8").read()
    check("np.load" not in src and ".npy" not in src, "non legge il file del watermark")
    check("open(" not in src and "./" not in src and "..\\" not in src, "nessun percorso relativo o file aperto")
    imports = [l.split()[1] for l in src.splitlines() if l.startswith(("import ", "from "))]
    check(all(m.split(".")[0] in ("cv2", "numpy", "scipy") for m in imports), "importa solo cv2/numpy/scipy")
    # interprete pulito (-I), cartella temporanea, senza il repository nel path
    code = ("import importlib.util as u; s=u.spec_from_file_location('d', r'%s'); m=u.module_from_spec(s); "
            "s.loader.exec_module(m); print(m.detection(r'%s', r'%s', r'%s'))")
    orig0 = paths[0]
    res = subprocess.run([sys.executable, "-I", "-c", code % (det_path, orig0, orig0, orig0)],
                         capture_output=True, text=True, cwd=tempfile.gettempdir())
    check(res.returncode == 0 and res.stdout.strip().startswith("(0,"),
          "funziona in un interprete pulito fuori dal repository (%s)" % (res.stdout.strip() or res.stderr.strip()[-80:]))

    print("3. Embedding generato")
    embedding = load_module(emb_path).embedding
    detection = load_detection(det_path)
    wm, printed = silent(embedding, paths[0], MARK_PATH)
    check(wm.dtype == np.uint8 and wm.shape == (512, 512) and printed == "", "uint8 512x512, nessuna stampa")
    check(abs(wpsnr(read_gray(paths[0]), wm) - wpsnr_lab(read_gray(paths[0]), wm)) < 1e-9, "WPSNR veloce = WPSNR del lab")

    print("4. Detection su %d immagini di verifica" % len(paths))
    counts, times, printed_all, emb_w, destroyed_w = {}, [], "", [], []
    scheme = get_scheme("dct_ss")
    det_mod = load_module(det_path)
    params = {"alpha": det_mod.ALPHA, "mode": det_mod.MODE, "n": det_mod.N}

    def run(kind, expected, attacked, path):
        nonlocal printed_all
        cv2.imwrite(p_att, attacked)
        t = time.time()
        (present, w), printed = silent(detection, path, p_wm, p_att)
        times.append(time.time() - t)
        printed_all += printed
        good, total = counts.get(kind, (0, 0))
        counts[kind] = (good + int(present == expected), total + 1)
        return w

    for i, path in enumerate(paths):
        orig = read_gray(path)
        wm = embedding(path, MARK_PATH)
        emb_w.append(wpsnr(orig, wm))
        cv2.imwrite(p_wm, wm)
        run("marchiata", 1, wm, path)
        run("originale", 0, orig, path)
        for k in range(5):
            run("altro watermark", 0, scheme.embed_image(orig, random_mark(1000 * i + k), params), path)
        run("altra immagine", 0, read_gray(paths[(i + 1) % len(paths)]), path)
        for destroyed in (awgn(wm, 150), sharpen(wm, 20, 2), awgn(sharpen(wm, 10, 1), 80)):
            destroyed_w.append(run("distrutta", 0, destroyed, path))
        for light in (jpeg(wm, 90), awgn(wm, 2), blur(wm, 0.5)):
            run("attacco leggero", 1, light, path)
    for kind, (good, total) in counts.items():
        check(good == total, "%-16s risultato atteso %d/%d" % (kind, good, total))
    check(printed_all == "", "nessuna stampa")
    check(max(times) < 5, "tempo massimo %.2f s (limite 5 s)" % max(times))
    print("    WPSNR embedding medio %.2f, minimo %.2f | distrutte fino a %.1f dB"
          % (np.mean(emb_w), np.min(emb_w), np.max(destroyed_w)))

    print("5. Attacchi")
    for fn in (chains.attacks, chains.apply_attacks, search.search_attack):
        args = list(inspect.signature(fn).parameters)
        check(not any("orig" in a for a in args), "%s%s non riceve l'originale" % (fn.__name__, tuple(args)))

    for p in (p_wm, p_att):
        if os.path.exists(p):
            os.remove(p)
    print("\nTUTTO OK" if not failures else "\n%d CONTROLLI FALLITI" % len(failures))
    return not failures


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default=GROUP)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    sys.exit(0 if main(a.group, a.n) else 1)
