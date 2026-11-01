"""
FASE DI DIFESA (8:30 - 10:00) - gruppo clk.

Cosa fa:
  1. legge le 3 immagini originali scaricate dal sito (cartella ORIGINALI_DIR);
  2. inserisce il nostro watermark clk.npy con embedding_clk.embedding;
  3. salva le immagini marchiate come clk_<nomeImmagine>.bmp (regola dei nomi)
     nella cartella DA_CARICARE_DIR, insieme a una copia di detection_clk.py;
  4. stampa la WPSNR(originale, marchiata) e i punti di qualita' previsti;
  5. controlla che la NOSTRA detection rispetti le regole (PASS/FAIL):
       a. marchiata contro se stessa                 -> 1
       b. originale usata come attaccata             -> 0
       c. qualche attacco con WPSNR >= 35            -> solo informativo
       d. immagine distrutta (WPSNR <= 25)           -> 0 (altrimenti penalita')
       e. le altre 2 immagini originali (scorrelate) -> 0
       f. stessa immagine marchiata con chiave casuale -> 0
       g. ogni chiamata dura < 5 s e non stampa nulla
     e che l'immagine riletta dal disco sia identica a quella scritta.

Uso (da gara_clk, con il venv attivo):
    python scripts/gara_difesa.py
    python scripts/gara_difesa.py --originali percorso --uscita percorso
"""
import argparse
import contextlib
import io
import shutil
import sys
import time
from pathlib import Path

SUBMISSION_DIR = Path(__file__).resolve().parents[2] / "da_consegnare"
sys.path.insert(0, str(SUBMISSION_DIR))

import cv2
import numpy as np

from attacks_clk import attacks
from detection_clk import detection, wpsnr
from embedding_clk import embedding

# --- cartelle (relative a questo file) -------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
ORIGINALI_DIR = BASE_DIR / "originali"          # qui le 3 immagini scaricate (es. harbour.bmp)
DA_CARICARE_DIR = BASE_DIR / "da_caricare"      # qui finiscono i file da caricare sul sito
PROVE_DIR = BASE_DIR / "prove_difesa"           # immagini temporanee dei controlli
WATERMARK_PATH = BASE_DIR / "clk.npy"           # il nostro watermark
GROUP = "clk"

MAX_SECONDS = 5.0                               # limite di tempo della detection
# attacchi di prova (fase c): devono restare sopra 35 dB per essere "validi"
TEST_ATTACKS = [("jpeg", [70]), ("blur", [0.6]), ("awgn", [4]), ("median", [3]),
                ("resize", [0.8]), ("sharp", [0.5])]


def quality_points(w):
    """Punti di qualita' dell'embedding (tabella del regolamento)."""
    for limit, points in [(66, 6), (62, 5), (58, 4), (54, 3), (50, 2), (35, 1)]:
        if w >= limit:
            return points
    return 0


def run_detection(original_path, watermarked_path, tested_path):
    """
    Chiama la nostra detection come faranno gli altri gruppi.
    Restituisce (output1, output2, secondi, testo_stampato).
    """
    printed = io.StringIO()
    start = time.time()
    with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):
        out1, out2 = detection(str(original_path), str(watermarked_path), str(tested_path))
    return out1, out2, time.time() - start, printed.getvalue()


def main():
    ap = argparse.ArgumentParser(description="Fase di difesa del gruppo clk")
    ap.add_argument("--originali", default=str(ORIGINALI_DIR), help="cartella con le 3 immagini originali")
    ap.add_argument("--uscita", default=str(DA_CARICARE_DIR), help="cartella dei file da caricare")
    ap.add_argument("--prove", default=str(PROVE_DIR), help="cartella per le immagini dei controlli")
    a = ap.parse_args()
    originals_dir, upload_dir, tests_dir = Path(a.originali), Path(a.uscita), Path(a.prove)
    upload_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in originals_dir.glob("*.bmp"))
    if len(images) != 3:
        print("ATTENZIONE: trovate %d immagini .bmp in %s (attese 3)" % (len(images), originals_dir))
    if not images:
        return

    failures = []          # elenco dei controlli falliti

    def check(ok, text):
        print("   [%s] %s" % ("PASS" if ok else "FAIL", text))
        if not ok:
            failures.append(text)

    # ------------------------------------------------------------------
    # 1-4. Embedding e salvataggio con il nome della gara
    # ------------------------------------------------------------------
    print("=== EMBEDDING ===")
    marked = {}                                     # immagine originale -> marchiata
    for orig_path in images:
        original = cv2.imread(str(orig_path), cv2.IMREAD_GRAYSCALE)
        check(original is not None and original.shape == (512, 512),
              "%s letta, 512x512 in scala di grigi" % orig_path.name)

        watermarked = embedding(str(orig_path), str(WATERMARK_PATH))
        out_path = upload_dir / ("%s_%s" % (GROUP, orig_path.name))     # clk_harbour.bmp
        cv2.imwrite(str(out_path), watermarked)

        # l'immagine riletta dal disco deve essere identica a quella scritta
        reread = cv2.imread(str(out_path), cv2.IMREAD_GRAYSCALE)
        check(reread is not None and np.array_equal(reread, watermarked),
              "%s salvata e riletta identica" % out_path.name)

        w = wpsnr(original, reread)
        print("   %s: WPSNR = %.2f dB -> %d punti di qualita'" % (out_path.name, w, quality_points(w)))
        check(w >= 35, "%s: WPSNR >= 35 dB" % out_path.name)
        marked[orig_path] = out_path

    # copia della detection da caricare (stesso file consegnato)
    shutil.copy(SUBMISSION_DIR / "detection_clk.py", upload_dir / "detection_clk.py")

    # ------------------------------------------------------------------
    # 5. Controlli sulla nostra detection
    # ------------------------------------------------------------------
    rng = np.random.default_rng()
    times = []
    for orig_path, wm_path in marked.items():
        print("\n=== CONTROLLI DETECTION: %s ===" % wm_path.name)
        stem = orig_path.stem

        def test(text, tested_path, expected):
            out1, out2, sec, printed = run_detection(orig_path, wm_path, tested_path)
            times.append(sec)
            if expected is None:          # solo informativo
                print("   [INFO] %s -> output1=%d, WPSNR=%.2f dB" % (text, out1, out2))
            else:
                check(out1 == expected, "%s -> output1=%d (atteso %d), WPSNR=%.2f dB"
                      % (text, out1, expected, out2))
            check(sec < MAX_SECONDS, "%s: tempo %.2f s < %.0f s" % (text, sec, MAX_SECONDS))
            check(printed == "", "%s: la detection non stampa nulla" % text)
            check(out1 in (0, 1) and isinstance(out2, float), "%s: output (0/1, float)" % text)
            return out1, out2

        # a. marchiata contro se stessa
        test("a. marchiata come attaccata", wm_path, 1)

        # b. originale usata come attaccata
        test("b. originale come attaccata", orig_path, 0)

        # c. attacchi leggeri (informativo: dice quanto siamo robusti)
        for name, params in TEST_ATTACKS:
            att_path = tests_dir / ("%s_%s.bmp" % (stem, name))
            cv2.imwrite(str(att_path), attacks(str(wm_path), name, params))
            test("c. attacco %s %s" % (name, params), att_path, None)

        # d. immagine distrutta: rumore fortissimo, WPSNR <= 25 -> deve dare 0
        destroyed_path = tests_dir / ("%s_distrutta.bmp" % stem)
        cv2.imwrite(str(destroyed_path), attacks(str(wm_path), "awgn", [100]))
        _, w_destroyed = test("d. immagine distrutta (awgn 100)", destroyed_path, 0)
        check(w_destroyed <= 25, "d. l'immagine distrutta ha davvero WPSNR <= 25")

        # e. le altre immagini originali (contenuto diverso)
        for other in images:
            if other != orig_path:
                test("e. immagine scorrelata %s" % other.name, other, 0)

        # f. stessa immagine con un watermark casuale (come un altro gruppo)
        key_path = tests_dir / ("%s_chiave_casuale.npy" % stem)
        np.save(key_path, rng.integers(0, 2, 1024))
        random_path = tests_dir / ("%s_chiave_casuale.bmp" % stem)
        cv2.imwrite(str(random_path), embedding(str(orig_path), str(key_path)))
        test("f. stessa immagine, chiave casuale", random_path, 0)

    # ------------------------------------------------------------------
    # Riepilogo
    # ------------------------------------------------------------------
    print("\n=== RIEPILOGO ===")
    print("Tempo detection: massimo %.2f s, medio %.2f s" % (max(times), np.mean(times)))
    if failures:
        print("CI SONO %d CONTROLLI FALLITI: NON caricare finche' non sono risolti!" % len(failures))
        for f in failures:
            print("  -", f)
    else:
        print("TUTTI I CONTROLLI PASSATI.")
    print("File da caricare sul sito (cartella %s):" % upload_dir)
    for wm_path in marked.values():
        print("  -", wm_path.name)
    print("  - detection_clk.py")


if __name__ == "__main__":
    main()
