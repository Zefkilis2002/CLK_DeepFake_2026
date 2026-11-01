"""
FASE DI ATTACCO (10:15 - 12:30) - gruppo clk.

Cosa fa, per ogni immagine groupB_<nomeImmagine>.bmp scaricata dal sito:
  1. carica la detection dell'altro gruppo (detection_groupB.py o .pyc);
  2. cerca l'attacco con la WPSNR piu' alta che cancella il marchio
     (attacks_clk.search_attack: solo i 6 attacchi ammessi, singoli o in catena);
  3. salva SOLO l'attacco migliore come clk_groupB_<nomeImmagine>.bmp;
  4. ricontrolla il file salvato con la loro detection (output1 = 0 e WPSNR >= 35);
  5. scrive il log per la verifica: Image, Group, WPSNR, Attack(s) with parameters
     (CSV e, se openpyxl e' installato, anche .xlsx per Excel).

REGOLA: l'originale NON viene mai usato per costruire, scegliere o regolare
l'attacco. Gli attacchi lavorano solo sull'immagine marchiata. L'originale
compare in un solo punto: come primo argomento della detection dell'altro
gruppo (dentro make_oracle), che ci dice solo "marchio presente sì/no" e la WPSNR.

Si puo' rilanciare: le immagini gia' attaccate con successo vengono saltate.
In caso di errore o lentezza di una detection, il problema viene segnalato e
si passa all'immagine successiva.

Uso (da gara_clk, con il venv attivo):
    python scripts/gara_attacco.py
    python scripts/gara_attacco.py --gruppi falco koala
"""
import argparse
import contextlib
import csv
import io
import sys
import time
from pathlib import Path

SUBMISSION_DIR = Path(__file__).resolve().parents[2] / "da_consegnare"
sys.path.insert(0, str(SUBMISSION_DIR))

import cv2

from attacks_clk import find_detection, load_detection, make_oracle, read_gray, search_attack

# --- cartelle (relative a questo file) -------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
AVVERSARI_DIR = BASE_DIR / "immagini_avversari"     # groupB_harbour.bmp, ...
DETECTION_DIR = BASE_DIR / "detection_avversari"    # detection_groupB.py / .pyc
ORIGINALI_DIR = BASE_DIR / "originali"              # harbour.bmp, ... (solo per le loro detection)
ATTACCATE_DIR = BASE_DIR / "attaccate"              # clk_groupB_harbour.bmp da caricare
LOG_NAME = "attack_log"                             # attack_log.csv / attack_log.xlsx
GROUP = "clk"

MIN_WPSNR = 35.0
HEADER = ["Image", "Group", "WPSNR", "Attack(s) with parameters"]


def activity_points(percent):
    """Punti di attivita' (tabella del regolamento)."""
    for limit, points in [(90, 6), (60, 4), (30, 2)]:
        if percent > limit:
            return points
    return 0


def read_log(csv_path):
    """Righe gia' presenti nel log (per poter rilanciare lo script)."""
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f, delimiter=";"))
    return [r for r in rows[1:] if len(r) == 4]


def write_log(csv_path, rows):
    """Riscrive il log completo in CSV (e in .xlsx se possibile)."""
    with open(csv_path, "w", newline="") as f:
        log = csv.writer(f, delimiter=";")
        log.writerow(HEADER)
        log.writerows(rows)
    try:
        from openpyxl import Workbook
    except ImportError:
        return False
    wb = Workbook()
    ws = wb.active
    ws.append(HEADER)
    for image, group, w, text in rows:
        ws.append([image, group, float(w), text])
    wb.save(str(csv_path.with_suffix(".xlsx")))
    return True


def main():
    ap = argparse.ArgumentParser(description="Fase di attacco del gruppo clk")
    ap.add_argument("--avversari", default=str(AVVERSARI_DIR), help="cartella groupB_img.bmp")
    ap.add_argument("--detection", default=str(DETECTION_DIR), help="cartella detection_groupB.py/.pyc")
    ap.add_argument("--originali", default=str(ORIGINALI_DIR), help="cartella delle 3 immagini originali")
    ap.add_argument("--uscita", default=str(ATTACCATE_DIR), help="cartella delle immagini attaccate")
    ap.add_argument("--gruppi", nargs="*", help="attacca solo questi gruppi (default: tutti)")
    a = ap.parse_args()
    targets_dir, det_dir = Path(a.avversari), Path(a.detection)
    originals_dir, out_dir = Path(a.originali), Path(a.uscita)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / (LOG_NAME + ".csv")
    tmp_path = out_dir / "_tmp_attacked.bmp"          # file temporaneo per l'oracolo

    rows = read_log(csv_path)
    done = {(r[1], r[0]) for r in rows}                # (gruppo, immagine) gia' attaccati
    detections = {}                                    # gruppo -> funzione detection
    groups_seen = set()

    for target in sorted(targets_dir.glob("*.bmp")):
        if "_" not in target.stem:
            continue
        group, image = target.stem.split("_", 1)       # groupB_harbour -> groupB, harbour
        if group == GROUP:
            continue                                   # le nostre immagini non si attaccano
        if a.gruppi and group not in a.gruppi:
            continue
        groups_seen.add(group)
        out_path = out_dir / ("%s_%s_%s.bmp" % (GROUP, group, image))   # clk_groupB_harbour.bmp
        if (group, image) in done and out_path.exists():
            print("%-22s gia' attaccata, salto" % target.name)
            continue

        # detection dell'altro gruppo (caricata una sola volta)
        if group not in detections:
            det_path = find_detection(str(det_dir), group)
            if det_path is None:
                print("%-22s detection_%s non trovata, salto" % (target.name, group))
                detections[group] = None
            else:
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        detections[group] = load_detection(det_path)
                except Exception as e:
                    print("%-22s detection_%s non si carica: %s" % (target.name, group, e))
                    detections[group] = None
        if detections[group] is None:
            continue

        orig_path = originals_dir / (image + ".bmp")
        if not orig_path.exists():
            print("%-22s originale %s non trovato, salto" % (target.name, orig_path.name))
            continue

        # l'originale entra SOLO qui, come argomento della loro detection
        oracle = make_oracle(detections[group], str(orig_path.resolve()),
                             str(target.resolve()), str(tmp_path.resolve()))
        start = time.time()
        try:
            # check iniziale: la loro detection deve vedere il marchio nell'immagine non attaccata
            t0 = time.time()
            present, _ = oracle(read_gray(str(target)))
            slow = time.time() - t0
            if not present:
                print("%-22s ATTENZIONE: la loro detection non trova il marchio nemmeno senza attacco"
                      % target.name)
            if slow > 5:
                print("%-22s ATTENZIONE: la loro detection impiega %.1f s per chiamata" % (target.name, slow))
            # ricerca: lavora solo sull'immagine marchiata
            result = search_attack(read_gray(str(target)), oracle)
        except Exception as e:
            print("%-22s errore durante la ricerca: %s" % (target.name, e))
            continue
        if result is None:
            print("%-22s nessun attacco riuscito con WPSNR >= 35 (%.0f s)" % (target.name, time.time() - start))
            continue

        attacked, _, text = result
        cv2.imwrite(str(out_path), attacked)
        # ricontrollo sul file salvato, cioe' esattamente quello che caricheremo
        try:
            present, w = oracle(read_gray(str(out_path)))
        except Exception as e:
            print("%-22s errore nel ricontrollo: %s" % (target.name, e))
            out_path.unlink()
            continue
        if present == 0 and w >= MIN_WPSNR:
            rows = [r for r in rows if (r[1], r[0]) != (group, image)]
            rows.append([image, group, "%.2f" % w, text])
            write_log(csv_path, rows)                  # salvato subito, per non perdere nulla
            print("%-22s OK  WPSNR %.2f dB | %s | %.0f s" % (target.name, w, text, time.time() - start))
        else:
            out_path.unlink()
            print("%-22s ricontrollo fallito (output1=%d, WPSNR %.2f): file scartato"
                  % (target.name, present, w))

    if tmp_path.exists():
        tmp_path.unlink()

    # ------------------------------------------------------------------
    # Riepilogo
    # ------------------------------------------------------------------
    xlsx = write_log(csv_path, rows)
    print("\n=== RIEPILOGO ATTACCHI ===")
    print("%-10s %-12s %-8s %s" % ("Gruppo", "Immagine", "WPSNR", "Attacco"))
    for image, group, w, text in sorted(rows, key=lambda r: (r[1], r[0])):
        print("%-10s %-12s %-8s %s" % (group, image, w, text))
    attacked_groups = {r[1] for r in rows}
    groups_seen |= attacked_groups
    if groups_seen:
        percent = 100.0 * len(attacked_groups) / len(groups_seen)
        print("Gruppi attaccati: %d su %d (%.0f%%) -> %d punti di attivita'"
              % (len(attacked_groups), len(groups_seen), percent, activity_points(percent)))
    print("Log: %s%s" % (csv_path, " e .xlsx" if xlsx else " (openpyxl non installato: niente .xlsx)"))
    print("Da caricare: i file clk_<gruppo>_<immagine>.bmp nella cartella %s" % out_dir)


if __name__ == "__main__":
    main()
