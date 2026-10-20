"""
Fase di difesa (C15).

  1. embedding delle immagini della gara con ogni preset (dal WPSNR più alto)
  2. sceglie il preset più alto in cui TUTTE le immagini restano nella fascia di punti dell'obiettivo
  3. salva NOME_immagine.bmp, genera i file di consegna da quel preset e lancia i controlli

Uso:  python -m scripts.defense_day --images cartella_immagini_gara [--group nome]
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import GROUP, load_presets, save_presets
from ctm.bench.run import embedding_points
from ctm.common.imgio import list_images, read_gray, write_bmp
from ctm.common.watermark import load_mark
from ctm.common.wpsnr import wpsnr
from ctm.schemes import get_scheme
from submission import checks
from submission.build_submission import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--group", default=GROUP)
    ap.add_argument("--out", default="defense_out")
    a = ap.parse_args()

    data, bits = load_presets(), load_mark()
    paths = list_images(a.images)
    originals = {p: read_gray(p) for p in paths}
    chosen = None
    for name, preset in sorted(data["presets"].items(), key=lambda kv: -kv[1]["target_wpsnr"]):
        if preset["scheme"] != "dct_ss":             # i template esistono solo per dct_ss
            print("%s: saltato (schema %s senza template di consegna)" % (name, preset["scheme"]))
            continue
        scheme = get_scheme(preset["scheme"])
        marked = {p: scheme.embed_image(o, bits, preset["params"]) for p, o in originals.items()}
        w = {p: wpsnr(originals[p], m) for p, m in marked.items()}
        need = embedding_points(preset["target_wpsnr"])
        ok = all(embedding_points(x) >= need for x in w.values())
        print("%s: %s -> %s" % (name, ", ".join("%.2f" % x for x in w.values()), "OK" if ok else "fuori fascia"))
        if ok:
            chosen = (name, marked)
            break
    if chosen is None:
        print("Nessun preset è nella sua fascia su tutte le immagini: scegliere a mano.")
        return

    name, marked = chosen
    for p, m in marked.items():
        write_bmp(os.path.join(a.out, "%s_%s" % (a.group, os.path.basename(p))), m)
    data["active"] = name
    save_presets(data)
    build(name, a.group)
    print("Preset attivo: %s. Immagini in %s, consegna in submission/out/." % (name, a.out))
    checks.main(a.group, 10)


if __name__ == "__main__":
    main()
