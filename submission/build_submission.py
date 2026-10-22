"""
C14: generatore dei file di consegna a partire da un preset.

  - legge il preset (attivo o indicato) da config/presets.json
  - riempie i template in submission/templates/ ({{ALPHA}}, {{TAU}}, ...)
  - copia DENTRO embedding e detection il codice di WPSNR, schema e similarità
    (blocchi tra "# >>> inline" e "# <<< inline" dei moduli del package),
    così la detection resta un file unico mentre il repository resta modulare
  - copia il package ctm/ e config/ per roc, attacchi e demo
Risultato in submission/out/ (mai modificato a mano).

Uso:  python -m submission.build_submission [--preset P64] [--group nome]
"""
import argparse
import os
import shutil
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import GROUP, REPO_DIR, SUBMISSION_OUT, active_preset

TEMPLATES = os.path.join(REPO_DIR, "submission", "templates")
SOURCES = os.path.join(REPO_DIR, "ctm")


def inline_block(relative_path):
    """Testo tra i marcatori '# >>> inline' e '# <<< inline' di un modulo."""
    with open(os.path.join(SOURCES, relative_path), encoding="utf-8") as f:
        text = f.read()
    start = text.index("# >>> inline") + len("# >>> inline")
    return text[start:text.index("# <<< inline")].strip("\n")


def fill(template_name, values):
    with open(os.path.join(TEMPLATES, template_name), encoding="utf-8") as f:
        text = f.read()
    for key, value in values.items():
        text = text.replace("{{%s}}" % key, str(value))
    assert "{{" not in text, "segnaposto non sostituito in " + template_name
    return text


def build(preset_name=None, group=GROUP, out_dir=SUBMISSION_OUT):
    name, preset = active_preset(preset_name)
    if preset["scheme"] != "dct_ss":
        raise ValueError("i template esistono solo per lo schema dct_ss")
    p = preset["params"]
    values = {"GROUP": group, "PRESET": name, "SCHEME": preset["scheme"], "PARAMS": repr(p),
              "ALPHA": repr(float(p["alpha"])), "MODE": p["mode"], "N": int(p["n"]),
              "TAU": repr(float(preset["tau"])),
              "WPSNR_CODE": inline_block(os.path.join("common", "wpsnr_lab.py")),
              "SCHEME_CODE": inline_block(os.path.join("schemes", "dct_ss.py")),
              "SIMILARITY_CODE": inline_block(os.path.join("common", "watermark.py"))}

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    outputs = {"embedding.py.tpl": "embedding.py",
               "detection.py.tpl": "detection_%s.py" % group,
               "roc.py.tpl": "roc.py", "attacks.py.tpl": "attacks.py",
               "demo.py.tpl": "demo.py", "README.md.tpl": "README.md"}
    for tpl, target in outputs.items():
        with open(os.path.join(out_dir, target), "w", encoding="utf-8") as f:
            f.write(fill(tpl, values))

    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(SOURCES, os.path.join(out_dir, "ctm"), ignore=ignore)
    os.makedirs(os.path.join(out_dir, "config"))
    for f in ("__init__.py", "settings.py", "presets.json"):
        shutil.copy(os.path.join(REPO_DIR, "config", f), os.path.join(out_dir, "config", f))
    return name, out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=None)
    ap.add_argument("--group", default=GROUP)
    a = ap.parse_args()
    name, out = build(a.preset, a.group)
    print("preset %s -> %s" % (name, out))
    for f in sorted(os.listdir(out)):
        print("  ", f)
