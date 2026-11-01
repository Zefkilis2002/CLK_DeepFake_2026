"""
PROVA GENERALE della giornata di gara sui dati simulati (challenge_sim).

Copia gli script, i moduli da_consegnare e il watermark in una cartella temporanea, mette le immagini e le
detection di challenge_sim nelle cartelle previste e lancia, nell'ordine,
gara_difesa.py e gara_attacco.py. I file veri di gara_clk non vengono toccati.

I dati simulati si prendono:
  - dalla cartella indicata con --sim (quella che contiene defense/ e attack/), oppure
  - se --sim non e' indicato, dalla branch challenge_sim del repository ../ctm
    (con "git archive", senza cambiare branch).

Uso (da gara_clk):
    python scripts/gara_prova.py
    python scripts/gara_prova.py --sim percorso/challenge_sim --gruppi falco koala lynx
"""
import argparse
import io
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
REPO_DIR = BASE_DIR.parent / "ctm"            # repository con la branch challenge_sim
SUBMISSION_DIR = BASE_DIR.parent / "da_consegnare"
SIM_BRANCH = "challenge_sim"
SCRIPTS = ["gara_difesa.py", "gara_attacco.py"]
MODULES = ["embedding_clk.py", "detection_clk.py", "attacks_clk.py"]


def extract_sim(dest):
    """Estrae la cartella challenge_sim dalla branch challenge_sim (senza checkout)."""
    data = subprocess.run(["git", "-C", str(REPO_DIR), "archive", SIM_BRANCH, "challenge_sim"],
                          check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(dest)
    return Path(dest) / "challenge_sim"


def main():
    ap = argparse.ArgumentParser(description="Prova generale sui dati simulati")
    ap.add_argument("--sim", help="cartella challenge_sim (con defense/ e attack/)")
    ap.add_argument("--gruppi", nargs="*", default=["falco", "koala", "lynx"],
                    help="gruppi da attaccare (default: falco koala lynx)")
    a = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="prova_gara_clk_"))
    sim = Path(a.sim) if a.sim else extract_sim(work / "dati")
    work_kit = work / "gara_clk"
    work_scripts = work_kit / "scripts"
    work_scripts.mkdir(parents=True)
    for name in SCRIPTS:
        shutil.copy(SCRIPT_DIR / name, work_scripts / name)
    work_modules = work / "da_consegnare"
    work_modules.mkdir()
    for name in MODULES:
        shutil.copy(SUBMISSION_DIR / name, work_modules / name)
    shutil.copy(BASE_DIR / "clk.npy", work_kit / "clk.npy")

    # le cartelle come il giorno della gara
    shutil.copytree(sim / "defense", work_kit / "originali", ignore=shutil.ignore_patterns("*.py", "*.npy"))
    shutil.copytree(sim / "attack" / "images", work_kit / "immagini_avversari")
    shutil.copytree(sim / "attack" / "detections", work_kit / "detection_avversari",
                    ignore=shutil.ignore_patterns("__pycache__"))
    print("Cartella di prova:", work_kit, flush=True)

    print("\n##### DIFESA #####", flush=True)
    subprocess.run([sys.executable, "scripts/gara_difesa.py"], cwd=work_kit, check=True)
    print("\n##### ATTACCO #####", flush=True)
    subprocess.run([sys.executable, "scripts/gara_attacco.py", "--gruppi"] + a.gruppi, cwd=work_kit, check=True)


if __name__ == "__main__":
    main()
