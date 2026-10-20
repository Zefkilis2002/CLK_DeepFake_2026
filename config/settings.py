"""
C1: costanti globali del progetto (unica fonte per percorsi, seed e scelte fisse).
I parametri che si regolano (schema, alpha, tau) stanno in presets.json.
"""
import json
import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(REPO_DIR)              # F:\ChallengeDeepFake

# dati (fuori dal repository: materiale del corso)
IMAGES_DIR = os.path.join(PROJECT_DIR, "images")
MARK_PATH = os.path.join(PROJECT_DIR, "clk.npy")

# cartelle generate (non versionate)
RESULTS_DIR = os.path.join(REPO_DIR, "results")
CACHE_DIR = os.path.join(REPO_DIR, "cache")
SUBMISSION_OUT = os.path.join(REPO_DIR, "submission", "out")

# file di configurazione versionati
CONFIG_DIR = os.path.join(REPO_DIR, "config")
PRESETS_PATH = os.path.join(CONFIG_DIR, "presets.json")
SPLIT_PATH = os.path.join(CONFIG_DIR, "split.json")

IMAGE_SIZE = 512
MARK_SIZE = 1024
MIN_WPSNR = 35.0                  # soglia di successo di un attacco
FPR_TARGET = 0.01                 # dentro [0, 0.1] delle regole
TAU_FLOOR_SIGMAS = 4              # tau >= media(H0) + 4 std(H0)
MASTER_SEED = 2026
WORKERS = max(1, (os.cpu_count() or 2) - 1)   # processi per multiprocessing

GROUP = "groupname"               # nome del gruppo (da cambiare)


# ---------------------------------------------------------------------------
# Preset e split: lettura/scrittura dei JSON
# ---------------------------------------------------------------------------
def load_presets():
    with open(PRESETS_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_presets(data):
    with open(PRESETS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def active_preset(name=None):
    """Restituisce (nome, preset) del preset attivo o di quello richiesto."""
    data = load_presets()
    name = name or data["active"]
    return name, data["presets"][name]


def load_split():
    """Divisione fissa delle immagini di prova: sviluppo (70%) e verifica (30%)."""
    with open(SPLIT_PATH, encoding="utf-8") as f:
        split = json.load(f)
    to_path = lambda names: [os.path.join(IMAGES_DIR, n) for n in names]
    return to_path(split["dev"]), to_path(split["heldout"])
