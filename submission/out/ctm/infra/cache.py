"""
C10: cache su disco. Un risultato costoso (punteggi ROC di un'immagine, esito
di un attacco) viene salvato in cache/<namespace>/<hash>.pkl e riletto se la
stessa chiave ricompare: un lavoro interrotto riparte senza ricalcolare.
"""
import os
import pickle

from config.settings import CACHE_DIR
from ctm.infra.seeds import config_hash

ENABLED = True


def cached(namespace, key, compute):
    """Restituisce compute() leggendolo dalla cache se la chiave è già stata calcolata."""
    if not ENABLED:
        return compute()
    folder = os.path.join(CACHE_DIR, namespace)
    path = os.path.join(folder, config_hash(key) + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    value = compute()
    os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(value, f)
    os.replace(tmp, path)          # scrittura atomica: niente file a metà se si interrompe
    return value
