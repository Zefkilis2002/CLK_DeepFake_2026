"""C10: tabella dei risultati in sola aggiunta (CSV), una riga per esperimento."""
import csv
import os
import time

from config.settings import RESULTS_DIR


def append_result(table, row):
    """Aggiunge 'row' (dizionario) a results/<table>.csv, con data e ora."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, table + ".csv")
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row), delimiter=";", extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)
    return path
