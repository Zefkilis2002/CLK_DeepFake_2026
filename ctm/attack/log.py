"""C12: log degli attacchi per la fase di verifica (formato richiesto, apribile con Excel)."""
import csv
import os

HEADER = ["Image", "Group", "WPSNR", "Attack(s) with parameters", "Chain", "Strength"]


def log_attack(path, image, group, wpsnr, text, chain, strength):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        if new:
            writer.writerow(HEADER)
        writer.writerow([image, group, "%.2f" % wpsnr, text, chain, "%.4f" % strength])
