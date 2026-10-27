"""
C11: oracoli = "l'immagine attaccata contiene ancora il marchio?".
È l'UNICO modulo che conosce l'immagine originale: la ricerca degli attacchi
riceve solo un oggetto con il metodo query(attaccata) -> (presente, wpsnr).

  LocalOracle  : uno schema dei laboratori + parametri + tau (banco di prova)
  FileOracle   : la funzione detection(path, path, path) di un gruppo (.py o .pyc)
"""
import contextlib
import importlib.util
import io
import os

import cv2

from ctm.detector.core import detect_core


class Oracle:
    def __init__(self):
        self.queries = 0

    def query(self, attacked):
        self.queries += 1
        return self._query(attacked)


class LocalOracle(Oracle):
    def __init__(self, scheme, params, tau, original, watermarked):
        super().__init__()
        self.scheme, self.params, self.tau = scheme, params, tau
        self.original, self.watermarked = original, watermarked
        self.ctx = scheme.prepare(original, params)

    def _query(self, attacked):
        present, w, _ = detect_core(self.scheme, self.params, self.tau,
                                    self.original, self.watermarked, attacked, self.ctx)
        return present, w


def load_detection(path):
    """Carica detection_<gruppo>.py o .pyc (come il grader .pyc dei laboratori)."""
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.detection


class FileOracle(Oracle):
    def __init__(self, detection, original_path, watermarked_path, tmp_path):
        super().__init__()
        self.detection = detection
        self.original_path, self.watermarked_path = original_path, watermarked_path
        self.tmp_path = tmp_path

    def _query(self, attacked):
        cv2.imwrite(self.tmp_path, attacked)
        with contextlib.redirect_stdout(io.StringIO()):          # ignora stampe altrui
            present, w = self.detection(self.original_path, self.watermarked_path, self.tmp_path)
        return int(present), float(w)
