"""
C8: costruzione dei punteggi H1/H0 per la ROC.

Per ogni immagine (un job, eseguibile in parallelo e salvato in cache):
  H1  sim(W_extracted, W_attacked) per K attacchi casuali con WPSNR >= 35
  H0  stesso attacco su: immagine con un ALTRO watermark (h0_other),
      originale non marchiata (h0_orig); più watermark casuale contro W_attacked (h0_random)
"""
import numpy as np

from config.settings import MIN_WPSNR
from ctm.common.attacks import random_attack
from ctm.common.imgio import read_gray
from ctm.common.watermark import mark_hash, random_mark, similarity, to_bipolar
from ctm.common.wpsnr import wpsnr
from ctm.infra.cache import cached
from ctm.infra.parallel import pmap
from ctm.infra.seeds import seed_for
from ctm.schemes import get_scheme


def image_scores(job):
    """Punteggi di una sola immagine. job = (schema, parametri, percorso, bits, K)."""
    scheme_name, params, path, bits, n_attacks = job
    key = ["roc", scheme_name, params, path, mark_hash(bits), n_attacks]

    def compute():
        scheme = get_scheme(scheme_name)
        orig = read_gray(path)
        ctx = scheme.prepare(orig, params)
        wm = scheme.embed(ctx, bits, params)
        w_ext = scheme.extract(ctx, wm, params)
        other = scheme.embed(ctx, random_mark(seed_for("other", path)), params)
        rows = []                                    # (punteggio, label, tipo)
        for k in range(n_attacks):
            seed = seed_for("attack", scheme_name, params, path, k)
            attacked, _ = random_attack(wm, np.random.default_rng(seed))
            if wpsnr(wm, attacked) < MIN_WPSNR:
                continue
            w_att = scheme.extract(ctx, attacked, params)
            rows.append((similarity(w_ext, w_att), 1, "h1"))
            rows.append((similarity(to_bipolar(random_mark(seed)), w_att), 0, "h0_random"))
            for img, kind in ((other, "h0_other"), (orig, "h0_orig")):
                att0, _ = random_attack(img, np.random.default_rng(seed))
                rows.append((similarity(w_ext, scheme.extract(ctx, att0, params)), 0, kind))
        return rows

    return cached("roc_scores", key, compute)


def build_scores(scheme_name, params, paths, bits, n_attacks=10, workers=None):
    """Punteggi di tutte le immagini: (scores, labels, kinds) come array numpy."""
    jobs = [(scheme_name, params, p, bits, n_attacks) for p in paths]
    rows = [r for per_image in pmap(image_scores, jobs, workers) for r in per_image]
    scores = np.array([r[0] for r in rows])
    labels = np.array([r[1] for r in rows])
    kinds = np.array([r[2] for r in rows])
    return scores, labels, kinds
