"""
C12: libreria di catene di attacchi. Ogni catena trasforma una forza s in [0,1]
(0 = leggerissimo, 1 = fortissimo) nei parametri dei suoi attacchi.
Contiene anche attacks(input1, attack_name, param_array) chiesta dalle regole.
"""
import numpy as np

from ctm.common.attacks import ATTACKS
from ctm.common.imgio import read_gray


def apply_attacks(img, names, params):
    """Applica in sequenza gli attacchi 'names' con i parametri 'params' (solo la marchiata)."""
    out = img
    for name, p in zip(names, params):
        out = ATTACKS[name](out, p)
    return out


def attacks(input1, attack_name, param_array):
    """Firma delle regole: percorso della marchiata, nome/i degli attacchi, parametri."""
    names = [attack_name] if isinstance(attack_name, str) else list(attack_name)
    return apply_attacks(read_gray(input1), names, list(np.atleast_1d(param_array)))


def k_median(s):
    return [3, 5, 7, 9][min(int(s * 4), 3)]


CHAINS = {
    "jpeg":             [("jpeg", lambda s: int(round(95 - 90 * s)))],
    "blur":             [("blur", lambda s: 0.2 + 2.8 * s)],
    "median":           [("median", k_median)],
    "awgn":             [("awgn", lambda s: 0.5 + 29.5 * s)],
    "resize":           [("resize", lambda s: 0.95 - 0.8 * s)],
    "sharp":            [("sharp", lambda s: 0.2 + 4.8 * s)],
    "blur+jpeg":        [("blur", lambda s: 0.2 + 1.8 * s), ("jpeg", lambda s: int(round(95 - 60 * s)))],
    "median+jpeg":      [("median", lambda s: 3 if s < 0.5 else 5), ("jpeg", lambda s: int(round(95 - 70 * s)))],
    "awgn+jpeg":        [("awgn", lambda s: 0.5 + 15 * s), ("jpeg", lambda s: int(round(95 - 60 * s)))],
    "resize+jpeg":      [("resize", lambda s: 0.95 - 0.6 * s), ("jpeg", lambda s: int(round(95 - 60 * s)))],
    "awgn+median":      [("awgn", lambda s: 0.5 + 15 * s), ("median", lambda s: 3 if s < 0.5 else 5)],
    "blur+awgn":        [("blur", lambda s: 0.2 + 1.8 * s), ("awgn", lambda s: 0.5 + 15 * s)],
    "median+blur+jpeg": [("median", lambda s: 3), ("blur", lambda s: 0.2 + 1.3 * s),
                         ("jpeg", lambda s: int(round(95 - 60 * s)))],
}


def chain_params(chain, s):
    return [n for n, _ in CHAINS[chain]], [f(s) for _, f in CHAINS[chain]]


def run_chain(img, chain, s):
    names, params = chain_params(chain, s)
    return apply_attacks(img, names, params)


def describe(chain, s):
    """Testo per il log Excel, es. 'blur 1.20 + jpeg 70'."""
    names, params = chain_params(chain, s)
    return " + ".join("%s %s" % (n, ("%.2f" % p) if isinstance(p, float) else p)
                      for n, p in zip(names, params))
