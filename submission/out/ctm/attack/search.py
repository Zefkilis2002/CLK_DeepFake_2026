"""
C12: ricerca dell'attacco riuscito con il WPSNR più alto.

Per ogni catena:
  1. forza massima con WPSNR >= 35.5 trovata in LOCALE (il WPSNR non costa query);
  2. una query lì: se il marchio resiste, la catena è scartata;
  3. bisezione su s per l'attacco più leggero che cancella ancora il marchio.
Il WPSNR locale salta le query che non possono migliorare il migliore trovato.
L'originale non compare mai: solo la marchiata e l'oracolo.
"""
from ctm.attack.chains import CHAINS, describe, run_chain
from ctm.common.wpsnr import wpsnr


def strongest_allowed(wm, chain, min_wpsnr=35.5, steps=6):
    if wpsnr(wm, run_chain(wm, chain, 1.0)) >= min_wpsnr:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(steps):
        mid = (lo + hi) / 2
        if wpsnr(wm, run_chain(wm, chain, mid)) >= min_wpsnr:
            lo = mid
        else:
            hi = mid
    return lo


def search_attack(wm, oracle, chains=None, bisect_steps=4):
    """Restituisce dict(image, wpsnr, chain, s, text) del migliore attacco, oppure None."""
    best = None
    for chain in (chains or CHAINS):
        s_hi = strongest_allowed(wm, chain)
        attacked = run_chain(wm, chain, s_hi)
        if best is not None and wpsnr(wm, attacked) <= best["wpsnr"]:
            continue
        present, w = oracle.query(attacked)
        if present or w < 35:
            continue
        found = {"image": attacked, "wpsnr": w, "chain": chain, "s": s_hi}
        s_lo = 0.0
        for _ in range(bisect_steps):
            mid = (s_lo + s_hi) / 2
            attacked = run_chain(wm, chain, mid)
            if wpsnr(wm, attacked) <= found["wpsnr"]:
                s_lo = mid
                continue
            present, w = oracle.query(attacked)
            if not present and w >= 35:
                s_hi = mid
                found = {"image": attacked, "wpsnr": w, "chain": chain, "s": mid}
            else:
                s_lo = mid
        if best is None or found["wpsnr"] > best["wpsnr"]:
            best = found
    if best is not None:
        best["text"] = describe(best["chain"], best["s"])
    return best
