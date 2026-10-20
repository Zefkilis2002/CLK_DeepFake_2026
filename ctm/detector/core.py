"""
C7: cuore della detection, senza file. La detection consegnata
(generata da submission/build_submission.py) aggiunge solo la lettura dei percorsi.
"""
from ctm.common.imgio import match_shape
from ctm.common.watermark import similarity
from ctm.common.wpsnr import wpsnr


def detect_core(scheme, params, tau, original, watermarked, attacked, ctx=None):
    """Restituisce (presente 0/1, WPSNR(marchiata, attaccata), similarità)."""
    attacked = match_shape(attacked, original.shape)
    ctx = ctx or scheme.prepare(original, params)
    w_extracted = scheme.extract(ctx, watermarked, params)
    w_attacked = scheme.extract(ctx, attacked, params)
    sim = similarity(w_extracted, w_attacked)
    return int(sim >= tau), wpsnr(watermarked, attacked), sim
