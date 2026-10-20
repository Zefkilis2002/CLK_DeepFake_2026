"""
Test veloci dei componenti (senza pytest).

Uso:  python -m tests.test_basic
"""
import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config.settings import load_split
from ctm.attack.chains import CHAINS, run_chain
from ctm.common.attacks import ATTACKS, awgn
from ctm.common.imgio import read_gray, to_uint8
from ctm.common.watermark import load_mark, random_mark, similarity, to_bipolar
from ctm.common.wpsnr import wpsnr, wpsnr_lab
from ctm.detector.core import detect_core
from ctm.infra import cache
from ctm.infra.seeds import seed_for
from ctm.schemes import SCHEMES

IMG = read_gray(load_split()[0][0])
BITS = load_mark()


def test_imgio():
    assert to_uint8(np.array([-3.0, 0.4, 0.6, 300.0])).tolist() == [0, 0, 1, 255]


def test_wpsnr():
    noisy = awgn(IMG, 3)
    assert abs(wpsnr(IMG, noisy) - wpsnr_lab(IMG, noisy)) < 1e-9
    assert abs(wpsnr(IMG, noisy) - wpsnr(noisy, IMG)) < 1e-9          # simmetrico
    assert wpsnr(IMG, awgn(IMG, 2)) > wpsnr(IMG, awgn(IMG, 8))          # cala con il rumore


def test_watermark():
    assert BITS.size == 1024 and similarity(np.zeros(1024), to_bipolar(BITS)) == 0.0
    assert (random_mark(5) == random_mark(5)).all()


def test_attacks():
    for name, fn in ATTACKS.items():
        p = {"awgn": 5, "blur": 1, "sharp": 1, "median": 3, "jpeg": 50, "resize": 0.5}[name]
        out = fn(IMG, p)
        assert out.dtype == np.uint8 and out.shape == IMG.shape, name
    assert (awgn(IMG, 5, seed=1) == awgn(IMG, 5, seed=1)).all()
    for chain in CHAINS:
        assert run_chain(IMG, chain, 0.5).shape == IMG.shape


def test_schemes_roundtrip():
    defaults = {"bitplane": {"plane": 1}, "dct_ss": {}, "dwt_hvs": {}}
    for name, scheme in SCHEMES.items():
        params = scheme.params(defaults[name])
        ctx = scheme.prepare(IMG, params)
        wm = scheme.embed(ctx, BITS, params)
        assert (wm == scheme.embed(ctx, BITS, params)).all(), name                 # deterministico
        assert similarity(to_bipolar(BITS), scheme.extract(ctx, wm, params)) > 0.8, name
        present, _, _ = detect_core(scheme, params, 0.2, IMG, wm, IMG)
        assert present == 0, name                                                  # originale -> 0


def test_dct_ss_copies():
    scheme = SCHEMES["dct_ss"]
    params = scheme.params({"n": 2048})
    ctx = scheme.prepare(IMG, params)
    w = scheme.extract(ctx, scheme.embed(ctx, BITS, params), params)
    assert w.size == 1024 and similarity(to_bipolar(BITS), w) > 0.8


def test_cache_and_seeds():
    assert seed_for("a", 1) == seed_for("a", 1) != seed_for("a", 2)
    old = cache.CACHE_DIR
    cache.CACHE_DIR = tempfile.mkdtemp()
    calls = []
    f = lambda: calls.append(1) or 42
    assert cache.cached("t", ["k"], f) == 42 and cache.cached("t", ["k"], f) == 42
    assert len(calls) == 1
    cache.CACHE_DIR = old


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("OK", t.__name__)
    print("tutti i %d test passati" % len(tests))
