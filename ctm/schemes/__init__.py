"""Registro degli schemi: nome -> oggetto Scheme."""
from ctm.schemes.bitplane import BitPlane
from ctm.schemes.dct_ss import DctSS
from ctm.schemes.dwt_hvs import DwtHvs

SCHEMES = {s.name: s for s in (BitPlane(), DctSS(), DwtHvs())}


def get_scheme(name):
    return SCHEMES[name]
