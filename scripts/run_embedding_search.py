"""
Ricerca dell'embedding a stadi (C9): taratura + halving, scrive i preset P58/P62/P66.

Uso:  python -m scripts.run_embedding_search
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import load_split
from ctm.common.watermark import load_mark
from ctm.search.halving import run_halving, write_presets


def main():
    dev, heldout = load_split()
    winners = run_halving(dev, heldout, load_mark())
    data = write_presets(winners)
    print("\nPreset scritti in config/presets.json:")
    for name, p in data["presets"].items():
        print("  %s: %s %s tau=%s" % (name, p["scheme"], p["params"], p["tau"]))
    print("Preset attivo:", data["active"], "(si cambia a mano in presets.json)")


if __name__ == "__main__":
    main()
