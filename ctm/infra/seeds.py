"""C10: seed deterministici, ricavati dagli ingredienti dell'esperimento."""
import hashlib
import json

from config.settings import MASTER_SEED


def config_hash(obj):
    """Impronta stabile di un dizionario/lista (chiave di cache, id di configurazione)."""
    text = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def seed_for(*parts):
    """Stesso input -> stesso seed, ad esempio seed_for('roc', config, immagine, k)."""
    return int(config_hash([MASTER_SEED, *parts]), 16) % (2 ** 31)
