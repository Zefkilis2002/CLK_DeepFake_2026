"""
C6: interfaccia comune degli schemi.

    ctx = scheme.prepare(original, params)    # trasformate/maschere dell'originale, una volta
    wm  = scheme.embed(ctx, bits, params)      # immagine marchiata uint8
    w   = scheme.extract(ctx, image, params)   # vettore float di 1024 valori (stima di x = +-1)

ROC, ricerca, banco di prova e detector usano solo questi tre metodi:
aggiungere uno schema = scrivere un file con una nuova sottoclasse.
"""


class Scheme:
    name = "base"
    default_params = {}

    def prepare(self, original, params):
        return {"original": original}

    def embed(self, ctx, bits, params):
        raise NotImplementedError

    def extract(self, ctx, image, params):
        raise NotImplementedError

    # scorciatoie
    def params(self, params=None):
        """Parametri di default completati con quelli passati."""
        full = dict(self.default_params)
        full.update(params or {})
        return full

    def embed_image(self, original, bits, params=None):
        p = self.params(params)
        return self.embed(self.prepare(original, p), bits, p)
