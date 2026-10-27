"""
C10: esecuzione in parallelo su più processi (multiprocessing).

Su Windows i processi figli rieseguono lo script: chi chiama pmap deve stare
sotto `if __name__ == "__main__":` e la funzione deve essere definita a livello
di modulo (non una lambda).
"""
from multiprocessing import Pool

from config.settings import WORKERS


def pmap(fn, jobs, workers=None):
    """Come list(map(fn, jobs)), ma su 'workers' processi. workers=1 -> sequenziale."""
    jobs = list(jobs)
    workers = WORKERS if workers is None else workers
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    with Pool(min(workers, len(jobs))) as pool:
        return pool.map(fn, jobs)
