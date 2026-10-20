# Capture the Mark: versione avanzata (branch `avanzata`)

Stesso schema e stessi file di consegna della versione base (branch `main`), con
l'infrastruttura completa del piano `docs/analisi/ctm_design_plan.md`: package
modulare, preset JSON, cache, multiprocessing, ricerca a stadi (halving) e
generatore dei file di consegna da template. Elenco completo di componenti e
meccanismi in `runbook/RUNBOOK.md`.

Tutti i comandi si lanciano dalla radice del repository (`ctm/`), con il venv del progetto:

```
..\.venv\Scripts\python.exe -m tests.test_basic                 # test dei componenti
..\.venv\Scripts\python.exe -m scripts.run_e0                   # esperimento E0
..\.venv\Scripts\python.exe -m scripts.run_embedding_search     # halving -> preset P58/P62/P66
..\.venv\Scripts\python.exe -m scripts.run_roc [PRESET]         # tau del preset -> presets.json
..\.venv\Scripts\python.exe -m scripts.run_bench [N]            # robustezza (banco di prova)
..\.venv\Scripts\python.exe -m submission.build_submission      # genera submission/out/
..\.venv\Scripts\python.exe -m submission.checks                # controlli sui file generati
..\.venv\Scripts\python.exe -m scripts.defense_day --images CARTELLA
..\.venv\Scripts\python.exe -m scripts.attack_day --targets T --detectors D --originals O
```

## Struttura

```
config/      settings.py (costanti, percorsi, GROUP), presets.json, split.json (70/30)
ctm/         package
  common/    imgio, wpsnr, watermark, attacks          (C2-C5)
  schemes/   base (interfaccia), bitplane, dct_ss, dwt_hvs  (C6)
  detector/  core: detect_core senza file               (C7)
  roc/       scores (H1/H0), tau (scelta, grafico)      (C8)
  search/    calibrate (bisezione alpha), halving        (C9)
  infra/     cache, results, parallel, seeds             (C10)
  attack/    oracle, chains, search, log, day            (C11, C12, C15)
  bench/     run: banco di prova e punti della gara      (C13)
scripts/     punti di ingresso, uno per attività
submission/  build_submission.py, templates/, checks.py, out/ (generato)   (C14)
runbook/     RUNBOOK.md                                                  (C15)
tests/       test_basic.py
```

## Regole della struttura
- Le dipendenze vanno solo verso il basso: `common` non importa nulla del progetto,
  `scripts/` e `submission/` importano il package e mai il contrario.
- I parametri regolabili stanno solo in `config/presets.json`; il preset attivo è `"active"`.
- `submission/out/` si genera dal preset e non si modifica a mano.
- Solo `ctm/attack/oracle.py` e `ctm/attack/day.py` conoscono il percorso dell'originale.
- `cache/` e `results/` sono generati e non versionati; cancellare `cache/` forza il ricalcolo.

Prima della consegna: impostare `GROUP` in `config/settings.py` e rigenerare con
`build_submission --group NOME`.
