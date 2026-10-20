# Runbook: versione avanzata

Elenco rapido di componenti, meccanismi e procedure. Lo schema e i file di consegna sono
quelli della versione base (runbook in `RUNBOOK.md` sul branch `main`). Qui cambia
l'infrastruttura intorno.

## Idea in una riga
Spread Spectrum **additivo** in DCT (LAB-3), detection non-blind con tau dalla ROC.
I parametri si scelgono con una ricerca a stadi, si salvano come preset, e i file di
consegna si generano dal preset.

## Componenti (numerazione del piano)
| ID | Dove | Cosa |
|---|---|---|
| C1 | `config/settings.py`, `presets.json`, `split.json` | costanti e percorsi; preset (schema, parametri, tau, riepilogo ROC); split fisso 70/30 |
| C2 | `ctm/common/imgio.py` | `read_gray`, `write_bmp`, `to_uint8` (unica regola di arrotondamento), `match_shape` |
| C3 | `ctm/common/wpsnr.py` | WPSNR del lab con convoluzione via FFT; `wpsnr_lab` originale per i test |
| C4 | `ctm/common/watermark.py` | `load_mark`, `random_mark(seed)`, `to_bipolar`, `similarity`, `mark_hash` |
| C5 | `ctm/common/attacks.py` | 6 attacchi uint8, `random_attack` della ROC, `SCREEN_SUITE` per l'halving |
| C6 | `ctm/schemes/` | interfaccia `Scheme` (`prepare`, `embed`, `extract`) + registro `SCHEMES` |
| C7 | `ctm/detector/core.py` | `detect_core` → (presente, WPSNR, similarità), senza file |
| C8 | `ctm/roc/` | punteggi H1/H0 per immagine (parallelo + cache), scelta di tau, grafico |
| C9 | `ctm/search/` | `calibrate_alpha` (bisezione verso un WPSNR obiettivo), `run_halving` S0-S3 |
| C10 | `ctm/infra/` | cache su disco, tabella CSV dei risultati, `pmap` multiprocessing, `seed_for` |
| C11 | `ctm/attack/oracle.py` | `LocalOracle` (schema + tau), `FileOracle` (detection altrui `.py`/`.pyc`) |
| C12 | `ctm/attack/chains.py`, `search.py`, `log.py` | 13 catene con forza s; ricerca con bisezione; log Excel |
| C13 | `ctm/bench/run.py` | banco di prova contro difensori locali; punti qualità/robustezza |
| C14 | `submission/` | generatore da template, controlli di conformità sui file generati |
| C15 | `ctm/attack/day.py`, `scripts/defense_day.py`, questo file | procedure del giorno della gara |

## Preset attuali (`config/presets.json`)
| Preset | Schema | alpha | tau | WPSNR p10 | Battuto a (stima) |
|---|---|---|---|---|---|
| P66 | DCT additivo, n 1024 | 4.36 | 0.1214 | 66.1 dB | 40.1 dB |
| **P64 (attivo)** | DCT additivo, n 1024 | 5.0 | 0.1218 | 63.8 dB | 38.0 dB |
| P62 | DCT additivo, n 1024 | 5.80 | 0.1174 | 62.0 dB | 39.1 dB |
| P58 | DCT additivo, n 2048 | 6.49 | 0.1061 | 58.1 dB | 37.4 dB |

P58/P62/P66 vengono dall'halving; P64 è il preset della versione base (tau ricalcolata).
Il DWT-HVS del LAB-4 arriva in S2-S3 ma perde sempre: viene battuto intorno a 43-46 dB.

## Schemi e varianti
- `dct_ss`: `mode` additive/multiplicative, `n` = 1024/2048/4096 (ripetizione dei bit + media delle copie).
- `dwt_hvs`: `alpha`, `exponent` della maschera (0.2 del LAB-4), `permute_seed` (posizioni mescolate).
- `bitplane`: `plane` 0-3 (solo difensore di confronto).

## Meccanismi
- **Preset**: unica fonte dei parametri. Embedding, detection e tau vengono dallo stesso preset;
  `"active"` sceglie quello in uso.
- **Halving (C9)**, per obiettivo 58/62/66 dB:
  S0 taratura di alpha (5 immagini) → S1 suite fissa di 8 attacchi (10 immagini, tiene 3) →
  S2 ROC + banco di prova (tiene 2) → S3 conferma su immagini di verifica → preset `P<obiettivo>`.
  Punteggio: punti qualità del 10° percentile WPSNR + punti robustezza (WPSNR a cui ci battono − 3 dB).
- **Cache**: chiave = hash di schema, parametri, immagine, impronta del watermark. Rilanciare
  riprende da dove si era fermato; cancellare `cache/` per ricalcolare.
- **Multiprocessing**: ROC, banco di prova ed E0 lavorano un'immagine per processo (`WORKERS` = CPU − 1).
- **Seed**: ogni attacco casuale ha un seed ricavato da (schema, parametri, immagine, indice): risultati riproducibili.
- **Risultati**: ogni esperimento aggiunge righe a `results/<tabella>.csv` (e0, roc, bench, halving).
- **Template (C14)**: `{{ALPHA}}`, `{{TAU}}`, ... sostituiti dal preset; WPSNR, schema e similarità
  copiati dai blocchi `# >>> inline` / `# <<< inline` del package. La detection resta un file unico.
- **Controlli (C14)**: file unico, solo cv2/numpy/scipy, prova in interprete pulito (`python -I`)
  fuori dal repository, nessuna stampa, < 5 s, marchiata 1, originale/altri watermark/altre immagini/
  distrutte 0, attacchi leggeri 1, funzioni di attacco senza originale.
- **Tau**: TPR massimo con FPR ≤ 0.01, mai sotto media(H0) + 4·std(H0); H0 = watermark casuale,
  altro watermark, originale attaccata.
- **Ricerca attacchi**: forza massima con WPSNR ≥ 35.5 trovata in locale, poi bisezione con
  l'oracolo; il WPSNR locale salta le query inutili.
- **Isolamento dell'originale**: solo gli oracoli e `attack/day.py` hanno il percorso dell'originale.
- **Giorno di attacco**: fase A (2 passi di bisezione, tutti i gruppi) poi fase B (4 passi,
  prima le immagini peggiori); si tiene il migliore verificato sul file salvato.

## Comandi (dalla radice del repository `ctm/`)
```
python -m tests.test_basic
python -m scripts.run_embedding_search      # lungo: scrive P58/P62/P66
python -m scripts.run_roc P62               # ricalcola tau di un preset
python -m scripts.run_bench 10
python -m submission.build_submission --preset P64 --group NOME
python -m submission.checks --group NOME
```

## Giorno della gara
**Difesa (8:30-10:00)**
1. `python -m scripts.defense_day --images CARTELLA --group NOME`: prova i preset dal più alto,
   sceglie quello in fascia su tutte e 3 le immagini, salva `NOME_immagine.bmp` in `defense_out/`,
   genera `submission/out/` e lancia i controlli.
2. Upload delle immagini e di `submission/out/detection_NOME.py`.
3. Dopo la difesa non si cambia più nessun file.

**Attacco (10:15-12:30)**
1. `python -m scripts.attack_day --targets T --detectors D --originals O --our NOME`
   (sui 3 portatili si possono dividere i gruppi in cartelle `targets` diverse).
2. Upload da `attacked/`, log Excel da `attack_log.csv`.

## Differenze rispetto alla base
| Base (`main`) | Avanzata (`avanzata`) |
|---|---|
| file piatti, ALPHA/TAU scritti a mano | package + preset JSON + file generati |
| alpha scelto a mano (E0 + bench) | calibrazione per bisezione + halving |
| esecuzione sequenziale | multiprocessing + cache + tabella risultati |
| `check_rules.py` sui file scritti a mano | `submission/checks.py` sui file generati, anche in interprete pulito |
| un solo passaggio di attacco | fase A (ampiezza) + fase B (profondità) |

## Ancora come nella base (non implementato)
Oracolo esterno in un sottoprocesso con timeout, ricerca level-set su WPSNR, attacchi localizzati.
