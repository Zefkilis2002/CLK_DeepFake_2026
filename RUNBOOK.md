# Runbook: versione base

Elenco rapido di cosa c'è nel progetto e di come si usa. Dettagli in `README.md` e
`docs/analisi/verifica_piano.md`.

## Idea in una riga
Spread Spectrum **additivo** in DCT (LAB-3): ±ALPHA sui 1024 coefficienti DCT più grandi
(DC escluso), detection non-blind con similarità coseno e soglia TAU dalla ROC.

## Parametri attivi
- `ALPHA = 5.0` (in `embedding.py` e `detection_groupname.py`, devono coincidere)
- `TAU = 0.1253` (solo in `detection_groupname.py`)
- Risultato atteso: WPSNR circa 64 dB (5 punti qualità), battuto in media intorno a 39 dB (circa 5 punti robustezza)
- Preset alternativo: `ALPHA = 4.0`, `TAU = 0.1246` → 68 dB, più fragile (circa 43 dB)

## Componenti
| File | Ruolo |
|---|---|
| `embedding.py` | `embedding(input1, input2)` → immagine marchiata uint8 |
| `detection_groupname.py` | `detection(input1, input2, input3)` → `(0/1, WPSNR)`; file unico, solo cv2/numpy/scipy |
| `roc.py` | calcola TAU (H1 + 3 tipi di H0, ROC, AUC, grafico) |
| `attacks.py` | `attacks(input1, attack_name, param_array)`, ricerca automatica, modalità gara |
| `common.py` | I/O, `to_uint8`, WPSNR, similarità, 6 attacchi, `random_attack` |
| `schemes.py` | bit-plane, SS additivo, SS moltiplicativo, DWT-HVS (difensori finti) |
| `e0_quality.py` | qualità al variare di alpha (esperimento E0) |
| `bench.py` | robustezza: con che WPSNR ci battono |
| `check_rules.py` | conformità alle regole, deve dare TUTTO OK |
| `demo.py` | demo completa su un'immagine |

## Meccanismi chiave
- **Posizioni**: sempre calcolate sull'originale (detection non-blind).
- **Additivo e non moltiplicativo**: qualità uniforme tra immagini, nessuna immagine persa per
  l'arrotondamento, nessuna correlazione tra marchi diversi.
- **Quantizzazione**: un'unica regola `clip` + `rint` in uint8.
- **WPSNR**: formula del lab, convoluzione fatta con FFT (identica, 0.1 s invece di 3 s).
- **Similarità**: coseno, 0 se un vettore è nullo (originale come attaccata → 0).
- **Tau**: TPR massimo con FPR ≤ 0.01, mai sotto media(H0) + 4·std(H0).
- **H0 della ROC**: watermark casuale, immagine con un altro watermark, originale attaccata.
- **Detection sempre eseguita**: nessun `if` sul WPSNR, nessuna stampa.
- **Ricerca attacchi**: 13 catene (6 singole, 6 coppie, 1 tripla) con forza s in [0, 1].
  Forza massima con WPSNR ≥ 35 trovata in locale, poi bisezione con la detection avversaria.
  Il WPSNR locale evita le query inutili.
- **Isolamento dell'originale**: le funzioni di attacco ricevono solo la marchiata; l'originale
  compare solo nell'oracolo (`make_oracle`), come argomento della detection dell'altro gruppo.
- **Detection altrui**: caricata con `importlib` da `.py` o `.pyc`, stampe soppresse.
- **Verifica**: l'attacco migliore viene salvato, riletto e verificato prima del log.
- **Log**: `attack_log.csv` (Image; Group; WPSNR; Attack(s)) da aprire con Excel.

## Comandi (da `ctm/`)
```
..\.venv\Scripts\python.exe check_rules.py     # prima di ogni consegna/upload
..\.venv\Scripts\python.exe roc.py 5           # nuova TAU per un alpha
..\.venv\Scripts\python.exe bench.py 10        # robustezza
..\.venv\Scripts\python.exe demo.py 0042       # demo
..\.venv\Scripts\python.exe attacks.py --our NOME --targets T --detectors D --originals O
```

## Giorno della gara
**Difesa (8:30-10:00)**
1. Scaricare le 3 immagini, `embedding()` con il preset attivo, salvare come `NOME_immagine.bmp`.
2. Controllare il WPSNR; se serve cambiare preset, aggiornare ALPHA e TAU nei due file.
3. `check_rules.py`, poi upload delle immagini e di `detection_NOME.py`.
4. Dopo la difesa non si cambia più nessun file.

**Attacco (10:15-12:30)**
1. Immagini in `targets/`, detection in `detections/`, originali in `originals/`.
2. `attacks.py` su tutti i gruppi: prima ampiezza (attività), poi rilanci sulle immagini peggiori.
3. Upload di `NOME_gruppoB_immagine.bmp` dalla cartella `attacked/`, log da `attack_log.csv`.

## Prima della consegna
- Rinominare `detection_groupname.py` → `detection_NOME.py` e aggiornare gli import e `GROUP` in `demo.py`.
- Confermare la data di consegna con gli assistenti.
