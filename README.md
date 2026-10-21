# Capture the Mark: codice della challenge

Watermarking con **Spread Spectrum additivo in DCT** (LAB-3): si somma ±ALPHA ai 1024
coefficienti DCT più grandi dell'immagine (DC escluso). La detection è non-blind: confronta il
marchio estratto dalla marchiata con quello estratto dall'attaccata (similarità coseno ≥ TAU).

## Parametri

| Parametro | Valore | Dove |
|---|---|---|
| ALPHA | 5.0 | `embedding.py` e `detection_groupname.py` (devono coincidere) |
| TAU | 0.1218 | `detection_groupname.py` |
| Coefficienti usati | 1024 (uno per bit) | entrambi i file |

Risultati attesi: WPSNR dell'embedding circa 64 dB (5 punti qualità). La nostra ricerca di
attacchi ci batte in media intorno a 38 dB (circa 5 punti robustezza). TAU dà FPR 0 sui casi
senza marchio. Le analisi complete (halving, preset, banco di prova) sono sul branch `avanzata`.

## File

| File | Ruolo |
|---|---|
| `embedding.py` | `embedding(input1, input2)` → immagine marchiata uint8 |
| `detection_groupname.py` | `detection(input1, input2, input3)` → `(0/1, WPSNR)`. File unico: WPSNR, estrazione e similarità sono dentro; usa solo numpy, scipy e OpenCV |
| `attacks.py` | `attacks(input1, attack_name, param_array)` + ricerca automatica dell'attacco migliore (bisezione) |
| `common.py` | WPSNR veloce, similarità, lettura immagini, i 6 attacchi (usato da attacks, roc, demo, check_rules) |
| `roc.py` | curva ROC, calcolo di TAU e verifica della TAU scritta nella detection |
| `demo.py` | prova completa: embedding → detection → attacchi → ricerca |
| `check_rules.py` | controlli delle regole della gara: deve finire con `TUTTO OK` |

## Cartelle e requisiti

```
ChallengeDeepFake/
├── clk.npy        # il nostro watermark
├── images/        # immagini di prova (solo per i test, non si consegnano)
└── ctm/           # questo repository (i file sopra)
```

Python 3.10 con numpy, scipy, opencv-python, scikit-learn, matplotlib (venv `..\.venv`).
Tutti i comandi si lanciano da `ctm/`.

## Comandi

```
..\.venv\Scripts\python.exe check_rules.py     # controlli delle regole
..\.venv\Scripts\python.exe demo.py 0042       # demo su images/0042.bmp
..\.venv\Scripts\python.exe roc.py             # ROC e verifica di TAU (circa 3 minuti)
```

## Giorno della gara

**Difesa (8:30-10:00)**
1. Per ognuna delle 3 immagini: `embedding("img.bmp", "../clk.npy")`, salvata come `NOME_img.bmp`.
2. Controllare il WPSNR, lanciare `check_rules.py`.
3. Caricare le 3 immagini e `detection_NOME.py`. Dopo le 10:00 non si cambia più nulla.

**Attacco (10:15-12:30)**
```
..\.venv\Scripts\python.exe attacks.py --our NOME --targets T --detectors D --originals O
```
- `T`: immagini degli altri gruppi (`gruppoB_img.bmp`); `D`: le loro `detection_gruppoB.py/.pyc`;
  `O`: le 3 originali (servono solo come argomento della loro detection).
- Caricare i file `NOME_gruppoB_img.bmp` dalla cartella `attacked/`; il log per l'Excel è
  `attack_log.csv`.

## Prima della consegna
Rinominare `detection_groupname.py` in `detection_NOME.py` e aggiornare i riferimenti in
`roc.py`, `demo.py`, `embedding.py` (commento) e `check_rules.py`, più `GROUP` in `demo.py`.
