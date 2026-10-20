# Capture the Mark: codice della challenge

Implementazione del piano `docs/analisi/ctm_design_plan.md`, in versione semplificata e basata
sui laboratori. Tutti gli script si lanciano **da questa cartella** (`ctm/`) con il venv del progetto:

```
cd ctm
..\.venv\Scripts\python.exe demo.py
```

## File

| File | Cosa fa | Da dove viene | Componente del piano |
|---|---|---|---|
| `embedding.py` | `embedding(input1, input2)`: Spread Spectrum **additivo** in DCT sui 1024 coefficienti più grandi | LAB-3 (`LAB3-ROC.ipynb`, `v='additive'`) | C6 (M2) |
| `detection_groupname.py` | `detection(input1, input2, input3)`: file unico con ALPHA, TAU, WPSNR, estrazione, similarità | LAB-3 + struttura LAB-4 | C7 |
| `roc.py` | calcolo di TAU con la ROC, una volta sola; salva il grafico in `results/` | LAB-3 ROC Example 2 + scelta di tau del LAB-4 | C8 |
| `attacks.py` | `attacks(input1, attack_name, param_array)` + ricerca automatica dell'attacco + modalità "giorno della gara" | LAB-3, LAB-4 (blur + sharpening) | C5, C11, C12 |
| `demo.py` | embedding, detection, attacchi e ricerca su un'immagine | tutti | richiesto dalle regole |
| `common.py` | lettura immagini, WPSNR, similarità, i 6 attacchi, `random_attack` | LAB-1..4 | C2-C5 |
| `schemes.py` | i 4 schemi dei laboratori con la stessa interfaccia (bit-plane, SS additivo, SS moltiplicativo, DWT-HVS) | LAB-3, LAB-4 | C6 |
| `e0_quality.py` | esperimento E0: WPSNR, pixel cambiati, similarità al variare di alpha | piano §6 | E0 |
| `bench.py` | banco di prova: con che WPSNR la nostra ricerca rompe il nostro schema e gli schemi dei lab | piano C13 | C13 |
| `check_rules.py` | controlli di conformità alle regole (da rilanciare a ogni cambio di ALPHA/TAU) | piano C14 | C14 |

Prima della consegna: rinominare `detection_groupname.py` in `detection_<nostrogruppo>.py` e
aggiornare le righe `import detection_groupname` (in `schemes.py`, `roc.py`, `bench.py`, `demo.py`)
e `GROUP` in `demo.py`.

## Parametri scelti (preset)

Misure su 101 immagini di prova (`images/`); ROC su 40 immagini di calibrazione + 20 di verifica,
banco di prova su 10 immagini mai usate dalla ROC.

| Preset | ALPHA | WPSNR embedding (medio / minimo) | Punti qualità | TAU | TPR verifica | Battuto in media a | Punti robustezza stimati |
|---|---|---|---|---|---|---|---|
| **P64 (attivo)** | **5.0** | 64.1 / 63.6 dB | 5 | **0.1253** | 0.92 | 38.7 dB | 5 |
| P68 | 4.0 | 68.0 / 67.4 dB | 6 | 0.1246 | 0.72 | 42.6 dB | 4 |

Per cambiare preset: stesso ALPHA in `embedding.py` **e** in `detection_groupname.py`, TAU nella
detection, poi `python check_rules.py`. Per un ALPHA nuovo: `python roc.py <alpha>` e copiare la TAU.

Con la soglia scelta: FPR 0 su 1101 campioni H0 di calibrazione e su quelli di verifica;
nessun falso positivo con 100 watermark diversi, immagini diverse, originali e immagini distrutte.
La detection impiega circa 0.2-0.5 s.

## Come si usa

```
python e0_quality.py          # qualità per schema e alpha
python roc.py                 # TAU per l'ALPHA di embedding.py (grafico in results/)
python check_rules.py         # controlli delle regole: deve finire con TUTTO OK
python bench.py 10            # robustezza su 10 immagini + prova degli attacchi sui lab
python demo.py 0042           # demo su images/0042.bmp, salva out/groupname_0042.bmp
```

### Attacchi

```python
from attacks import attacks
img = attacks("groupb_lena.bmp", ["blur", "jpeg"], [1.0, 70])   # blur sigma 1, poi JPEG QF 70
```
Nomi: `awgn` (sigma), `blur` (sigma), `sharp` (alpha), `jpeg` (QF), `resize` (scala), `median` (k).

Il giorno della gara, con le immagini scaricate in `targets/` (`gruppoB_immagine.bmp`), le detection
in `detections/` (`detection_gruppoB.py` o `.pyc`) e le 3 originali in `originals/`:

```
python attacks.py --our groupname --targets targets --detectors detections --originals originals
```
Per ogni immagine salva in `attacked/` il miglior attacco verificato (`groupname_gruppoB_immagine.bmp`)
e aggiunge una riga a `attack_log.csv` (Image; Group; WPSNR; Attack(s)), da aprire con Excel per il log.

**Uso dell'originale:** le funzioni di attacco ricevono solo l'immagine marchiata. Il percorso
dell'originale compare solo in `make_oracle`, come primo argomento della detection dell'altro gruppo,
che ci dice soltanto se l'attacco è riuscito (lettura confermata dall'assistente, piano §6 punto 15).

## Procedura del giorno della gara (C15, condensata)

1. **Difesa:** embedding delle 3 immagini con il preset attivo, controllo del WPSNR, upload.
   Se serve cambiare preset, cambiare ALPHA e TAU nei due file, `check_rules.py`, poi upload della detection.
   Dopo la fase di difesa non si cambia più nessun file.
2. **Attacco:** prima una passata su tutti i gruppi (punti attività), poi si rilancia sulle immagini
   con il WPSNR più basso. Il log Excel si costruisce da `attack_log.csv`.

## Scelte rispetto al piano

Vedi la risposta nel thread del progetto per la verifica completa. In breve:
- **Schema:** SS additivo in DCT invece del moltiplicativo, per i motivi in testa a `embedding.py`.
- **WPSNR:** stessa formula del lab, ma la convoluzione è fatta con la FFT: risultato identico
  (verificato in `check_rules.py`), 0.1 s invece di 3 s, lontano dal limite di 5 s.
- **Tau:** TPR massimo con FPR ≤ 0.01 (dentro [0, 0.1] delle regole) e mai sotto media(H0) + 4·std(H0).
- **Ricerca attacchi:** per ogni catena, forza massima con WPSNR ≥ 35 calcolata localmente e poi
  bisezione con la detection; il WPSNR locale evita query inutili (idea del piano C12, più semplice).
- **Infrastruttura semplificata:** niente package, preset JSON, cache, multiprocessing, halving,
  generatore di template, subprocess con timeout: file piatti, costanti in cima ai file, tabella dei preset qui sopra.
