# Consegna del gruppo groupname

File generati dal preset **P66** (`dct_ss`, parametri `{'alpha': 4.360352, 'mode': 'additive', 'n': 1024}`, TAU = 0.1214).

| File | Contenuto |
|---|---|
| `embedding.py` | `embedding(input1, input2)`: Spread Spectrum in DCT (LAB-3), file autonomo |
| `detection_groupname.py` | `detection(input1, input2, input3)`: file unico, solo numpy/scipy/OpenCV, WPSNR incluso |
| `roc.py` | calcolo di TAU con la ROC: `python roc.py <cartella_immagini> <watermark.npy>` |
| `attacks.py` | `attacks(input1, attack_name, param_array)` e ricerca automatica (`python attacks.py --help`) |
| `demo.py` | `python demo.py <immagine.bmp> <watermark.npy>` |
| `ctm/`, `config/` | codice comune usato da roc, attacchi e demo (non dalla detection) |

Attacchi ammessi: `awgn` (sigma), `blur` (sigma), `sharp` (alpha), `jpeg` (QF), `resize` (scala),
`median` (k), anche in sequenza: `attacks(path, ["blur", "jpeg"], [1.0, 70])`.
Le funzioni di attacco ricevono solo l'immagine marchiata; l'originale è usato soltanto come
argomento della detection dell'altro gruppo per sapere se l'attacco è riuscito.

Requisiti: Python 3.10, numpy, scipy, opencv-python, scikit-learn, matplotlib (PyWavelets solo
per gli schemi di confronto in `ctm/schemes/dwt_hvs.py`).
