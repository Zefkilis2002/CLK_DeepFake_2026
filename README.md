# Capture the Mark (Versione Avanzata)

Questa cartella contiene il software completo sviluppato per la challenge **Capture the Mark**.

Rispetto alla versione base con singoli script manuali, qui tutto è organizzato come una libreria Python modulare, robusta e automatizzata:
- **Automatica:** i file finali richiesti per la consegna vengono generati da soli a partire da template collaudati.
- **Veloce:** sfrutta il calcolo parallelo (multiprocessing su più CPU) e una cache su disco per non ripetere calcoli già eseguiti.
- **Ottimizzata:** i parametri ideali per nascondere il marchio e per la soglia di decisione sono già stati calcolati e salvati in profili pronti all'uso (**preset**).

Per la guida tecnica completa e il dettaglio di tutti i componenti, consulta [runbook/RUNBOOK.md](file:///f:/ChallengeDeepFake/ctm/runbook/RUNBOOK.md).

---

## 🚀 Guida rapida ai comandi

Tutti i comandi vanno eseguiti dal terminale all'interno di questa cartella (`ctm/`), utilizzando l'interprete Python del progetto (`..\.venv\Scripts\python.exe`):

### 1. Test e verifiche di funzionamento
```bash
# Esegue un test rapido per verificare che tutti i componenti funzionino a dovere
..\.venv\Scripts\python.exe -m tests.test_basic
```

### 2. Calibrazione e simulazioni (Opzionali)
```bash
# Esperimento iniziale E0
..\.venv\Scripts\python.exe -m scripts.run_e0

# Cerca in automatico i parametri migliori (forza del marchio alpha) per vari livelli di qualità
..\.venv\Scripts\python.exe -m scripts.run_embedding_search

# Calcola o ricalcola la soglia ottimale tau tramite la curva ROC per un dato preset
..\.venv\Scripts\python.exe -m scripts.run_roc P64

# Banco di prova: simula una sfida contro attacchi noti per valutare il punteggio
..\.venv\Scripts\python.exe -m scripts.run_bench 10
```

### 3. Preparazione della consegna
```bash
# Genera i file finali puliti da consegnare nella cartella submission/out/
..\.venv\Scripts\python.exe -m submission.build_submission --group NOME_GRUPPO

# Verifica in automatico che i file generati rispettino tutte le regole della gara
..\.venv\Scripts\python.exe -m submission.checks --group NOME_GRUPPO
```

### 4. Il giorno della sfida ufficiale
```bash
# FASE DIFESA (mattina): marchia le 3 immagini ricevute e prepara la consegna
..\.venv\Scripts\python.exe -m scripts.defense_day --images CARTELLA_IMMAGINI --group NOME_GRUPPO

# FASE ATTACCO (dopo la difesa): attacca le immagini degli altri gruppi per rimuovere il loro marchio
..\.venv\Scripts\python.exe -m scripts.attack_day --targets T --detectors D --originals O --our NOME_GRUPPO
```

Entrambi gli script stampano e commentano ogni passo (STEP 0-6). Le immagini da caricare finiscono in
`challenge_day/defense_upload/` e `challenge_day/attack_upload/` (con `attack_log.csv` per l'Excel).
Attacco: `--only g1,g2` divide i gruppi tra i portatili, `--phase A` fa solo la passata veloce.

---

## 📁 Com'è organizzato il progetto

| Cartella / File | A cosa serve (in parole semplici) |
| :--- | :--- |
| **`config/`** | **Impostazioni e Profili:** contiene i percorsi, il nome del gruppo (`settings.py`) e i profili di parametri già pronti (`presets.json`). |
| **`ctm/common/`** | **Mattoni base:** lettura/scrittura immagini (`imgio`), calcolo qualità visiva (`wpsnr`), gestione marchio e similarità (`watermark`), suite degli attacchi (`attacks`). |
| **`ctm/schemes/`** | **Algoritmi di embedding:** il codice che inserisce ed estrae il marchio, basato su Spread Spectrum in DCT (`dct_ss.py`). |
| **`ctm/detector/`** | **Verificatore:** funzione (`core.py`) che calcola la similarità e decide se il marchio è presente o assente. |
| **`ctm/roc/`** | **Studio della soglia:** analizza statisticamente le immagini e disegna la curva ROC per calcolare la soglia ideale $\tau$. |
| **`ctm/search/`** | **Ricerca automatica:** algoritmi (bisezione e halving) per trovare la forza d'inserimento ($\alpha$) migliore per ogni livello di qualità. |
| **`ctm/infra/`** | **Velocità e affidabilità:** gestione della cache su disco, calcolo parallelo su più CPU e salvataggio dei risultati in CSV. |
| **`ctm/attack/`** | **Motore di attacco:** strategie per attaccare le immagini degli avversari e oracolo per verificare se abbiamo cancellato il loro marchio. |
| **`ctm/bench/`** | **Banco di prova:** simulatore per testare la nostra difesa contro attacchi avversari e stimare i punti. |
| **`scripts/`** | **Punti d'ingresso:** gli script eseguibili da terminale, uno per ogni attività specifica. |
| **`submission/`** | **Generatore di consegna:** inietta i parametri nei template e produce il pacchetto finale conforme alle regole (`submission/out/`). |
| **`runbook/`** | **Guida tecnica:** documentazione dettagliata con tutte le formule e le spiegazioni di gara (`RUNBOOK.md`). |
| **`tests/`** | **Test automatici:** verifiche rapide di integrità del codice (`test_basic.py`). |

---

## 📌 Le 5 Regole d'oro da ricordare

1. **Nessun numero magico nel codice:** tutti i parametri regolabili (come la forza $\alpha$ o la soglia $\tau$) risiedono solo in `config/presets.json` (il profilo attivo è indicato da `"active"`).
2. **Non modificare a mano i file in `submission/out/`:** quella cartella viene generata e sovrascritta automaticamente da `build_submission`. Eventuali modifiche vanno fatte nei moduli sorgente o nei template.
3. **La cartella `common/` è indipendente:** non importa nessun altro modulo del progetto (solo librerie standard come NumPy, OpenCV e SciPy).
4. **Isolamento dell'originale nell'attacco:** durante la fase di attacco il codice non conosce l'immagine originale avversaria (come da regolamento di gara); solo l'oracolo sa se il marchio è sopravvissuto.
5. **Cache e riproducibilità:** le cartelle `cache/` e `results/` sono generate automaticamente. Se desideri forzare un ricalcolo da zero di un esperimento lungo, basta svuotare o eliminare la cartella `cache/`.
