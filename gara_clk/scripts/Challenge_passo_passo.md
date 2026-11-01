# Challenge: procedura passo per passo

Questa guida è aggiornata alla separazione tra `da_consegnare/` e
`gara_clk/scripts/`. La posizione della guida non determina la cartella da cui
lanciare i comandi: **eseguili da `F:\ChallengeDeepFake\gara_clk`, non da `scripts/`.**

## Avvio rapido con la struttura attuale

```powershell
Set-Location F:\ChallengeDeepFake
.\.venv\Scripts\Activate.ps1
Set-Location .\gara_clk

# Prima della gara: prova generale con i dati simulati
python .\scripts\gara_prova.py

# Durante la difesa, dopo aver inserito le tre BMP in originali/
python .\scripts\gara_difesa.py

# Durante l'attacco, dopo aver scaricato immagini e detector avversari
python .\scripts\gara_attacco.py

# Per limitare l'attacco ai gruppi indicati
python .\scripts\gara_attacco.py --gruppi falco koala
```

I comandi sono elencati per fase: avvia quello della fase corrente.

| Percorso | Uso |
|---|---|
| `F:\ChallengeDeepFake\da_consegnare\` | Embedding, detection, attacks, ROC, run_roc e PNG; i moduli sono importati automaticamente dagli strumenti della gara |
| `F:\ChallengeDeepFake\gara_clk\scripts\` | I tre comandi operativi e questa guida |
| `F:\ChallengeDeepFake\gara_clk\clk.npy` | Watermark usato dalla difesa |
| `F:\ChallengeDeepFake\gara_clk\originali\` | Le tre immagini originali scaricate |
| `F:\ChallengeDeepFake\gara_clk\immagini_avversari\` | Immagini marchiate degli altri gruppi |
| `F:\ChallengeDeepFake\gara_clk\detection_avversari\` | Detector degli altri gruppi |
| `F:\ChallengeDeepFake\gara_clk\da_caricare\` | Immagini di difesa e copia del detector da caricare |
| `F:\ChallengeDeepFake\gara_clk\attaccate\` | Immagini attaccate e log |

Mantieni `da_consegnare/` e `gara_clk/` affiancate nella radice del progetto.
Non occorre copiare i moduli Python dentro `scripts/`.

**Prerequisito da ripristinare:** il `.venv` presente punta a Python 3.10 in
`C:\Users\kozeu\AppData\Local\Programs\Python\Python310\python.exe`, attualmente
assente. I percorsi della guida sono coerenti con la struttura attuale, ma
l'esecuzione completa richiede prima il ripristino dell'ambiente dei laboratori.

I valori di WPSNR, i tempi e i messaggi riportati sotto sono esempi di esecuzioni
precedenti, non risultati appena verificati. Per gli orari definitivi consultare
le comunicazioni della challenge; le differenze tra le fonti sono annotate nella
[documentazione](../../docs/README.md).

---

##### ***DEFENSE PHASE***



1. ***Activate the virtual enviroment!***



|PS F:\\ChallengeDeepFake> ***.\\.venv\\Scripts\\Activate.ps1***|
|-|

(.venv) PS F:\\ChallengeDeepFake>





***2. Move to "gara\_clk" folder***



|(.venv) PS F:\\ChallengeDeepFake> ***cd gara\_clk***|
|-|





***3. Download the 3 images of the challenge and put them into the "originali" folder***



F:\\ChallengeDeepFake\\gara\_clk\\originali\\harbour.bmp

F:\\ChallengeDeepFake\\gara\_clk\\originali\\market.bmp

F:\\ChallengeDeepFake\\gara\_clk\\originali\\tower.bmp





***4. Execute the defense script: it generates the watermarked images and tests our detection\_clk.py***



|(.venv) PS F:\\ChallengeDeepFake\\gara\_clk> ***python .\\scripts\\gara\_difesa.py***|
|-|

=== EMBEDDING ===

[PASS] harbour.bmp letta, 512x512 in scala di grigi

[PASS] clk\_harbour.bmp salvata e riletta identica

clk\_harbour.bmp: WPSNR = 63.90 dB -> 5 punti di qualita'

[PASS] clk\_harbour.bmp: WPSNR >= 35 dB

...

clk\_market.bmp: WPSNR = 63.95 dB -> 5 punti di qualita'

...

clk\_tower.bmp: WPSNR = 63.98 dB -> 5 punti di qualita'

...

=== CONTROLLI DETECTION: clk\_harbour.bmp ===

*[PASS] a. marchiata come attaccata -> output1=1 (atteso 1), WPSNR=9999999.00 dB*

*[PASS] b. originale come attaccata -> output1=0 (atteso 0), WPSNR=63.90 dB*

*[INFO] c. attacco jpeg [70] -> output1=1, WPSNR=56.16 dB*

*[INFO] c. attacco blur [0.6] -> output1=1, WPSNR=56.08 dB*

*[INFO] c. attacco awgn [4] -> output1=1, WPSNR=51.27 dB*

*[INFO] c. attacco median [3] -> output1=1, WPSNR=52.25 dB*

*[INFO] c. attacco resize [0.8] -> output1=1, WPSNR=60.28 dB*

*[INFO] c. attacco sharp [0.5] -> output1=1, WPSNR=54.44 dB*

*[PASS] d. immagine distrutta (awgn 100) -> output1=0 (atteso 0), WPSNR=24.34 dB*

*[PASS] d. l'immagine distrutta ha davvero WPSNR <= 25*

*[PASS] e. immagine scorrelata market.bmp -> output1=0 (atteso 0), WPSNR=16.30 dB*

*[PASS] e. immagine scorrelata tower.bmp -> output1=0 (atteso 0), WPSNR=16.36 dB*

*[PASS] f. stessa immagine, chiave casuale -> output1=0 (atteso 0), WPSNR=60.90 dB*

...

=== RIEPILOGO ===

Tempo detection: massimo 1.64 s, medio 1.27 s

TUTTI I CONTROLLI PASSATI.

File da caricare sul sito (cartella F:\\ChallengeDeepFake\\gara\_clk\\da\_caricare):

\- clk\_harbour.bmp

\- clk\_market.bmp

\- clk\_tower.bmp

\- detection\_clk.py



**WHAT *gara\_difesa.py DOES?***

\- Reads the 3 original images from originali/ and our secret key clk.npy.

\- Adds the watermark using embedding\_clk.py.

\- Saves the marked images into the da\_caricare/ folder with the name of the rules: clk\_imageName.bmp (and checks that the image read back from disk is identical).

\- Computes and prints WPSNR (\~64 dB) and the quality points (62-66 dB = 5 points, >= 66 dB = 6 points).

\- Tests our detection\_clk.py on every image: watermarked -> 1, original -> 0, destroyed image (WPSNR <= 25) -> 0, the other 2 originals -> 0, same image with a random key -> 0, plus some light attacks just to see the robustness.

\- Checks that every detection call takes less than 5 seconds and prints nothing.

\- Copies detection\_clk.py from the project folder da\_consegnare/ into gara\_clk/da\_caricare/.



***5. Upload the files before 10:00***



Only if the last lines say "TUTTI I CONTROLLI PASSATI", upload from F:\\ChallengeDeepFake\\gara\_clk\\da\_caricare:

\- clk\_harbour.bmp, clk\_market.bmp, clk\_tower.bmp

\- detection\_clk.py

After the end of the defense phase NOTHING can be changed (2 points penalty).





##### ***ATTACK PHASE***



***1. Download the images and the detections of the other groups***



Images (groupB\_imageName.bmp) into F:\\ChallengeDeepFake\\gara\_clk\\immagini\_avversari\\

Detections (detection\_groupB.py or .pyc) into F:\\ChallengeDeepFake\\gara\_clk\\detection\_avversari\\

The 3 originals stay in F:\\ChallengeDeepFake\\gara\_clk\\originali\\ (they are used only as input of the other groups' detections).





***2. Execute the attack script***



|(.venv) PS F:\\ChallengeDeepFake\\gara\_clk> ***python .\\scripts\\gara\_attacco.py***|
|-|

*falco\_harbour.bmp      OK  WPSNR 37.59 dB | blur 2.83 | 4 s*

*falco\_market.bmp       nessun attacco riuscito con WPSNR >= 35 (7 s)*

*koala\_harbour.bmp      OK  WPSNR 41.91 dB | median 5 + jpeg 25 | 7 s*

*lynx\_harbour.bmp       OK  WPSNR 47.84 dB | resize 0.46 + jpeg 46 | 7 s*

*orca\_harbour.bmp       OK  WPSNR 47.76 dB | resize 0.46 + jpeg 46 | 20 s*

*tundra\_market.bmp      OK  WPSNR 35.58 dB | blur 1.64 | 24 s*

*vespa\_market.bmp       OK  WPSNR 36.09 dB | median 3 + blur 1.36 + jpeg 41 | 34 s*

...

=== RIEPILOGO ATTACCHI ===

Gruppo     Immagine     WPSNR    Attacco

falco      harbour      37.59    blur 2.83

koala      harbour      41.91    median 5 + jpeg 25

lynx       harbour      47.84    resize 0.46 + jpeg 46

...

Gruppi attaccati: 7 su 7 (100%) -> 6 punti di attivita'

Log: F:\\ChallengeDeepFake\\gara\_clk\\attaccate\\attack\_log.csv

Da caricare: i file clk\_<gruppo>\_<immagine>.bmp nella cartella F:\\ChallengeDeepFake\\gara\_clk\\attaccate



If it stops or you need to run it again, just relaunch it: the images already attacked are skipped. To attack only some groups:

|(.venv) PS F:\\ChallengeDeepFake\\gara\_clk> ***python .\\scripts\\gara\_attacco.py --gruppi falco koala***|
|-|



**WHAT *gara\_attacco.py DOES?***

\- For every image groupB\_imageName.bmp loads the detection of that group (.py or .pyc).

\- Searches the attack with the highest WPSNR that removes the watermark, using only the 6 allowed attacks (awgn, blur, sharp, jpeg, resize, median), alone or combined.

\- The attack uses ONLY the watermarked image: the original is never used to build the attack, it is only the first input of the other group's detection.

\- Keeps only the best successful attack for each image (watermark removed and WPSNR >= 35) and saves it into attaccate/ as clk\_groupB\_imageName.bmp.

\- Checks again the saved file with their detection.

\- Writes the log attaccate/attack\_log.csv (Image, Group, WPSNR, Attack(s) with parameters) and prints the % of attacked groups (> 30% = 2 points, > 60% = 4, > 90% = 6).



***3. Upload the attacked images***



Upload from F:\\ChallengeDeepFake\\gara\_clk\\attaccate the files clk\_groupB\_imageName.bmp, indicating the attacked group.



***4. After the competition send the log***



Send attaccate/attack\_log.xlsx with the columns Image, Group, WPSNR, Attack(s) with parameters.
The script also writes attack\_log.csv. The XLSX is generated only if openpyxl is installed;
if it is missing, convert the CSV to an Excel workbook before submitting the required Excel log.
