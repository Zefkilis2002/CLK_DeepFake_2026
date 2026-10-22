# Capture the Mark: challenge code

Watermarking with **additive Spread Spectrum in DCT** (LAB-3): ±ALPHA is added to the 1024
largest DCT coefficients of the image (DC excluded). Detection is non-blind: it compares the
mark extracted from the watermarked image with the one extracted from the attacked image (cosine similarity ≥ TAU).

## Parameters

| Parameter | Value | Where |
|---|---|---|
| ALPHA | 5.0 | `embedding.py` and `detection_groupname.py` (must match) |
| TAU | 0.1218 | `detection_groupname.py` |
| Coefficients used | 1024 (one per bit) | both files |

Expected results: embedding WPSNR around 64 dB (5 quality points). Our own attack search beats
us on average at around 38 dB (about 5 robustness points). TAU gives FPR 0 on the cases
without the mark. The full analyses (halving, presets, test bench) are on the `avanzata` branch.

## Files

| File | Role |
|---|---|
| `embedding.py` | `embedding(input1, input2)` → watermarked uint8 image |
| `detection_groupname.py` | `detection(input1, input2, input3)` → `(0/1, WPSNR)`. Single file: WPSNR, extraction and similarity are all inside; uses only numpy, scipy and OpenCV |
| `attacks.py` | `attacks(input1, attack_name, param_array)` + automatic search for the best attack (bisection) |
| `common.py` | fast WPSNR, similarity, image reading, the 6 attacks (used by attacks, roc, demo, check_rules) |
| `roc.py` | ROC curve, computation of TAU and check of the TAU written in the detection |
| `demo.py` | full run: embedding → detection → attacks → search |
| `check_rules.py` | competition rule checks: must end with `TUTTO OK` |
| `day_defense.py`, `day_attack.py` | the challenge day, step by step (see below) |

## Folders and requirements

```
ChallengeDeepFake/
├── clk.npy        # our watermark
├── images/        # test images (for testing only, not submitted)
└── ctm/           # this repository (the files above)
```

Python 3.10 with numpy, scipy, opencv-python, scikit-learn, matplotlib (venv `..\.venv`).
All commands are run from `ctm/`.

## Commands

```
..\.venv\Scripts\python.exe check_rules.py     # rule checks
..\.venv\Scripts\python.exe demo.py 0042       # demo on images/0042.bmp
..\.venv\Scripts\python.exe roc.py             # ROC and TAU check (about 3 minutes)
```

## Competition day

Two scripts follow the day step by step; each step is commented in the code and printed on screen.

| Phase | Script | Output |
|---|---|---|
| Defense (8:30-10:00) | `day_defense.py` | `challenge_day/defense_upload/`: `NAME_img.bmp` x3 + `detection_NAME.py` |
| Attack (10:15-12:30) | `day_attack.py` | `challenge_day/attack_upload/`: `NAME_groupB_img.bmp` + `attack_log.csv` |

```
..\.venv\Scripts\python.exe day_defense.py --group NAME --images FOLDER_WITH_3_IMAGES
..\.venv\Scripts\python.exe day_attack.py --group NAME --targets T --detections D --originals O
```
- `day_defense.py`: checks names and ALPHA, embeds, prints WPSNR and quality points, tests our
  detection on the 3 images (watermarked 1, original 0, other watermark 0, destroyed 0, light
  attack 1, time, no prints), copies the detection file for the upload.
- `day_attack.py`: `T` = the other groups' images (`groupB_img.bmp`), `D` = their
  `detection_groupB.py/.pyc`, `O` = the 3 originals (only passed to their detection). It checks
  every detection, then phase A (one quick success per image) and phase B (lightest attack that
  still removes the mark), re-checks the saved files and writes the Excel log.
  `--only g1,g2` splits the groups between laptops; `--phase A` stops after the quick pass.
- Simulation: on the `challenge_sim` branch use `--images challenge_sim/defense` and
  `--targets challenge_sim/attack/images --detections challenge_sim/attack/detections --originals challenge_sim/defense`.

## Before submission
Rename `detection_groupname.py` to `detection_NAME.py` and update the references in
`roc.py`, `demo.py`, `embedding.py` (comment) and `check_rules.py`, plus `GROUP` in `demo.py`.
