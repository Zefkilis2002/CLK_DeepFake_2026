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

**Defense (8:30-10:00)**
1. For each of the 3 images: `embedding("img.bmp", "../clk.npy")`, saved as `NAME_img.bmp`.
2. Check the WPSNR, run `check_rules.py`.
3. Upload the 3 images and `detection_NAME.py`. After 10:00 nothing can be changed.

**Attack (10:15-12:30)**
```
..\.venv\Scripts\python.exe attacks.py --our NAME --targets T --detectors D --originals O
```
- `T`: the other groups' images (`groupB_img.bmp`); `D`: their `detection_groupB.py/.pyc`;
  `O`: the 3 originals (only needed as an argument to their detection).
- Upload the `NAME_groupB_img.bmp` files from the `attacked/` folder; the log for the Excel sheet is
  `attack_log.csv`.

## Before submission
Rename `detection_groupname.py` to `detection_NAME.py` and update the references in
`roc.py`, `demo.py`, `embedding.py` (comment) and `check_rules.py`, plus `GROUP` in `demo.py`.
