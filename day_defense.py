"""
CHALLENGE DAY, DEFENSE PHASE (8:30 - 10:00)

What happens in this phase (competition rules):
  - we download the 3 challenge images (512x512 grayscale BMP) from the website;
  - we embed OUR watermark (clk.npy) into each of them;
  - we upload the 3 watermarked images, named <group>_<image>.bmp
    (points depend on WPSNR(original, watermarked): >= 66 dB = 6 points);
  - we upload our detection file, detection_<group>.py, which the other
    groups will use during the attack phase;
  - after 10:00 NOTHING can be changed (2-point penalty).

What this script does, step by step:
  STEP 0  preliminary checks: detection file name, same ALPHA in embedding
          and detection, watermark file
  STEP 1  reads the 3 downloaded images
  STEP 2  embeds the watermark and saves <group>_<image>.bmp in the upload folder
  STEP 3  measures WPSNR(original, watermarked) and the quality points
  STEP 4  tests our detection on these exact images, as the teaching
          assistants and the other groups will use it (rules checked:
          watermarked -> 1, original -> 0, other watermark -> 0,
          destroyed image -> 0, light attack -> 1, < 5 s, no prints)
  STEP 5  copies detection_<group>.py into the upload folder
  STEP 6  prints what to upload

Usage (from the ctm/ folder):
  ..\.venv\Scripts\python.exe day_defense.py --group groupname --images FOLDER_WITH_3_IMAGES

  Simulation (challenge_sim branch):
  ..\.venv\Scripts\python.exe day_defense.py --group groupname --images challenge_sim/defense

Options:
  --mark   watermark file (default ../clk.npy)
  --out    upload folder  (default challenge_day/defense_upload)
"""
import argparse
import contextlib
import importlib.util
import io
import os
import shutil
import sys
import time

import cv2
import numpy as np

import embedding
from common import CTM_DIR, MARK_PATH, awgn, jpeg, list_images

# quality points of the competition: (minimum WPSNR, points)
EMBEDDING_POINTS = [(66, 6), (62, 5), (58, 4), (54, 3), (50, 2), (35, 1)]


def quality_points(w):
    return next((p for limit, p in EMBEDDING_POINTS if w >= limit), 0)


def silent(fn, *args):
    """Runs fn capturing anything it prints (the rules forbid prints)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        result = fn(*args)
    return result, out.getvalue()


def main():
    ap = argparse.ArgumentParser(description="Defense phase of the challenge day")
    ap.add_argument("--group", required=True, help="our group nickname (lowercase, no symbols)")
    ap.add_argument("--images", required=True, help="folder with the 3 downloaded images")
    ap.add_argument("--mark", default=MARK_PATH, help="our watermark .npy")
    ap.add_argument("--out", default=os.path.join(CTM_DIR, "challenge_day", "defense_upload"))
    a = ap.parse_args()

    # ------------------------------------------------------------------
    # STEP 0. Preliminary checks
    # ------------------------------------------------------------------
    print("STEP 0  preliminary checks")
    if not (a.group.isalnum() and a.group == a.group.lower()):
        sys.exit("  the group name must be lowercase letters/digits only")
    det_path = os.path.join(CTM_DIR, "detection_%s.py" % a.group)
    if not os.path.exists(det_path):
        sys.exit("  %s not found: rename detection_groupname.py to detection_%s.py" % (det_path, a.group))
    # load the file from its path, exactly like the other groups will do
    spec = importlib.util.spec_from_file_location("detection_%s" % a.group, det_path)
    consts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(consts)
    detection = consts.detection
    if consts.ALPHA != embedding.ALPHA:
        sys.exit("  ALPHA differs: embedding.py %s, detection %s" % (embedding.ALPHA, consts.ALPHA))
    bits = np.load(a.mark)
    if bits.size != 1024 or set(np.unique(bits)) - {0, 1}:
        sys.exit("  the watermark must be 1024 bits 0/1")
    print("  OK: %s, ALPHA = %s, TAU = %s, watermark %s" % (os.path.basename(det_path), consts.ALPHA,
                                                         consts.TAU, os.path.basename(a.mark)))

    # ------------------------------------------------------------------
    # STEP 1. The downloaded images
    # ------------------------------------------------------------------
    print("\nSTEP 1  challenge images in %s" % a.images)
    paths = list_images(a.images)
    if len(paths) != 3:
        print("  WARNING: %d images found, the challenge gives 3" % len(paths))
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        print("  %-20s %s %s" % (os.path.basename(p), img.shape, img.dtype))

    # ------------------------------------------------------------------
    # STEP 2. Embedding and saving with the competition names
    # ------------------------------------------------------------------
    print("\nSTEP 2  embedding -> %s" % a.out)
    os.makedirs(a.out, exist_ok=True)
    marked = {}
    for p in paths:
        wm, printed = silent(embedding.embedding, p, a.mark)     # embedding(input1, input2)
        out_path = os.path.join(a.out, "%s_%s" % (a.group, os.path.basename(p)))
        cv2.imwrite(out_path, wm)
        marked[p] = out_path
        print("  saved %s%s" % (os.path.basename(out_path), "  (WARNING: embedding printed!)" if printed else ""))

    # ------------------------------------------------------------------
    # STEP 3. Quality: WPSNR(original, watermarked) with the official lab code
    # ------------------------------------------------------------------
    print("\nSTEP 3  quality (WPSNR original vs watermarked)")
    points = []
    for p, out_path in marked.items():
        w = consts.wpsnr(cv2.imread(p, 0), cv2.imread(out_path, 0))
        points.append(quality_points(w))
        print("  %-20s %.2f dB -> %d points" % (os.path.basename(p), w, points[-1]))
    print("  average quality points: %.2f" % np.mean(points))

    # ------------------------------------------------------------------
    # STEP 4. Our detection on these images (what the others will see)
    # ------------------------------------------------------------------
    print("\nSTEP 4  detection self-test (expected value in brackets)")
    tmp = os.path.join(a.out, "_tmp_test.bmp")
    failures, times = 0, []
    rng = np.random.default_rng(0)
    for p, wm_path in marked.items():
        orig, wm = cv2.imread(p, 0), cv2.imread(wm_path, 0)
        other = embedding.embed_dct_ss(orig, rng.integers(0, 2, 1024))   # another group's mark
        cases = [("watermarked", wm, 1), ("original", orig, 0), ("other watermark", other, 0),
                 ("JPEG 90 (light)", jpeg(wm, 90), 1), ("AWGN 150 (destroyed)", awgn(wm, 150), 0)]
        for name, img, expected in cases:
            cv2.imwrite(tmp, img)
            t = time.time()
            (present, w), printed = silent(detection, p, wm_path, tmp)
            times.append(time.time() - t)
            ok = present == expected and not printed
            failures += not ok
            print("  %-12s %-22s -> %d (%d)  WPSNR %6.2f  %s" % (os.path.basename(p), name, present,
                                                               expected, min(w, 999), "" if ok else "<-- PROBLEM"))
    os.remove(tmp)
    print("  slowest detection: %.2f s (limit 5 s)" % max(times))
    if failures or max(times) >= 5:
        print("  STOP: fix the problems before uploading")

    # ------------------------------------------------------------------
    # STEP 5. Detection file for the upload
    # ------------------------------------------------------------------
    shutil.copy(det_path, a.out)
    print("\nSTEP 5  copied %s into the upload folder" % os.path.basename(det_path))

    # ------------------------------------------------------------------
    # STEP 6. What to upload
    # ------------------------------------------------------------------
    print("\nSTEP 6  upload from %s:" % a.out)
    for f in sorted(os.listdir(a.out)):
        print("  -", f)
    print("  Upload the images (you can re-upload them before 10:00) and the detection file.")
    print("  After 10:00 do not change any uploaded file.")


if __name__ == "__main__":
    main()
