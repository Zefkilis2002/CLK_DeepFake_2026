"""
CHALLENGE DAY, DEFENSE PHASE (8:30 - 10:00), advanced version.

Competition rules for this phase:
  - download the 3 challenge images (512x512 grayscale BMP);
  - embed our watermark and upload <group>_<image>.bmp (points from
    WPSNR(original, watermarked): >= 66 dB = 6 points; re-uploads allowed before 10:00);
  - upload our detection file detection_<group>.py (used by the other groups);
  - after 10:00 nothing can be changed (2-point penalty).

Difference from the basic version (main): parameters are not written by hand.
The script tries the PRESETS in config/presets.json (P66, P64, P62, P58: same
scheme, different alpha/tau) and GENERATES the detection file from the chosen
preset with submission/build_submission.py.

Steps:
  STEP 0  checks: group name, presets with templates (only dct_ss can be delivered)
  STEP 1  reads the 3 downloaded images
  STEP 2  tries the presets from the highest WPSNR target down and keeps the first
          one whose quality points hold on ALL 3 images
  STEP 3  saves <group>_<image>.bmp in the upload folder
  STEP 4  marks the preset as active and generates submission/out/ (embedding,
          detection_<group>.py, roc, attacks, demo, README) from it
  STEP 5  compliance checks on the generated files (submission/checks.py) and a
          self-test of the generated detection on the 3 challenge images
  STEP 6  copies detection_<group>.py next to the images and prints what to upload

Usage (from the repository root, ctm/):
  ..\.venv\Scripts\python.exe -m scripts.defense_day --images FOLDER_WITH_3_IMAGES --group NAME
  Simulation (images from the challenge_sim branch):
  ..\.venv\Scripts\python.exe -m scripts.defense_day --images PATH\challenge_sim\defense --group NAME
"""
import argparse
import contextlib
import io
import os
import shutil
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from config.settings import GROUP, MARK_PATH, REPO_DIR, SUBMISSION_OUT, load_presets, save_presets
from ctm.attack.oracle import load_detection
from ctm.bench.run import embedding_points
from ctm.common.attacks import awgn, jpeg
from ctm.common.imgio import list_images, read_gray, write_bmp
from ctm.common.watermark import load_mark, random_mark
from ctm.common.wpsnr import wpsnr
from ctm.schemes import get_scheme
from submission import checks
from submission.build_submission import build


def main():
    ap = argparse.ArgumentParser(description="Defense phase of the challenge day (advanced)")
    ap.add_argument("--images", required=True, help="folder with the 3 downloaded images")
    ap.add_argument("--group", default=GROUP, help="our group nickname")
    ap.add_argument("--mark", default=MARK_PATH, help="our watermark .npy")
    ap.add_argument("--out", default=os.path.join(REPO_DIR, "challenge_day", "defense_upload"))
    a = ap.parse_args()

    # ------------------------------------------------------------------
    # STEP 0. Checks
    # ------------------------------------------------------------------
    print("STEP 0  checks")
    if not (a.group.isalnum() and a.group == a.group.lower()):
        sys.exit("  the group name must be lowercase letters/digits only")
    data, bits = load_presets(), load_mark(a.mark)
    presets = sorted(((n, p) for n, p in data["presets"].items() if p["scheme"] == "dct_ss"),
                     key=lambda np_: -np_[1]["target_wpsnr"])       # highest quality first
    print("  group %s | presets to try: %s" % (a.group, ", ".join(n for n, _ in presets)))

    # ------------------------------------------------------------------
    # STEP 1. The downloaded images
    # ------------------------------------------------------------------
    print("\nSTEP 1  challenge images in %s" % a.images)
    originals = {p: read_gray(p) for p in list_images(a.images)}
    for p, img in originals.items():
        print("  %-20s %s %s" % (os.path.basename(p), img.shape, img.dtype))
    if len(originals) != 3:
        print("  WARNING: %d images found, the challenge gives 3" % len(originals))

    # ------------------------------------------------------------------
    # STEP 2. Choose the preset: highest target whose points hold on all 3 images
    # ------------------------------------------------------------------
    print("\nSTEP 2  choosing the preset (WPSNR per image)")
    chosen = None
    for name, preset in presets:
        scheme = get_scheme(preset["scheme"])
        marked = {p: scheme.embed_image(o, bits, preset["params"]) for p, o in originals.items()}
        w = [wpsnr(originals[p], m) for p, m in marked.items()]
        needed = embedding_points(preset["target_wpsnr"])
        ok = all(embedding_points(x) >= needed for x in w)
        print("  %s alpha %-8s %s -> %s" % (name, preset["params"]["alpha"],
                                          "  ".join("%.2f" % x for x in w),
                                          "OK, %d points per image" % needed if ok else "below its band"))
        if ok:
            chosen = (name, preset, marked)
            break
    if chosen is None:
        sys.exit("  no preset stays in its band: choose one by hand in config/presets.json")
    name, preset, marked = chosen

    # ------------------------------------------------------------------
    # STEP 3. Save the images with the competition names
    # ------------------------------------------------------------------
    print("\nSTEP 3  saving into %s" % a.out)
    os.makedirs(a.out, exist_ok=True)
    saved = {}
    for p, m in marked.items():
        out_path = os.path.join(a.out, "%s_%s" % (a.group, os.path.basename(p)))
        write_bmp(out_path, m)
        saved[p] = out_path
        print("  saved", os.path.basename(out_path))

    # ------------------------------------------------------------------
    # STEP 4. Active preset + generated delivery files
    # ------------------------------------------------------------------
    data["active"] = name
    save_presets(data)
    build(name, a.group)
    print("\nSTEP 4  active preset %s (alpha %s, tau %s); files generated in %s"
          % (name, preset["params"]["alpha"], preset["tau"], SUBMISSION_OUT))

    # ------------------------------------------------------------------
    # STEP 5. Compliance checks + self-test on the challenge images
    # ------------------------------------------------------------------
    print("\nSTEP 5a compliance checks on the generated files (test images)")
    with contextlib.redirect_stdout(io.StringIO()) as report:
        passed = checks.main(a.group, 10)
    print("  " + ("all checks passed" if passed else "FAILED:\n" + report.getvalue()))

    print("STEP 5b generated detection on the 3 challenge images (expected value in brackets)")
    det_path = os.path.join(SUBMISSION_OUT, "detection_%s.py" % a.group)
    detection = load_detection(det_path)                     # loaded like the other groups will
    tmp, times, problems = os.path.join(a.out, "_tmp_test.bmp"), [], 0
    scheme = get_scheme(preset["scheme"])
    for p, wm_path in saved.items():
        orig, wm = originals[p], read_gray(wm_path)
        cases = [("watermarked", wm, 1), ("original", orig, 0),
                 ("other watermark", scheme.embed_image(orig, random_mark(7), preset["params"]), 0),
                 ("JPEG 90 (light)", jpeg(wm, 90), 1), ("AWGN 150 (destroyed)", awgn(wm, 150), 0)]
        for case, img, expected in cases:
            write_bmp(tmp, img)
            out = io.StringIO()
            t = time.time()
            with contextlib.redirect_stdout(out):
                present, w = detection(p, wm_path, tmp)
            times.append(time.time() - t)
            ok = present == expected and not out.getvalue()
            problems += not ok
            print("  %-12s %-22s -> %d (%d)  WPSNR %6.2f %s" % (os.path.basename(p), case, present,
                                                              expected, min(w, 999), "" if ok else "<-- PROBLEM"))
    os.remove(tmp)
    print("  slowest detection %.2f s (limit 5 s)" % max(times))
    if problems or not passed or max(times) >= 5:
        print("  STOP: fix the problems before uploading")

    # ------------------------------------------------------------------
    # STEP 6. What to upload
    # ------------------------------------------------------------------
    shutil.copy(det_path, a.out)
    print("\nSTEP 6  upload from %s:" % a.out)
    for f in sorted(os.listdir(a.out)):
        print("  -", f)
    print("  After 10:00 do not change any uploaded file.")


if __name__ == "__main__":
    main()
