"""
CHALLENGE DAY, ATTACK PHASE (10:15 - 12:30)

What happens in this phase (competition rules):
  - we download the other groups' watermarked images (<groupB>_<image>.bmp)
    and their detection files (detection_<groupB>.py or .pyc);
  - we attack each image with the 6 allowed attacks only (awgn, blur, sharp,
    jpeg, resize, median), using ONLY the watermarked image;
  - an attack succeeds if their detection returns 0 AND WPSNR >= 35 dB;
  - for each image we upload only the best success (highest WPSNR), named
    <ourgroup>_<groupB>_<image>.bmp;
  - points: % of groups attacked (>90% = 6), images above the average WPSNR,
    +2 for a group nobody else attacked. After the challenge an Excel log is
    required: Image | Group | WPSNR | Attack(s) with parameters.

The original images are passed ONLY to the other groups' detection (it is its
first argument): our attack functions never see them.

What this script does, step by step:
  STEP 0  inventory: matches each image with its detection file and original
  STEP 1  smoke test of every detection: it loads, finds its own mark, says 0
          on the original; measures its speed (a slow detection = slow search)
  STEP 2  PHASE A (breadth): one quick probe per attack chain on EVERY image,
          to get at least one success per group as early as possible
          (activity points); results are saved immediately
  STEP 3  PHASE B (depth): full search (bisection on the attack strength) for
          the lightest attack that still removes the mark, worst images first
  STEP 4  saves the best attack per image and re-checks the SAVED file with
          their detection (that is exactly what we upload)
  STEP 5  writes the log for the Excel sheet
  STEP 6  summary: groups attacked, activity points, WPSNR per image

Usage (from the ctm/ folder):
  ..\.venv\Scripts\python.exe day_attack.py --group groupname --targets FOLDER_IMAGES --detections FOLDER_DETECTIONS --originals FOLDER_3_ORIGINALS

  Simulation (challenge_sim branch):
  ..\.venv\Scripts\python.exe day_attack.py --group groupname --targets challenge_sim/attack/images --detections challenge_sim/attack/detections --originals challenge_sim/defense

Options:
  --only falco,koala   attack only these groups (to split the work between laptops)
  --phase A            stop after phase A (quick results); default: A then B
  --out                upload folder (default challenge_day/attack_upload)
"""
import argparse
import contextlib
import csv
import io
import os
import time

import cv2

from attacks import find_detection, load_detection, make_oracle, search_attack
from common import CTM_DIR, read_gray

ACTIVITY_POINTS = [(90, 6), (60, 4), (30, 2)]       # % of groups attacked -> points


def counted(oracle):
    """Counts how many times we ask the other group's detection."""
    def wrapper(img):
        wrapper.queries += 1
        return oracle(img)
    wrapper.queries = 0
    return wrapper


def main():
    ap = argparse.ArgumentParser(description="Attack phase of the challenge day")
    ap.add_argument("--group", required=True, help="our group nickname")
    ap.add_argument("--targets", required=True, help="folder with <groupB>_<image>.bmp")
    ap.add_argument("--detections", required=True, help="folder with detection_<groupB>.py/.pyc")
    ap.add_argument("--originals", required=True, help="folder with the 3 original images")
    ap.add_argument("--only", default="", help="comma-separated groups to attack")
    ap.add_argument("--phase", default="AB", choices=["A", "AB"])
    ap.add_argument("--out", default=os.path.join(CTM_DIR, "challenge_day", "attack_upload"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.abspath(os.path.join(a.out, "_tmp_attacked.bmp"))
    only = set(filter(None, a.only.split(",")))

    # ------------------------------------------------------------------
    # STEP 0. Inventory: image -> (group, image name, detection, original)
    # ------------------------------------------------------------------
    print("STEP 0  inventory")
    targets = []
    for fname in sorted(os.listdir(a.targets)):
        if not fname.lower().endswith(".bmp") or "_" not in fname:
            continue
        group, image = fname[:-4].split("_", 1)
        if group == a.group or (only and group not in only):
            continue
        det = find_detection(a.detections, group)
        orig = os.path.join(a.originals, image + ".bmp")
        if det is None or not os.path.exists(orig):
            print("  %-25s SKIPPED (%s missing)" % (fname, "detection" if det is None else "original"))
            continue
        targets.append({"group": group, "image": image, "det": det,
                        "wm": os.path.abspath(os.path.join(a.targets, fname)), "orig": os.path.abspath(orig)})
    groups = sorted({t["group"] for t in targets})
    print("  %d images from %d groups: %s" % (len(targets), len(groups), ", ".join(groups)))

    # ------------------------------------------------------------------
    # STEP 1. Smoke test of every detection
    # ------------------------------------------------------------------
    print("\nSTEP 1  checking the other groups' detections")
    detections, working = {}, set()
    for group in groups:
        t = next(x for x in targets if x["group"] == group)
        try:
            det = load_detection(t["det"])
            with contextlib.redirect_stdout(io.StringIO()):
                own = det(t["orig"], t["wm"], t["wm"])[0]          # their image: expected 1
                start = time.time()                                  # timed on a real comparison
                on_orig = det(t["orig"], t["wm"], t["orig"])[0]    # the original: expected 0
                seconds = time.time() - start
            detections[group] = det
            working.add(group)
            note = "" if (own, on_orig) == (1, 0) else "  <-- unexpected answers, attack anyway"
            print("  %-10s %-24s %.2f s per call | own image %d, original %d%s"
                  % (group, os.path.basename(t["det"]), seconds, own, on_orig, note))
        except Exception as e:                                       # broken file: skip, tell the TAs
            print("  %-10s CANNOT RUN (%s: %s)" % (group, type(e).__name__, e))
    targets = [t for t in targets if t["group"] in working]

    def attack(t, bisect_steps):
        """Search on one image; the original is used only inside the oracle."""
        oracle = counted(make_oracle(detections[t["group"]], t["orig"], t["wm"], tmp))
        best = search_attack(read_gray(t["wm"]), oracle, bisect_steps=bisect_steps)
        return best, oracle.queries

    best = {}            # (group, image) -> (image array, wpsnr, description)

    def keep(t, result, phase, queries):
        key = (t["group"], t["image"])
        old = best.get(key)
        if result is not None and (old is None or result[1] > old[1]):
            best[key] = result
            cv2.imwrite(os.path.join(a.out, "%s_%s_%s.bmp" % (a.group, t["group"], t["image"])), result[0])
            print("  %s %-10s %-10s %6.2f dB  %-32s (%d queries)" % (phase, t["group"], t["image"],
                                                                   result[1], result[2], queries))
        elif old is None:
            print("  %s %-10s %-10s not broken above 35 dB (%d queries)" % (phase, t["group"], t["image"], queries))

    # ------------------------------------------------------------------
    # STEP 2. PHASE A: breadth (strongest allowed probe per chain, no bisection)
    # ------------------------------------------------------------------
    print("\nSTEP 2  PHASE A: one quick success per image")
    for t in targets:
        result, q = attack(t, bisect_steps=0)
        keep(t, result, "A", q)

    # ------------------------------------------------------------------
    # STEP 3. PHASE B: depth (bisection), images with the lowest WPSNR first
    # ------------------------------------------------------------------
    if a.phase == "AB":
        print("\nSTEP 3  PHASE B: lightest attack that still removes the mark")
        # phase A already tried the strongest allowed attack of every chain: images it could
        # not break will not break in phase B either, so only broken images are refined
        broken = [t for t in targets if (t["group"], t["image"]) in best]
        for t in sorted(broken, key=lambda t: best[(t["group"], t["image"])][1]):
            result, q = attack(t, bisect_steps=4)
            keep(t, result, "B", q)

    # ------------------------------------------------------------------
    # STEP 4. Re-check the saved files with their detection
    # ------------------------------------------------------------------
    print("\nSTEP 4  verifying the files to upload")
    verified = []
    for t in targets:
        key = (t["group"], t["image"])
        if key not in best:
            continue
        path = os.path.join(a.out, "%s_%s_%s.bmp" % (a.group, t["group"], t["image"]))
        with contextlib.redirect_stdout(io.StringIO()):
            present, w = detections[t["group"]](t["orig"], t["wm"], path)
        ok = present == 0 and w >= 35
        print("  %-32s detection %d, WPSNR %.2f -> %s" % (os.path.basename(path), present, w,
                                                        "OK" if ok else "NOT VALID, do not upload"))
        if ok:
            verified.append((t["image"], t["group"], w, best[key][2]))
        else:
            os.remove(path)
    if os.path.exists(tmp):
        os.remove(tmp)

    # ------------------------------------------------------------------
    # STEP 5. Log for the Excel sheet (Image | Group | WPSNR | Attack(s) with parameters)
    # ------------------------------------------------------------------
    log_path = os.path.join(a.out, "attack_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Image", "Group", "WPSNR", "Attack(s) with parameters"])
        for row in verified:
            writer.writerow([row[0], row[1], "%.2f" % row[2], row[3]])
    print("\nSTEP 5  log written to %s (open it with Excel)" % log_path)

    # ------------------------------------------------------------------
    # STEP 6. Summary
    # ------------------------------------------------------------------
    broken_groups = {g for _, g, _, _ in verified}
    share = 100.0 * len(broken_groups) / max(1, len(groups))
    activity = next((p for limit, p in ACTIVITY_POINTS if share > limit), 0)
    print("\nSTEP 6  summary")
    print("  groups attacked: %d / %d (%.0f%%) -> %d activity points" % (len(broken_groups), len(groups), share, activity))
    if verified:
        print("  images broken: %d, average WPSNR %.2f dB" % (len(verified), sum(r[2] for r in verified) / len(verified)))
    print("  upload the %s_<groupB>_<image>.bmp files in %s" % (a.group, a.out))


if __name__ == "__main__":
    main()
