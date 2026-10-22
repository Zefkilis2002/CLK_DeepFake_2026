"""
CHALLENGE DAY, ATTACK PHASE (10:15 - 12:30), advanced version.

Competition rules for this phase:
  - download the other groups' images (<groupB>_<image>.bmp) and their
    detection files (detection_<groupB>.py or .pyc);
  - attack with the 6 allowed attacks only, using ONLY the watermarked image;
  - success = their detection returns 0 AND WPSNR >= 35 dB;
  - upload the best success per image as <ourgroup>_<groupB>_<image>.bmp;
  - points: % of groups attacked (>90% = 6), images above the average WPSNR,
    +2 for a group nobody else attacked; Excel log required afterwards.

Same steps as day_attack.py on main, built on the package:
  ctm/attack/oracle.py  FileOracle: wraps their detection and counts the queries.
                        It is the ONLY object that knows the original image path.
  ctm/attack/search.py  search_attack: 13 attack chains, strength s in [0, 1];
                        local WPSNR (free) avoids useless queries.
  ctm/attack/log.py     the Excel log (with chain and strength, to reproduce attacks).
  ctm/infra/results.py  every result is also appended to results/attack_day.csv.

Steps:
  STEP 0  inventory: image -> group, detection file, original
  STEP 1  smoke test of every detection (loads, finds its mark, 0 on the original, speed)
  STEP 2  PHASE A breadth: strongest allowed probe of each chain on EVERY image
          (no bisection) -> one success per group as soon as possible
  STEP 3  PHASE B depth: bisection on the broken images, worst WPSNR first
  STEP 4  re-checks the SAVED files with their detection
  STEP 5  Excel log
  STEP 6  summary (groups attacked, activity points)

Usage (from the repository root, ctm/):
  ..\.venv\Scripts\python.exe -m scripts.attack_day --group NAME --targets T --detectors D --originals O
  Options: --only g1,g2 (split the groups between laptops)   --phase A (quick pass only)
"""
import argparse
import contextlib
import io
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import GROUP, REPO_DIR
from ctm.attack.day import find_detection
from ctm.attack.log import log_attack
from ctm.attack.oracle import FileOracle, load_detection
from ctm.attack.search import search_attack
from ctm.common.imgio import read_gray, write_bmp
from ctm.infra.results import append_result

ACTIVITY_POINTS = [(90, 6), (60, 4), (30, 2)]


def main():
    ap = argparse.ArgumentParser(description="Attack phase of the challenge day (advanced)")
    ap.add_argument("--group", "--our", dest="group", default=GROUP, help="our group nickname")
    ap.add_argument("--targets", required=True, help="folder with <groupB>_<image>.bmp")
    ap.add_argument("--detectors", "--detections", dest="detectors", required=True)
    ap.add_argument("--originals", required=True, help="folder with the 3 original images")
    ap.add_argument("--only", default="", help="comma-separated groups to attack")
    ap.add_argument("--phase", default="AB", choices=["A", "AB"])
    ap.add_argument("--out", default=os.path.join(REPO_DIR, "challenge_day", "attack_upload"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.abspath(os.path.join(a.out, "_tmp_attacked.bmp"))
    only = set(filter(None, a.only.split(",")))

    # ------------------------------------------------------------------
    # STEP 0. Inventory
    # ------------------------------------------------------------------
    print("STEP 0  inventory")
    targets = []
    for fname in sorted(os.listdir(a.targets)):
        if not fname.lower().endswith(".bmp") or "_" not in fname:
            continue
        group, image = fname[:-4].split("_", 1)
        if group == a.group or (only and group not in only):
            continue
        det = find_detection(a.detectors, group)
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
    detections = {}
    for group in groups:
        t = next(x for x in targets if x["group"] == group)
        try:
            det = load_detection(t["det"])
            with contextlib.redirect_stdout(io.StringIO()):
                own = det(t["orig"], t["wm"], t["wm"])[0]          # their image: expected 1
                start = time.time()
                on_orig = det(t["orig"], t["wm"], t["orig"])[0]    # the original: expected 0
                seconds = time.time() - start
            detections[group] = det
            note = "" if (own, on_orig) == (1, 0) else "  <-- unexpected answers, attack anyway"
            print("  %-10s %-24s %.2f s per call | own image %d, original %d%s"
                  % (group, os.path.basename(t["det"]), seconds, own, on_orig, note))
        except Exception as e:
            print("  %-10s CANNOT RUN (%s: %s)" % (group, type(e).__name__, e))
    targets = [t for t in targets if t["group"] in detections]

    best = {}            # (group, image) -> result dict of search_attack

    def run(t, phase, bisect_steps):
        """One search; the original path lives only inside the oracle."""
        oracle = FileOracle(detections[t["group"]], t["orig"], t["wm"], tmp)
        result = search_attack(read_gray(t["wm"]), oracle, bisect_steps=bisect_steps)
        key, old = (t["group"], t["image"]), best.get((t["group"], t["image"]))
        if result is not None and (old is None or result["wpsnr"] > old["wpsnr"]):
            best[key] = result
            write_bmp(os.path.join(a.out, "%s_%s_%s.bmp" % (a.group, t["group"], t["image"])), result["image"])
            print("  %s %-10s %-10s %6.2f dB  %-32s (%d queries)" % (phase, t["group"], t["image"],
                                                                   result["wpsnr"], result["text"], oracle.queries))
        elif old is None:
            print("  %s %-10s %-10s not broken above 35 dB (%d queries)" % (phase, t["group"], t["image"], oracle.queries))

    # ------------------------------------------------------------------
    # STEP 2. PHASE A: breadth
    # ------------------------------------------------------------------
    print("\nSTEP 2  PHASE A: one quick success per image")
    for t in targets:
        run(t, "A", bisect_steps=0)

    # ------------------------------------------------------------------
    # STEP 3. PHASE B: depth, only on broken images, worst first
    # ------------------------------------------------------------------
    if a.phase == "AB":
        print("\nSTEP 3  PHASE B: lightest attack that still removes the mark")
        broken = [t for t in targets if (t["group"], t["image"]) in best]
        for t in sorted(broken, key=lambda t: best[(t["group"], t["image"])]["wpsnr"]):
            run(t, "B", bisect_steps=4)

    # ------------------------------------------------------------------
    # STEP 4. Re-check the saved files
    # ------------------------------------------------------------------
    print("\nSTEP 4  verifying the files to upload")
    log_path = os.path.join(a.out, "attack_log.csv")
    if os.path.exists(log_path):
        os.remove(log_path)
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
        if not ok:
            os.remove(path)
            continue
        r = best[key]
        verified.append((t["image"], t["group"], w))
        # STEP 5 (per row). Excel log + results table
        log_attack(log_path, t["image"], t["group"], w, r["text"], r["chain"], r["s"])
        append_result("attack_day", {"group": t["group"], "image": t["image"], "wpsnr": w,
                                     "attack": r["text"], "chain": r["chain"], "strength": r["s"]})
    if os.path.exists(tmp):
        os.remove(tmp)
    print("\nSTEP 5  log written to %s (open it with Excel)" % log_path)

    # ------------------------------------------------------------------
    # STEP 6. Summary
    # ------------------------------------------------------------------
    broken_groups = {g for _, g, _ in verified}
    share = 100.0 * len(broken_groups) / max(1, len(groups))
    activity = next((p for limit, p in ACTIVITY_POINTS if share > limit), 0)
    print("\nSTEP 6  summary")
    print("  groups attacked: %d / %d (%.0f%%) -> %d activity points" % (len(broken_groups), len(groups), share, activity))
    if verified:
        print("  images broken: %d, average WPSNR %.2f dB" % (len(verified), sum(v[2] for v in verified) / len(verified)))
    print("  upload the %s_<groupB>_<image>.bmp files in %s" % (a.group, a.out))


if __name__ == "__main__":
    main()
