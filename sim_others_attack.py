import cv2
import numpy as np
from pathlib import Path

import common
import attacks


BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "challenge_sim" / "attack" / "images"
DETECTIONS_DIR = BASE_DIR / "challenge_sim/attack/detections"
OUTPUT_DIR = BASE_DIR / "attacked_out"
TEAMS = ["falco", "koala", "lynx"]
IMAGES = ["harbour.bmp", "market.bmp", "tower.bmp"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng()
    for team_name in TEAMS:
        detection_path = DETECTIONS_DIR / f"detection_{team_name}.pyc"
        detection = attacks.load_detection(str(detection_path))

        for image_name in IMAGES:
            image_path = IMAGES_DIR / f"{team_name}_{image_name}"
            output_path = OUTPUT_DIR / f"{team_name}_{Path(image_name).stem}_attacked.bmp"
            original_path = BASE_DIR / "challenge_sim" / "defense" / image_name

            if not image_path.is_file():
                raise FileNotFoundError(f"Watermarked image not found: {image_path}")
            if not original_path.is_file():
                raise FileNotFoundError(f"Original image not found: {original_path}")

            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"Could not read {image_path}")

            attacked_image, description = common.random_attack(image, rng)

            if not cv2.imwrite(str(output_path), attacked_image):
                raise RuntimeError(f"Failed to save the image to {output_path}")

            verdict, detector_wpsnr = detection(
                str(original_path),
                str(image_path),
                str(output_path),
            )

            print(
                f"Team: {team_name:<5} | Image: {image_name:<12} | "
                f"Attack: {description:<24} | Verdict: {verdict} | "
                f"WPSNR: {detector_wpsnr:.2f} dB | Output: {output_path.name}"
            )

if __name__ == "__main__":
    main()
