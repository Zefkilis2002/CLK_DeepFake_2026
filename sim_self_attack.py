import cv2
import numpy as np
from pathlib import Path

import common
import detection_groupname
from embedding import embedding


BASE_DIR = Path(__file__).resolve().parent
ORIGINALS_DIR = BASE_DIR / "challenge_sim" / "defense"
WATERMARKED_DIR = BASE_DIR / "out"
OTHER_TEAMS_DIR = BASE_DIR / "challenge_sim" / "attack" / "images"
OUTPUT_DIR = BASE_DIR / "self_attack_out"
IMAGES = ["harbour.bmp", "market.bmp", "tower.bmp"]
OTHER_TEAM = "falco"


def run_detection(
    case_name,
    original_path,
    watermarked_path,
    tested_path,
):
    verdict, detector_wpsnr = detection_groupname.detection(
        str(original_path),
        str(watermarked_path),
        str(tested_path),
    )
    print(
        f"Case: {case_name:<35} | Verdict: {verdict} | "
        f"WPSNR: {detector_wpsnr:.2f} dB | Tested: {tested_path.name}"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng()

    for image_index, image_name in enumerate(IMAGES):
        original_path = ORIGINALS_DIR / image_name
        watermarked_path = WATERMARKED_DIR / image_name

        if not original_path.is_file():
            raise FileNotFoundError(f"Original image not found: {original_path}")
        if not watermarked_path.is_file():
            raise FileNotFoundError(f"Watermarked image not found: {watermarked_path}")

        # Watermarked image after a random attack.
        watermarked = cv2.imread(str(watermarked_path), cv2.IMREAD_GRAYSCALE)
        if watermarked is None:
            raise RuntimeError(f"Could not read {watermarked_path}")
        attacked_watermarked, attack_description = common.random_attack(
            watermarked,
            rng,
        )
        attacked_watermarked_path = OUTPUT_DIR / f"{Path(image_name).stem}_watermarked_attack.bmp"
        if not cv2.imwrite(str(attacked_watermarked_path), attacked_watermarked):
            raise RuntimeError(f"Failed to save {attacked_watermarked_path}")
        run_detection(
            f"watermarked attacked ({attack_description})",
            original_path,
            watermarked_path,
            attacked_watermarked_path,
        )

        # Original image used as the tested attacked image.
        run_detection(
            "original as attacked",
            original_path,
            watermarked_path,
            original_path,
        )

        # Another team's watermarked version of the same image.
        other_team_path = OTHER_TEAMS_DIR / f"{OTHER_TEAM}_{image_name}"
        if not other_team_path.is_file():
            raise FileNotFoundError(f"Other team's image not found: {other_team_path}")
        run_detection(
            f"{OTHER_TEAM} watermarked image",
            original_path,
            watermarked_path,
            other_team_path,
        )

        # Same image marked with a random watermark key.
        random_key = rng.integers(0, 2, size=1024, dtype=np.uint8)
        random_key_path = OUTPUT_DIR / f"{Path(image_name).stem}_random_key.npy"
        np.save(random_key_path, random_key)
        random_key_image = embedding(str(original_path), str(random_key_path))
        random_key_image_path = OUTPUT_DIR / f"{Path(image_name).stem}_random_key.bmp"
        if not cv2.imwrite(str(random_key_image_path), random_key_image):
            raise RuntimeError(f"Failed to save {random_key_image_path}")
        run_detection(
            "same image, random watermark key",
            original_path,
            watermarked_path,
            random_key_image_path,
        )

        # Unmarked image after a random attack.
        original = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise RuntimeError(f"Could not read {original_path}")
        attacked_original, original_attack_description = common.random_attack(
            original,
            rng,
        )
        attacked_original_path = OUTPUT_DIR / f"{Path(image_name).stem}_original_attack.bmp"
        if not cv2.imwrite(str(attacked_original_path), attacked_original):
            raise RuntimeError(f"Failed to save {attacked_original_path}")
        run_detection(
            f"unmarked attacked ({original_attack_description})",
            original_path,
            watermarked_path,
            attacked_original_path,
        )

        # An unrelated original image, with different content.
        unrelated_name = IMAGES[(image_index + 1) % len(IMAGES)]
        unrelated_path = ORIGINALS_DIR / unrelated_name
        run_detection(
            f"unrelated image ({unrelated_name})",
            original_path,
            watermarked_path,
            unrelated_path,
        )


if __name__ == "__main__":
    main()
