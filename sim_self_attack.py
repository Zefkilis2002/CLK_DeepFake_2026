import cv2
import numpy as np
from pathlib import Path

import detection_groupname
import common


BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "out"
OUTPUT_DIR = BASE_DIR / "attacked_out"
WATERMARK_PATH = IMAGES_DIR / "clk.npy"
IMAGES = ["harbour.bmp", "market.bmp", "tower.bmp"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for image_name in IMAGES:
        image_path = IMAGES_DIR / image_name
        output_path = OUTPUT_DIR / image_path.name
        
        # attacked_image = attacks.attacks(str(image_path), "jpeg", [8])
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Could not read {image_path}")

        attacked_image, desc = common.random_attack(
            image,
            np.random.default_rng(),
        )
        
        if not cv2.imwrite(str(output_path), attacked_image):
            raise RuntimeError(f"Failed to save the image to {output_path}")
        
        original_path = BASE_DIR / "challenge_sim" / "defense" / image_name
        verdict = detection_groupname.detection(
            str(original_path),
            str(image_path),
            str(output_path),
        )
        
        print(f"Verdict: {verdict} desc: {desc}")
        
    for image_name in IMAGES:
        image_path = IMAGES_DIR / image_name
        output_path = OUTPUT_DIR / image_path.name
        
    #     # attacked_image = attacks.attacks(str(image_path), "jpeg", [8])
    #     image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    #     if image is None:
    #         raise RuntimeError(f"Could not read {image_path}")

    #     attacked_image, desc = common.random_attack(
    #         image,
    #         np.random.default_rng(),
    #     )
        
    #     if not cv2.imwrite(str(output_path), attacked_image):
    #         raise RuntimeError(f"Failed to save the image to {output_path}")
        
        original_path = BASE_DIR / "challenge_sim" / "defense" / image_name
        verdict = detection_groupname.detection(
            str(original_path),
            str(image_path),
            str(output_path),
        )
        
        print(f"Verdict: {verdict} desc: {desc}")
        

if __name__ == "__main__":
    main()
