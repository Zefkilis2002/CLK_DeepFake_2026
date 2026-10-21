import cv2
from pathlib import Path

from embedding import embedding
from common import wpsnr


BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "challenge_sim" / "defense"
OUTPUT_DIR = BASE_DIR / "out"
WATERMARK_PATH = IMAGES_DIR / "clk.npy"
IMAGES = ["harbour.bmp", "market.bmp", "tower.bmp"]


def main():
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for image_name in IMAGES:
        image_path = IMAGES_DIR / image_name
        output_path = OUTPUT_DIR / image_path.name

        result = embedding(str(image_path), str(WATERMARK_PATH))

        if not cv2.imwrite(str(output_path), result):
            raise RuntimeError(f"Failed to save the image to {output_path}")

        print(f"Saved: {output_path}")

        original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        watermarked = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
        
        if original is None or watermarked is None:
            raise RuntimeError(f"Could not read {image_path} or {output_path}")

        print(f"WPSNR of {image_name}: {wpsnr(original, watermarked):.2f} dB")


if __name__ == "__main__":
    main()
