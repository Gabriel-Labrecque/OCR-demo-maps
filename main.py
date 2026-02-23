import os
import shutil
import logging
import cv2
import numpy as np
from pathlib import Path
import easyocr

# Local imports
from text_extraction import TextExtraction

# Silence EasyOCR internal logging and progress messages
logging.getLogger('easyocr').setLevel(logging.ERROR)


def main():
    # 1. Path Configuration
    base_dir = Path(__file__).parent
    input_folder = base_dir / "input"
    output_folder = base_dir / "result"

    # 2. Results Directory Cleanup
    if output_folder.exists():
        print(f"Cleaning up old results in {output_folder}...")
        try:
            shutil.rmtree(output_folder)
        except PermissionError:
            print("Warning: Could not delete result folder. Ensure no images are open.")

    output_folder.mkdir(parents=True, exist_ok=True)

    # 3. Validation
    if not input_folder.exists():
        print(f"Error: Input folder '{input_folder}' not found.")
        return

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    image_files = [f for f in input_folder.iterdir() if f.suffix.lower() in valid_extensions]

    if not image_files:
        print("No valid images found in the input folder.")
        return

    # 4. Persistent Service Initialization
    print(f"Initializing EasyOCR for {len(image_files)} images...")
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    # 5. Batch Processing Loop
    for img_path in image_files:
        try:
            print(f"Processing: {img_path.name}")

            # Instantiate TextExtraction with the persistent reader
            extractor = TextExtraction(img_path=str(img_path), reader=reader)

            # OCR Inference
            text_info = extractor.read_text_from_image(scale_xy=(1.0, 1.0))

            # Annotation (Handles float64 -> uint8 BGR internally)
            final_image = extractor.draw_bounding_box(text_info)

            # 6. Save result using requested naming convention
            output_filename = f"{img_path.stem}-bbox{img_path.suffix}"
            save_path = output_folder / output_filename

            cv2.imwrite(str(save_path), final_image)
            print(f"  Successfully saved to: {output_filename}")

        except Exception as e:
            # THIS BLOCK WAS MISSING - It handles errors so the loop continues
            print(f"  [ERROR] Failed to process {img_path.name}: {e}")

    print("\nBatch processing complete.")


if __name__ == "__main__":
    main()