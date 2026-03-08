import os
import cv2
import numpy as np
import skimage.util
import easyocr
import preprocessing as preprocess

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def upscale_image(img_array, method=None, scale=2):
    if method is None:
        return skimage.util.img_as_float(img_array), 1
    elif method == "lanczos":
        return preprocess.upscale_lanczos(img_array, scale), scale
    elif method == "ai":
        return preprocess.upscale_ai(img_array, scale), scale
    else:
        raise ValueError(f"Unknown upscale method: {method}. Use None, 'lanczos', or 'ai'.")


def preprocess_for_detection(img, upscale_method, upscale_scale):
    """
    Apply contrast enhancement pipeline before detection.
    Input: float64 RGB [0.0, 1.0]
    Output: (uint8 RGB upscaled, actual_scale)
    """
    print("preprocessing starting...")
    img, actual_scale = upscale_image(
        img,
        method=upscale_method,
        scale=upscale_scale
    )
    img = preprocess.denoise_meanshift(img, spatial_radius=15, color_radius=0.075)
    img = preprocess.denoise_ai(img, 1.0)
    img = preprocess.clahe_color_amplification(img, amplification=0.1)
    img = preprocess.color_equalization(img, clip_limit=0.01)
    img = preprocess.prepare_for_ocr(img)
    print("preprocessing ending...")
    return img, actual_scale


def save_preprocessed(img_array, img_path):
    base, ext = os.path.splitext(img_path)
    output_path = os.path.join("result", f"{os.path.basename(base)}-preprocessed{ext}")
    os.makedirs("result", exist_ok=True)
    cv2.imwrite(output_path, skimage.util.img_as_ubyte(img_array))
    print(f"Saved preprocessed: {output_path}")


def run_detection(reader, img_uint8):
    """
    Run EasyOCR detection with built-in rotation support.
    Returns deduplicated list of quad polygon boxes.
    """
    results = reader.readtext(img_uint8, rotation_info=[90, 180, 270])

    quads = []
    for bbox, text, confidence in results:
        quads.append([[int(p[0]), int(p[1])] for p in bbox])

    return quads


def rescale_boxes(boxes, scale):
    return [[[coord / scale for coord in point] for point in box] for box in boxes]


def draw_boxes(img_array, boxes, img_path, upscale_scale=1):
    img = skimage.util.img_as_ubyte(img_array).copy()

    if upscale_scale != 1:
        boxes = rescale_boxes(boxes, upscale_scale)

    for box in boxes:
        pts = np.array(box, dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

    base, ext = os.path.splitext(img_path)
    output_path = os.path.join("result", f"{os.path.basename(base)}-bbx{ext}")
    os.makedirs("result", exist_ok=True)

    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Saved: {output_path}")


def process_images(upscale_method=None, upscale_scale=2):
    print("Loading EasyOCR detector...")
    reader = easyocr.Reader(['en', 'fr'], gpu=False)
    print("Model ready.")

    for filename in os.listdir("input/"):
        if os.path.splitext(filename)[1].lower() not in SUPPORTED_EXTENSIONS:
            continue

        img_path = os.path.join("input/", filename)
        print(f"Processing {filename}...")

        img_float = preprocess.read_image(img_path)
        img_preprocessed, actual_scale = preprocess_for_detection(img_float, upscale_method, upscale_scale)

        save_preprocessed(img_preprocessed, img_path)

        boxes = run_detection(reader, img_preprocessed)

        print(f"--- {filename} ---")
        print(f"Found {len(boxes)} text regions")

        draw_boxes(img_float, boxes, img_path, upscale_scale=actual_scale)


# ── entry point ───────────────────────────────────────────────────────────────
process_images(
    upscale_method="lanczos", # "lanczos", "ai" or None
    upscale_scale=2
)