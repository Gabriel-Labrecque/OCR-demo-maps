import os
import cv2
import numpy as np
import skimage.util
import easyocr
import preprocessing as preprocess

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def upscale_image(img_array, method=None, scale=2):
    if method is None:
        return skimage.util.img_as_ubyte(img_array), 1
    elif method == "lanczos":
        return skimage.util.img_as_ubyte(preprocess.upscale_lanczos(img_array, scale)), scale
    elif method == "ai":
        return skimage.util.img_as_ubyte(preprocess.upscale_ai(img_array, scale)), scale
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
    img = preprocess.bilateral_denoise(img)
    #img = preprocess.clahe_color_amplification(img, amplification=0.025)
    img = preprocess.prepare_for_ocr(img)
    print("preprocessing ending...")
    return img, actual_scale


def save_preprocessed(img_array, img_path):
    base, ext = os.path.splitext(img_path)
    output_path = os.path.join("result", f"{os.path.basename(base)}-preprocessed{ext}")
    os.makedirs("result", exist_ok=True)
    cv2.imwrite(output_path, skimage.util.img_as_ubyte(img_array))
    print(f"Saved preprocessed: {output_path}")


def boxes_overlap(box1, box2, threshold=0.3):
    """Check if two quad boxes overlap significantly using bounding rect IoU."""
    def bounding_rect(box):
        pts = np.array(box)
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        return x1, y1, x2, y2

    ax1, ay1, ax2, ay2 = bounding_rect(box1)
    bx1, by1, bx2, by2 = bounding_rect(box2)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return False

    intersection = (ix2 - ix1) * (iy2 - iy1)
    area1 = (ax2 - ax1) * (ay2 - ay1)
    area2 = (bx2 - bx1) * (by2 - by1)
    union = area1 + area2 - intersection

    return (intersection / union) > threshold


def deduplicate_boxes(all_boxes, threshold=0.5):
    """Remove overlapping boxes keeping the first occurrence."""
    kept = []
    for box in all_boxes:
        if not any(boxes_overlap(box, kept_box, threshold=threshold) for kept_box in kept):
            kept.append(box)
    return kept


def run_detection(reader, img_uint8):
    """
    Run EasyOCR detection with built-in rotation support.
    Returns deduplicated list of quad polygon boxes.
    """
    results = reader.detect(img_uint8)

    horizontal_boxes = results[0][0] if len(results[0]) > 0 else []
    free_boxes = results[0][1] if len(results[0]) > 1 else []

    quads = []

    for box in horizontal_boxes:
        x1, x2, y1, y2 = box[0], box[1], box[2], box[3]
        quads.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    for box in free_boxes:
        quads.append([[p[0], p[1]] for p in box])

    return deduplicate_boxes(quads)


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
    reader = easyocr.Reader(['en', 'fr'], gpu=False, recognizer=False)
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
    upscale_method="ai",
    upscale_scale=2
)