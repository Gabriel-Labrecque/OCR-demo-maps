import os
import gc
import cv2
import torch
import numpy as np
import skimage.util
from pathlib import Path
from transformers import AutoProcessor, AutoModelForCausalLM
import preprocessing as preprocess

BASE_DIR = Path(__file__).parent
os.environ["HF_HOME"] = str(BASE_DIR / "models")

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MODEL_ID = "microsoft/Florence-2-large"


def upscale_image(img_array, method=None, scale=2):
    """
    method: None = no upscaling, 'lanczos' = Lanczos, 'ai' = EDSR
    returns (upscaled_img as float64 [0.0, 1.0], actual_scale_used)
    """
    if method is None:
        return skimage.util.img_as_float(img_array), 1
    elif method == "lanczos":
        return preprocess.upscale_lanczos(img_array, scale), scale
    elif method == "ai":
        return preprocess.upscale_ai(img_array, scale), scale
    else:
        raise ValueError(f"Unknown upscale method: {method}. Use None, 'lanczos', or 'ai'.")


def preprocess_for_ocr(img, upscale_method, upscale_scale):
    """
    Apply contrast enhancement pipeline before OCR.
    Input:  float64 RGB [0.0, 1.0]
    Output: (uint8 RGB, actual_scale)
    """
    print("Preprocessing starting...")

    img, actual_scale = upscale_image(img, method=upscale_method, scale=upscale_scale)
    img = preprocess.bilateral_denoise(img, sigma_color=0.025, sigma_spatial=10)
    #img = preprocess.clahe_color_amplification(img, amplification=0.5)
    #img = preprocess.color_quantization(img, n_colors=12)

    # Convert to uint8 RGB (Florence expects RGB)
    if img.dtype != np.uint8:
        img = skimage.util.img_as_ubyte(img)

    print("Preprocessing done.")
    return img, actual_scale


def rescale_boxes(boxes, scale):
    """Scale quad_boxes back down to original image coordinates."""
    return [[coord / scale for coord in box] for box in boxes]


def draw_boxes(img_array, parsed_answer, img_path, upscale_scale=1):
    img = skimage.util.img_as_ubyte(img_array).copy()
    result = parsed_answer.get('<OCR_WITH_REGION>', {})
    boxes = result.get('quad_boxes', [])
    labels = result.get('labels', [])

    if not boxes:
        print("No boxes to draw.")
        return

    if upscale_scale != 1:
        boxes = rescale_boxes(boxes, upscale_scale)

    for box, label in zip(boxes, labels):
        pts = np.array([[int(box[i]), int(box[i + 1])] for i in range(0, 8, 2)], dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.putText(img, label, (int(box[0]), int(box[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    base, ext = os.path.splitext(img_path)
    output_path = str(BASE_DIR / "result" / f"{os.path.basename(base)}-bbx{ext}")
    os.makedirs(BASE_DIR / "result", exist_ok=True)

    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Saved: {output_path}")


def run_ocr(model, processor, img_uint8_rgb):
    """
    Run Florence-2 OCR with region detection.
    Input: uint8 RGB image
    """
    inputs = processor(text="<OCR_WITH_REGION>", images=img_uint8_rgb, return_tensors="pt")

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=4096,
        do_sample=False,
        num_beams=3,
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(
        generated_text,
        task="<OCR_WITH_REGION>",
        image_size=(img_uint8_rgb.shape[1], img_uint8_rgb.shape[0])
    )

    del inputs, generated_ids
    gc.collect()

    return parsed_answer


def process_images(upscale_method=None, upscale_scale=2):
    print("Loading Florence-2 model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to("cpu")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("Model ready.")

    for filename in os.listdir(BASE_DIR / "input"):
        if os.path.splitext(filename)[1].lower() not in SUPPORTED_EXTENSIONS:
            continue

        img_path = str(BASE_DIR / "input" / filename)
        print(f"\nProcessing {filename}...")

        img_float = preprocess.read_image(img_path)  # float64 RGB [0.0, 1.0]

        img_preprocessed, actual_scale = preprocess_for_ocr(img_float, upscale_method, upscale_scale)

        base, ext = os.path.splitext(filename)
        preprocessed_path = str(BASE_DIR / "result" / f"{base}-preprocessed{ext}")
        cv2.imwrite(preprocessed_path, cv2.cvtColor(img_preprocessed, cv2.COLOR_RGB2BGR))
        print(f"Saved preprocessed: {preprocessed_path}")

        parsed_answer = run_ocr(model, processor, img_preprocessed)
        del img_preprocessed
        gc.collect()

        print(f"--- {filename} ---")
        result = parsed_answer.get('<OCR_WITH_REGION>', {})
        print(f"Found {len(result.get('labels', []))} text regions")
        print(parsed_answer)

        draw_boxes(img_float, parsed_answer, img_path, upscale_scale=actual_scale)


# entry point
process_images(
    upscale_method="lanczos",  # None, "lanczos", or "ai"
    upscale_scale=2
)