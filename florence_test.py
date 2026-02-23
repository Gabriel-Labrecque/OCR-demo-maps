import os
import gc
import cv2
import torch
import numpy as np
import skimage.util
from transformers import AutoProcessor, AutoModelForCausalLM
from preprocessing import read_image, upscale_lanczos, upscale_ai

os.environ["HF_HOME"] = "./models"

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MODEL_ID = "microsoft/Florence-2-large"


def upscale_image(img_array, method=None, scale=2):
    """
    method: None = no upscaling, 'lanczos' = Lanczos, 'ai' = EDSR
    returns (upscaled_img as uint8, actual_scale_used)
    """
    if method is None:
        return skimage.util.img_as_ubyte(img_array), 1
    elif method == "lanczos":
        upscaled = upscale_lanczos(img_array, scale)
    elif method == "ai":
        upscaled = upscale_ai(img_array, scale)
    else:
        raise ValueError(f"Unknown upscale method: {method}. Use None, 'lanczos', or 'ai'.")

    return skimage.util.img_as_ubyte(upscaled), scale


def rescale_boxes(boxes, scale):
    """Scale quad_boxes back down to original image coordinates."""
    return [[coord / scale for coord in box] for box in boxes]


def draw_boxes(img_array, parsed_answer, img_path, upscale_scale=1):
    img = skimage.util.img_as_ubyte(img_array).copy()
    result = parsed_answer['<OCR_WITH_REGION>']
    boxes = result['quad_boxes']
    labels = result['labels']

    if upscale_scale != 1:
        boxes = rescale_boxes(boxes, upscale_scale)

    for box, label in zip(boxes, labels):
        pts = np.array([[box[i], box[i + 1]] for i in range(0, 8, 2)], dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.putText(img, label, (int(box[0]), int(box[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    base, ext = os.path.splitext(img_path)
    output_path = os.path.join("result", f"{os.path.basename(base)}-bbx{ext}")
    os.makedirs("result", exist_ok=True)

    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Saved: {output_path}")


def run_ocr(model, processor, img_array):
    inputs = processor(text="<OCR_WITH_REGION>", images=img_array, return_tensors="pt")
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
        image_size=(img_array.shape[1], img_array.shape[0])
    )
    del inputs, generated_ids
    gc.collect()
    return parsed_answer


def process_images(upscale_method=None, upscale_scale=2):
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, trust_remote_code=True
    ).to("cpu")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("Model ready.")

    for filename in os.listdir("input/"):
        if os.path.splitext(filename)[1].lower() not in SUPPORTED_EXTENSIONS:
            continue

        img_path = os.path.join("input/", filename)
        print(f"Processing {filename}...")

        img_array = read_image(img_path)  # float64 [0.0, 1.0]
        upscaled, actual_scale = upscale_image(img_array, method=upscale_method, scale=upscale_scale)

        parsed_answer = run_ocr(model, processor, upscaled)
        del upscaled
        gc.collect()

        print(f"--- {filename} ---")
        print(parsed_answer)

        draw_boxes(img_array, parsed_answer, img_path, upscale_scale=actual_scale)


# ── entry point ───────────────────────────────────────────────────────────────
process_images(
    upscale_method="lanczos",  # None, "lanczos", or "ai"
    upscale_scale=2
)