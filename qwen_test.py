import os
import gc
import re
import time
import torch
import numpy as np
import skimage.util
import cv2
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from preprocessing import read_image, upscale_lanczos, upscale_ai
from PIL import Image

os.environ["HF_HOME"] = "./models"

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


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


def parse_grounding_output(raw_text, img_width, img_height):
    """
    Parses Qwen grounding output into the same structure as Florence-2:
    {'quad_boxes': [[x1,y1,x2,y1,x2,y2,x1,y2], ...], 'labels': [...]}

    Qwen outputs coordinates normalized to [0, 1000], so we rescale to pixel space.
    Format: <|object_ref_start|>label<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>
    """
    pattern = r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'
    matches = re.findall(pattern, raw_text)

    quad_boxes = []
    labels = []

    for match in matches:
        label, x1, y1, x2, y2 = match
        # Qwen normalizes coords to [0, 1000], rescale to pixel space
        x1 = int(x1) / 1000 * img_width
        y1 = int(y1) / 1000 * img_height
        x2 = int(x2) / 1000 * img_width
        y2 = int(y2) / 1000 * img_height

        # Convert to quad_box format [x1,y1, x2,y1, x2,y2, x1,y2] matching Florence-2
        quad_boxes.append([x1, y1, x2, y1, x2, y2, x1, y2])
        labels.append(label.strip())

    return {'quad_boxes': quad_boxes, 'labels': labels}


def rescale_boxes(boxes, scale):
    """Scale quad_boxes back down to original image coordinates."""
    return [[coord / scale for coord in box] for box in boxes]


def draw_boxes(img_array, parsed_answer, img_path, upscale_scale=1):
    img = skimage.util.img_as_ubyte(img_array).copy()
    boxes = parsed_answer['quad_boxes']
    labels = parsed_answer['labels']

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
    image = Image.fromarray(img_array)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    "Detect all text labels visible on this map image, "
                    "including place names, river names, and region names. "
                    "Output each label with its bounding box."
                )}
            ]
        }
    ]

    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )

    trimmed_ids = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]

    raw_text = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=False,  # keep grounding tokens for parsing
        clean_up_tokenization_spaces=False
    )[0]

    del inputs, generated_ids
    gc.collect()

    return raw_text


def process_images(upscale_method=None, upscale_scale=2):
    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        trust_remote_code=True
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

        raw_text = run_ocr(model, processor, upscaled)

        # parse into Florence-2 style structure
        img_h, img_w = upscaled.shape[:2]
        parsed_answer = parse_grounding_output(raw_text, img_w, img_h)

        del upscaled
        gc.collect()

        print(f"--- {filename} ---")
        print(parsed_answer)

        draw_boxes(img_array, parsed_answer, img_path, upscale_scale=actual_scale)


# ── entry point ───────────────────────────────────────────────────────────────
start = time.time()
process_images(
    upscale_method=None,  # None, "lanczos", or "ai"
    upscale_scale=2
)
print(f"Total time: {(time.time() - start):.2f}s")