import os
import json

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# Minimal, from-scratch baseline.
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MAX_NEW_TOKENS = 1024
MAX_IMAGE_PIXELS = 2560 * 2560


def get_runtime_config():
    """Central config for model/runtime values used across the pipeline."""
    return {
        "model_id": MODEL_ID,
        "torch_dtype": torch.float32,
        "device": "cpu",
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_image_pixels": MAX_IMAGE_PIXELS,
    }


def build_cleaning_prompt(detections, context = ""):
    """
    Prompt used to clean and post-process Florence OCR detections.
    The model receives the original image for visual grounding, the context string from
    the first pass, and the raw detection list serialized as JSON.

    Cleaning rules applied by the model:
      1. Remove single-letter words (likely artifacts).
      2. Fix OCR artifacts in words (e.g. noise characters, broken glyphs).
      3. Merge horizontally adjacent boxes that form a logical continuation
         (e.g. split words across tiles): keep the leftmost bbox, extend it to cover both.
      4. Merge vertically adjacent boxes that continue the same label:
         join their text with \\n and extend the bbox to cover both.
      5. Use given geographic/historical context to correct plausible misreads
         (e.g. "Rivière Chauot" → "Rivière Chaudière" on a 1755 Quebec map).
    """
    detections_json = json.dumps(detections, ensure_ascii=False)
    context_line = f"Map context: {context}\n" if context else ""
    return (
        f"{context_line}"
        "You are a post-processing assistant for OCR results on historical and geographical maps.\n"
        "Below is a list of raw OCR detections from Florence-2, each with a text label and a "
        "bounding box [x1, y1, x2, y2] in image pixel coordinates.\n"
        "Clean the detection list by applying ALL of the following rules:\n"
        "1. Fix OCR artifacts within words (stray characters, broken accents, garbled glyphs).\n"
        "2. Horizontally adjacent detections that form a logical word or phrase continuation "
        "(including words split by a tile boundary, e.g. 'Océan Atl' + 'Atlantique' being 'Océan Atlantique') "
        "must be merged into one detection. Use the merged text and a bbox that spans both originals.\n"
        "3. Vertically adjacent detections that continue the same multi-line label must be merged: "
        "Join their texts with \\n and extend the bbox to cover both.\n"
        "4. Use the map context and geographic/historical knowledge to correct plausible misreads "
        "(e.g. 'Rivière Chauot' is probably 'Rivière Chaudière' on a Quebec 1755 map, if you correctly geolocalize it). "
        "Correct when you are sufficiently confident; otherwise keep the original text.\n"
        "Return ONLY valid JSON with this schema, no extra text:\n"
        '{"detections":[{"text":str,"bbox_xyxy":[int,int,int,int]}]}\n'
        f"Raw detections:\n{detections_json}"
    )



def build_ocr_messages(image, prompt, processor):
    """Generic prompt input builder: builds messages, applies chat template, and returns processor inputs."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    )


def load_model_and_processor(config):
    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config["model_id"],
        torch_dtype=config["torch_dtype"],
        trust_remote_code=True,
    ).to(config["device"])

    processor = AutoProcessor.from_pretrained(
        config["model_id"],
        trust_remote_code=True,
        max_pixels=config["max_image_pixels"],
    )
    print("Model ready.")
    return model, processor


def run_inference(
    model, processor, image, prompt,
    config, skip_special_tokens = False
):
    """Runs model inference on a pre-loaded PIL image with the given prompt."""

    inputs = build_ocr_messages(image, prompt, processor)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config["max_new_tokens"],
            do_sample=False,
        )
    trimmed_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    return processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )[0]


def get_image_for_qwen(input_path: str, image_size: dict) -> Image.Image:
    
    image = Image.open(input_path).convert("RGB")
    w, h = image.size
    if w * h > MAX_IMAGE_PIXELS:
        scale = (MAX_IMAGE_PIXELS / (w * h)) ** 0.5
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    return image

    
def parse_json_or_empty(raw_text):
    raw = raw_text.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return []
        else:
            return []
    # If obj is a dict with 'detections', return that, else if it's a list, return it, else []
    if isinstance(obj, dict) and "detections" in obj and isinstance(obj["detections"], list):
        return obj["detections"]
    elif isinstance(obj, list):
        return obj
    else:
        return []


def save_result(image_path, output_path, result):
    """
    Save the cleaned Qwen results to output_path (JSON) and output a -bbx image with bboxes drawn, matching Florence's output.py.
    result: dict with keys 'image_size', 'detections', and optionally 'context'.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved: {output_path}")

    # Draw bboxes on the raw input image
    img = cv2.imread(image_path)
    ext = os.path.splitext(image_path)[1]
    for det in result.get("detections", []):
        bbox = det.get("bbox_xyxy", [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            label_x, label_y = x1, max(y1 - 4, 0)
        else:
            continue
        cv2.putText(img, det["text"], (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    bbx_path = os.path.splitext(output_path)[0] + f"-bbx{ext}"
    cv2.imwrite(bbx_path, img)
    print(f"Saved: {bbx_path}")


