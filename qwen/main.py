import os
import sys
import json
import time
import argparse
from typing import Dict, List

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# Minimal, from-scratch baseline.
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
INPUT_DIR = os.environ.get("INPUT_DIR", "/data/input")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/data/result")
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MAX_NEW_TOKENS = 1024
MAX_IMAGE_PIXELS = 2560 * 2560


def get_runtime_config() -> Dict:
    """Central config for model/runtime values used across the pipeline."""
    return {
        "model_id": MODEL_ID,
        "torch_dtype": torch.float32,
        "device": "cpu",
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_image_pixels": MAX_IMAGE_PIXELS,
    }


def list_input_images(input_dir: str) -> List[str]:
    if not os.path.isdir(input_dir):
        return []
    files = []
    for name in sorted(os.listdir(input_dir)):
        if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS:
            files.append(os.path.join(input_dir, name))
    return files


def build_context_prompt() -> str:
    """Prompt used on the full image to extract geographic/thematic context before OCR."""
    return (
        "You are analyzing a historical or geographical map. "
        "Describe its context in one or two short sentences: identify the region, time period, "
        "and main theme if visible (e.g. political borders, climate, population, military). "
        "Do not describe visual style, only factual content. "
        "Output plain text only, no formatting."
    )


def build_cleaning_prompt(detections: List[Dict], context: str = "") -> str:
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
      5. Use geographic/historical context to correct plausible misreads
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



def build_ocr_messages(image: Image.Image, prompt: str, processor) -> Dict:
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


def load_model_and_processor(config: Dict):
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
    model, processor, image: Image.Image, prompt: str,
    config: Dict, skip_special_tokens: bool = False
) -> str:
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




def parse_json_or_empty(raw_text: str) -> Dict:
    raw = raw_text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {"detections": []}


def save_result(image_path: str, image_size: Dict, detections: List[Dict]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base}-qwen-cleaned.json")
    payload = {"image_size": image_size, "detections": detections}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen OCR cleaner for Florence-2 output.")
    parser.add_argument("--json", required=True, help="Path to the Florence-2 OCR JSON file.")
    parser.add_argument("--image", required=True, help="Path to the original map image.")
    args = parser.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        florence_data = json.load(f)

    image_size = florence_data.get("image_size", {})
    detections = florence_data.get("detections", [])
    image = Image.open(args.image).convert("RGB")

    # Downscale image for context pass if too large
    w, h = image.size
    if w * h > MAX_IMAGE_PIXELS:
        scale = (MAX_IMAGE_PIXELS / (w * h)) ** 0.5
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    start = time.time()
    config = get_runtime_config()
    model, processor = load_model_and_processor(config)

    print("Running context pass...")
    context = run_inference(
        model, processor, image, build_context_prompt(), config, skip_special_tokens=True
    ).strip()
    print(f"  Context: {context}")

    print(f"Running cleaning pass on {len(detections)} detections...")
    prompt = build_cleaning_prompt(detections, context)
    raw = run_inference(model, processor, image, prompt, config)
    cleaned = parse_json_or_empty(raw)

    save_result(args.image, image_size, cleaned.get("detections", []))
    print(f"Total time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
