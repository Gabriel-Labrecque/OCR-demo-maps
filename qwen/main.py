import os
import json
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


MODEL_ID = "Qwen/Qwen3.5-4B"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MAX_NEW_TOKENS = 512
MAX_IMAGE_PIXELS = 2560 * 2560
MAX_PROMPT_DETECTIONS = 160
ASSISTANT_JSON_PREFILL = '{"detections":['


def get_runtime_config():
    """Central config for model/runtime values used across the pipeline."""
    return {
        "model_id": MODEL_ID,
        "torch_dtype": torch.float32,
        "device": "cpu",
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_image_pixels": MAX_IMAGE_PIXELS,
    }


def _sanitize_detection_for_prompt(det):
    """Keep only compact fields needed for repair, dropping noisy keys."""
    if not isinstance(det, dict):
        return None
    text = str(det.get("text", "")).strip()
    bbox = det.get("bbox_xyxy", [])
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    return {"text": text, "bbox_xyxy": [x1, y1, x2, y2]}


def build_cleaning_prompt(detections: list, context: str = "") -> str:
    cleaned = []
    for det in detections or []:
        compact = _sanitize_detection_for_prompt(det)
        if compact is not None:
            cleaned.append(compact)

    # Keep prompt compact for lower latency on CPU inference.
    cleaned = cleaned[:MAX_PROMPT_DETECTIONS]
    packed_detections = json.dumps(
        {"detections": cleaned}, ensure_ascii=False, separators=(",", ":")
    )
    short_context = " ".join(str(context).split())[:180]
    context_line = f"Context hint (may be imperfect): {short_context}\n" if short_context else ""

    return (
        "Task: repair OCR detections for this historical map image.\n"
        "Use the image to correct obvious OCR errors in the provided detections.\n"
        "Rules:\n"
        "- Return JSON only (no markdown, no explanation).\n"
        "- Output must start with {\"detections\": and follow this schema:\n"
        '{"detections":[{"text":"string","bbox_xyxy":[int,int,int,int]}]}\n'
        "- Keep one object per text region.\n"
        "- Keep bbox unchanged unless clearly wrong.\n"
        "- Preserve visible language/script and spelling exactly as shown (including accents/case).\n"
        "- Use context only as a weak hint; if context conflicts with visible text, trust the image.\n"
        "- Fix OCR confusions only when visually clear.\n"
        "- Do not invent new places or words not visible in the image.\n"
        "- Remove stray markup/noise tokens (ex: </s>).\n"
        "- Remove obvious garbage/unreadable detections.\n"
        f"{context_line}"
        f"Input detections: {packed_detections}\n"
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
        ,
        {
            "role": "assistant",
            "content": ASSISTANT_JSON_PREFILL,
        }
    ]
    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        continue_final_message=True,
    )
    return processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    )


def load_model_and_processor(config):
    print("Loading model...")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        config["model_id"],
        torch_dtype=config["torch_dtype"],
    ).to(config["device"])

    processor = AutoProcessor.from_pretrained(
        config["model_id"],
    )
    print("Model ready.")
    return model, processor


def run_inference(
    model, processor, image, prompt,
    config, skip_special_tokens = True
):
    """Runs model inference on a pre-loaded PIL image with the given prompt."""

    inputs = build_ocr_messages(image, prompt, processor)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config["max_new_tokens"],
            do_sample=True,
            num_beams=1,
        )
    trimmed_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    decoded = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )[0]

    stripped = decoded.lstrip()
    if stripped.startswith('{"detections"'):
        return decoded
    return ASSISTANT_JSON_PREFILL + decoded

    
def _extract_first_json_value(text):
    decoder = json.JSONDecoder()
    starts = [i for i, ch in enumerate(text) if ch in "[{"]
    for start in starts:
        try:
            obj, _ = decoder.raw_decode(text[start:])
            return obj
        except json.JSONDecodeError:
            continue
    return None


def _normalize_detections(obj):
    if isinstance(obj, dict):
        detections = obj.get("detections", [])
    elif isinstance(obj, list):
        detections = obj
    else:
        return []

    cleaned = []
    for det in detections:
        compact = _sanitize_detection_for_prompt(det)
        if compact is not None:
            cleaned.append(compact)
    return cleaned


def parse_json_or_empty(raw_text):
    raw = raw_text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()

    obj = _extract_first_json_value(raw)
    if obj is None:
        return []
    return _normalize_detections(obj)


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


