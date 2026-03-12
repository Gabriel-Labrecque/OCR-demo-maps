import os
import sys
import gc
import json

from celery import Celery
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/data/result")
QWEN_OUTPUT_SUFFIX = os.environ.get("QWEN_OUTPUT_SUFFIX", "-qwen-cleaned")
os.environ.setdefault("HF_HOME", "/app/models")

app = Celery(
    "qwen_worker",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
)
app.conf.worker_prefetch_multiplier = 1
app.conf.task_acks_late = True

import main as qwen


@app.task(name="qwen.run_pipeline")
def run_qwen(intermediate_json: str, image_path: str) -> str:

    print(f"Received Qwen OCR task to process {image_path}")

    with open(intermediate_json, "r", encoding="utf-8") as f:
        florence_data = json.load(f)

    image_size = florence_data.get("image_size", {})
    detections = florence_data.get("detections", [])

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    if w * h > qwen.MAX_IMAGE_PIXELS:
        scale = (qwen.MAX_IMAGE_PIXELS / (w * h)) ** 0.5
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    config = qwen.get_runtime_config()
    model, processor = qwen.load_model_and_processor(config)

    context = qwen.run_inference(
        model, processor, image, qwen.build_context_prompt(),
        config, skip_special_tokens=True,
    ).strip()
    print(f"[Qwen] Context: {context}")

    prompt = qwen.build_cleaning_prompt(detections, context)
    raw = qwen.run_inference(model, processor, image, prompt, config)
    cleaned = qwen.parse_json_or_empty(raw)

    del model, processor
    gc.collect()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base}{QWEN_OUTPUT_SUFFIX}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"image_size": image_size, "detections": cleaned.get("detections", [])},
                  f, ensure_ascii=False, indent=2)
    print(f"[Qwen] Saved: {out_path}")

    try:
        os.remove(intermediate_json)
        print(f"[Qwen] Cleaned up: {intermediate_json}")
    except OSError as e:
        print(f"[Qwen] Warning: could not remove intermediate file: {e}")

    return out_path


if __name__ == "__main__":
    app.worker_main(["worker", "--loglevel=info", "--concurrency=1"])
