import os
import sys
import gc
import json

from celery import Celery

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

INTERMEDIATE_DIR = os.environ.get("INTERMEDIATE_DIR", "/data/intermediate")
FLORENCE_OUTPUT_SUFFIX = os.environ.get("FLORENCE_OUTPUT_SUFFIX", "-florence-ocr")
os.environ.setdefault("HF_HOME", "/app/models")

app = Celery(
    "florence_worker",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
)
app.conf.worker_prefetch_multiplier = 1
app.conf.task_acks_late = True

import main as florence


@app.task(name="florence.run_pipeline")
def run_florence(image_path: str) -> str:
    config = florence.get_runtime_config()
    model, processor = florence.load_model_and_processor(config)
    result = florence.run_pipeline(model, processor, image_path, config)

    del model, processor
    gc.collect()

    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(INTERMEDIATE_DIR, f"{base}{FLORENCE_OUTPUT_SUFFIX}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[Florence] Saved: {json_path}")

    return json_path


if __name__ == "__main__":
    app.worker_main(["worker", "--loglevel=info", "--concurrency=1"])
