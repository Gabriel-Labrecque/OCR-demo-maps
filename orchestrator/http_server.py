from fastapi import FastAPI, Query
from celery import Celery
import os

app = FastAPI()

_broker = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
celery_app = Celery(
    "ocr_worker",
    broker=_broker,
    backend=_broker,
)

@app.post("/process")
def process_image(filename: str = Query(..., description="Image filename in input/ directory")):
    print(f"Received request to process {filename}")
    image_path = f"/data/input/{filename}"
    if not os.path.exists(image_path):
        return {"error": f"File {filename} does not exist."}
    result = celery_app.signature("ocr.run", args=[image_path]).delay()
    output_path = result.get()
    return {"output_path": output_path}
