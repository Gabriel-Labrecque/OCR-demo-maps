import os

from celery import Celery

_broker = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")

app = Celery(
    "ocr_worker",
    broker=_broker,
    backend=_broker,  # needed for .get() to retrieve subtask results
)

# Configure Celery to ensure tasks are acknowledged after completion and to avoid
# losing a task in the event of a crash.
app.conf.worker_prefetch_multiplier = 1
app.conf.task_acks_late = True


@app.task(name="ocr.run")
def run(image_path: str) -> str:
    """
    Dispatch the full OCR pipeline for one image and wait for it to finish.
    Blocking here ensures --concurrency=1 prevents the next pipeline from
    starting before the current florence→qwen chain fully completes.
    """
    print(f"Received OCR task to process {image_path}")

    # disable_sync_subtasks=False silences Celery's guard against calling .get()
    # inside a task. Safe here because the subtasks run in separate containers.
    out_path = app.signature("florence.run_pipeline", args=[image_path]) \
        .set(queue="florence").delay().get(disable_sync_subtasks=False)

    # out_path = app.signature("qwen.run_pipeline", args=[out_path, image_path]) \
    #    .set(queue="qwen").delay().get(disable_sync_subtasks=False)

    return out_path


if __name__ == "__main__":
    app.worker_main(["worker", "--loglevel=info", "--concurrency=1"])
