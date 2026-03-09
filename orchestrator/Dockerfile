FROM ocr-base

COPY --chown=appuser:appuser worker.py /app/

USER appuser

WORKDIR /app

CMD ["python", "-m", "celery", "-A", "worker", "worker", "--loglevel=info", "--concurrency=1", "--queues=ocr"]
