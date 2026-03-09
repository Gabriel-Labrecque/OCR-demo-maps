FROM ocr-base

# Florence-2 requires transformers==4.43.0
RUN pip install --no-cache-dir transformers==4.43.0 opencv-python-headless

COPY --chown=appuser:appuser . /app/florence/

USER appuser

WORKDIR /app/florence

CMD ["python", "-m", "celery", "-A", "worker", "worker", "--loglevel=info", "--concurrency=1", "--queues=florence"]
