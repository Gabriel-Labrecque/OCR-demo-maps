FROM ocr-base

# Qwen requires latest transformers
RUN pip install --no-cache-dir transformers

COPY --chown=appuser:appuser . /app/qwen/

USER appuser

WORKDIR /app/qwen

CMD ["python", "-m", "celery", "-A", "worker", "worker", "--loglevel=info", "--concurrency=1", "--queues=qwen"]
