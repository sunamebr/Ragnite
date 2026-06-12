FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RAGNITE_DATA_DIR=/data \
    RAGNITE_HOST=0.0.0.0

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[server,mcp,anthropic,qdrant,pdf]"

RUN useradd --create-home ragnite && mkdir -p /data && chown ragnite /data
USER ragnite
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import httpx; httpx.get('http://127.0.0.1:8000/healthz').raise_for_status()"

CMD ["ragnite", "serve", "--host", "0.0.0.0", "--port", "8000"]
