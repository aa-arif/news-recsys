# Serving image: ONNX Runtime + FAISS + FastAPI. Model artifacts are mounted, not baked
# in, because they are rebuilt far more often than the code and are far larger than it.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/usr/local

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first: this layer only rebuilds when the lockfile changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY scripts ./scripts
RUN uv sync --frozen --no-dev

ENV NEWSREC_SERVE_HOST=0.0.0.0 \
    NEWSREC_SERVE_PORT=8000

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "news_recsys.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "warning"]
