# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project --extra embeddings
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra embeddings
ENV PATH="/app/.venv/bin:$PATH"

# Train at build time with tuned hyper-parameters; the image ships only the artifacts.
FROM base AS train
RUN recsys train --tune --out /app/artifacts

FROM base AS serve
COPY --from=train /app/artifacts /app/artifacts
RUN useradd --create-home app && chown -R app /app
USER app
ENV RECSYS_ARTIFACTS=/app/artifacts
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["recsys", "serve", "--port", "8000"]
