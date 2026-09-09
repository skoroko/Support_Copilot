# syntax=docker/dockerfile:1.7

FROM python:3.13-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.6.10 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY requirements.txt README.md .env.example ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/.venv/bin/python -r requirements.txt

COPY app ./app
COPY bot ./bot
COPY data ./data
COPY alembic.ini ./alembic.ini
COPY alembic ./alembic

FROM python:3.13-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

WORKDIR /app

RUN useradd --create-home --uid 1000 appuser
COPY --from=builder --chown=appuser:appuser /app /app

USER appuser

EXPOSE 8000 9000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
