# syntax=docker/dockerfile:1.7

FROM python:3.13-slim-bookworm@sha256:2325bb286ec344af3e5898cc224b5844e2707ac6e26b1632516fd3edc84a5e26 AS builder

COPY --from=ghcr.io/astral-sh/uv:0.6.10@sha256:57da96c4557243fc0a732817854084e81af9393f64dc7d172f39c16465b5e2ba /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock README.md .env.example ./
ARG INSTALL_RERANKER=false
ARG INSTALL_TRACING=false
RUN --mount=type=cache,target=/root/.cache/uv \
    extras="" && \
    if [ "$INSTALL_RERANKER" = "true" ]; then extras="$extras --extra reranker"; fi && \
    if [ "$INSTALL_TRACING" = "true" ]; then extras="$extras --extra tracing"; fi && \
    uv sync --locked --no-dev --no-install-project --no-editable $extras

COPY app ./app
COPY bot ./bot
COPY support_copilot ./support_copilot
COPY data ./data
COPY alembic.ini ./alembic.ini
COPY alembic ./alembic

FROM python:3.13-slim-bookworm@sha256:2325bb286ec344af3e5898cc224b5844e2707ac6e26b1632516fd3edc84a5e26 AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

WORKDIR /app

RUN useradd --create-home --uid 1000 appuser
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/app /app/app
COPY --from=builder --chown=appuser:appuser /app/bot /app/bot
COPY --from=builder --chown=appuser:appuser /app/support_copilot /app/support_copilot
COPY --from=builder --chown=appuser:appuser /app/data /app/data
COPY --from=builder --chown=appuser:appuser /app/alembic /app/alembic
COPY --from=builder --chown=appuser:appuser /app/alembic.ini /app/alembic.ini

USER appuser

EXPOSE 8000 9000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
