# syntax=docker/dockerfile:1.7

FROM python:3.13-slim-bookworm@sha256:2325bb286ec344af3e5898cc224b5844e2707ac6e26b1632516fd3edc84a5e26 AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.18@sha256:3adc3706091ce7c2fe595e669628caedd6d951551b92b258b7e7dbe06d9440bc /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CONCURRENT_INSTALLS=2 \
    UV_CONCURRENT_BUILDS=2 \
    UV_CONCURRENT_DOWNLOADS=4 \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
ARG INSTALL_RERANKER=false
ARG INSTALL_TRACING=false
RUN --mount=type=cache,target=/root/.cache/uv \
    extras="" && \
    if [ "$INSTALL_RERANKER" = "true" ]; then extras="$extras --extra reranker"; fi && \
    if [ "$INSTALL_TRACING" = "true" ]; then extras="$extras --extra tracing"; fi && \
    uv sync --locked --no-dev --no-install-project --no-editable $extras

# Ship the verified tokenizer data with the image: the first chat must not
# download it synchronously while holding the agent's event loop.
ENV TIKTOKEN_CACHE_DIR=/app/tokenizer_cache
RUN python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"

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
    LOG_LEVEL=INFO \
    TIKTOKEN_CACHE_DIR=/app/tokenizer_cache

WORKDIR /app

RUN useradd --create-home --uid 1000 appuser
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/tokenizer_cache /app/tokenizer_cache
COPY --from=builder --chown=appuser:appuser /app/app /app/app
COPY --from=builder --chown=appuser:appuser /app/bot /app/bot
COPY --from=builder --chown=appuser:appuser /app/support_copilot /app/support_copilot
COPY --from=builder --chown=appuser:appuser /app/data /app/data
COPY --from=builder --chown=appuser:appuser /app/alembic /app/alembic
COPY --from=builder --chown=appuser:appuser /app/alembic.ini /app/alembic.ini

USER appuser

EXPOSE 8000 9000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
