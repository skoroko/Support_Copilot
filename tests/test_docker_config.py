from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DockerConfigTests(unittest.TestCase):
    def test_dockerfile_matches_homework_contract(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("# syntax=docker/dockerfile:1.7", dockerfile)
        self.assertIn("FROM python:3.13-slim-bookworm AS builder", dockerfile)
        self.assertIn("FROM python:3.13-slim-bookworm AS runtime", dockerfile)
        self.assertIn("COPY --from=ghcr.io/astral-sh/uv:0.6.10 /uv /uvx /bin/", dockerfile)
        self.assertIn("UV_COMPILE_BYTECODE=1", dockerfile)
        self.assertIn("UV_LINK_MODE=copy", dockerfile)
        self.assertIn("UV_PYTHON_DOWNLOADS=0", dockerfile)
        self.assertIn("COPY requirements.txt README.md .env.example ./", dockerfile)
        self.assertIn("COPY app ./app", dockerfile)
        self.assertIn("useradd --create-home --uid 1000 appuser", dockerfile)
        self.assertIn("COPY --from=builder --chown=appuser:appuser /app /app", dockerfile)
        self.assertIn("USER appuser", dockerfile)
        self.assertIn("EXPOSE 8000", dockerfile)
        self.assertIn('CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]', dockerfile)

    def test_compose_has_app_redis_healthchecks_and_no_redis_ports(self) -> None:
        compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")

        self.assertIn("image: llm-service:v1", compose)
        self.assertIn("ports:\n      - \"8000:8000\"", compose)
        self.assertIn("APP_REDIS_URL: redis://redis:6379/0", compose)
        self.assertIn("condition: service_healthy", compose)
        self.assertIn("http://localhost:8000/ready", compose)
        self.assertIn("image: redis:7.4-alpine", compose)
        self.assertIn("redis_data:/data", compose)
        self.assertIn('test: ["CMD", "redis-cli", "ping"]', compose)
        redis_section = compose.split("  redis:", maxsplit=1)[1].split("volumes:", maxsplit=1)[0]
        self.assertNotIn("ports:", redis_section)

    def test_dockerignore_excludes_secrets_tests_and_python_artifacts(self) -> None:
        dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

        for pattern in [".git", ".env", ".env.*", "!.env.example", "tests/", "__pycache__/", "*.pyc"]:
            self.assertIn(pattern, dockerignore)


if __name__ == "__main__":
    unittest.main()
