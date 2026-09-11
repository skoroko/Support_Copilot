from __future__ import annotations

from importlib.metadata import version
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("PHOENIX_WORKING_DIR", str(ROOT / "var" / "phoenix-eval"))

from app.eval.metrics import build_judge, build_metrics  # noqa: E402


def main() -> None:
    packages = (
        "ragas",
        "pandas",
        "arize-phoenix",
        "arize-phoenix-evals",
        "arize-phoenix-otel",
        "openinference-instrumentation-llama-index",
    )
    for package in packages:
        print(f"{package}=={version(package)}")
    models = build_judge(
        os.getenv("RAG_EVAL_JUDGE_MODEL", "gpt-5.4-mini"),
        os.getenv("RAG_EVAL_EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key="sk-import-check-only",
    )
    metrics = build_metrics(models)
    print("metrics=" + ",".join(metrics))
    print("eval setup: OK")


if __name__ == "__main__":
    main()
