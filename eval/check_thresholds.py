from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = ROOT / "eval" / "runs"
DEFAULT_THRESHOLDS = ROOT / "eval" / "thresholds.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check latest evaluation run against release thresholds.")
    parser.add_argument("run", nargs="?", type=Path, help="Evaluation run JSON. Defaults to latest eval/runs/*.json.")
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    return parser.parse_args()


def latest_run_path() -> Path:
    runs = sorted(DEFAULT_RUNS_DIR.glob("*.json"))
    if not runs:
        raise FileNotFoundError(f"No evaluation runs found in {DEFAULT_RUNS_DIR}")
    return runs[-1]


def load_thresholds(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        payload = yaml.safe_load(text)
        if not isinstance(payload, dict):
            raise ValueError("thresholds.yaml must contain a mapping")
        return {str(key): float(value) for key, value in payload.items()}
    except ImportError:
        thresholds: dict[str, float] = {}
        for line in text.splitlines():
            stripped = line.split("#", 1)[0].strip()
            if not stripped:
                continue
            key, value = stripped.split(":", 1)
            thresholds[key.strip()] = float(value.strip())
        return thresholds


def load_run(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Evaluation run must be a JSON object")
    if not isinstance(payload.get("aggregates"), dict):
        raise ValueError("Evaluation run must contain aggregates object")
    return payload


def check_thresholds(run: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    failures: list[str] = []
    aggregates = run["aggregates"]
    for metric, minimum in thresholds.items():
        actual = float(aggregates.get(metric, 0))
        if actual < minimum:
            failures.append(f"{metric}: {actual} < {minimum}")
    return failures


def main() -> int:
    args = parse_args()
    run_path = args.run or latest_run_path()
    run = load_run(run_path)
    thresholds = load_thresholds(args.thresholds)
    failures = check_thresholds(run, thresholds)

    if failures:
        print(f"Evaluation thresholds failed for {run_path}:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"Evaluation thresholds passed for {run_path}")
    for metric, minimum in thresholds.items():
        print(f"- {metric}: {run['aggregates'][metric]} >= {minimum}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
