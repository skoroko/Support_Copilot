from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "docs" / "agent-comparison-results"
REPORT_PATH = PROJECT_ROOT / "docs" / "agent-react-report.md"
EXPECTED_CORRECTNESS = {
    1: ("да", "да"),
    2: ("да", "да"),
    3: ("нет", "да"),
    4: ("да", "да"),
    5: ("да", "да"),
}


def _table_rows(report: str) -> dict[int, list[str]]:
    rows: dict[int, list[str]] = {}
    for line in report.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0].isdigit():
            rows[int(cells[0])] = cells
    return rows


def _as_int(value: str) -> int:
    return int(value.replace(" ", "").replace("*", ""))


def test_live_results_match_report_table_and_totals():
    result_paths = sorted(RESULTS_DIR.glob("*.json"))
    assert len(result_paths) == 5

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    report = REPORT_PATH.read_text(encoding="utf-8")
    rows = _table_rows(report)

    assert "pending" not in report.lower()
    assert set(rows) == {1, 2, 3, 4, 5}

    totals = {"naive_steps": 0, "react_steps": 0, "naive_tokens": 0, "react_tokens": 0, "revisions": 0}
    for index, payload in enumerate(payloads, start=1):
        naive = payload["naive"]
        react = payload["react"]
        row = rows[index]

        assert naive.get("error") is None
        assert react.get("error") is None
        assert _as_int(row[2]) == naive["steps"]
        assert _as_int(row[3]) == react["iterations"]
        assert (row[4], row[5]) == EXPECTED_CORRECTNESS[index]
        assert _as_int(row[6]) == naive["usage"]["total_tokens"]
        assert _as_int(row[7]) == react["usage"]["total_tokens"]
        assert _as_int(row[8]) == react["revisions"]

        totals["naive_steps"] += naive["steps"]
        totals["react_steps"] += react["iterations"]
        totals["naive_tokens"] += naive["usage"]["total_tokens"]
        totals["react_tokens"] += react["usage"]["total_tokens"]
        totals["revisions"] += react["revisions"]

    assert totals == {
        "naive_steps": 11,
        "react_steps": 12,
        "naive_tokens": 5_823,
        "react_tokens": 10_989,
        "revisions": 2,
    }
