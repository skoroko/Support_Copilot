import csv
from pathlib import Path

import pytest

from scripts.evaluate_citation_judge import evaluate, load_answers


def test_loader_caps_saved_answers_and_rejects_empty_rows(tmp_path: Path):
    path = tmp_path / "saved.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["response", "retrieved_contexts"])
        writer.writeheader()
        writer.writerows({"response": f"Answer {index}", "retrieved_contexts": "PRIVATE"} for index in range(40))
    assert len(load_answers(path, 34)) == 34
    with pytest.raises(ValueError, match="between 1 and 34"):
        load_answers(path, 35)


@pytest.mark.asyncio
async def test_judge_preserves_failure_and_checkpoints_without_raw_answers():
    class Judge:
        calls = 0
        last_error = None
        last_usage = None

        async def agenerate(self, prompt, response_model):
            self.calls += 1
            assert "PRIVATE_CONTEXT" not in prompt
            if self.calls == 2:
                self.last_error = {"type": "ConnectionError"}
                raise ConnectionError("unavailable")
            return response_model(value="yes", reason="marker")

    judge = Judge()
    checkpoints = []
    rows = await evaluate(["Saved answer [1]", "Second answer", "Third [2]"], judge,
                          on_row=lambda rows: checkpoints.append(len(rows)))
    assert judge.calls == 3
    assert checkpoints == [1, 2, 3]
    assert rows[1]["has_citation_judge"] is None
    assert rows[1]["error"] == {"type": "ConnectionError"}
    assert rows[1]["agreement"] is None
    assert rows[2]["status"] == "completed"
    assert all("response" not in row and "retrieved_contexts" not in row for row in rows)
