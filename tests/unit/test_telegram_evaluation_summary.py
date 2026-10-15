"""Saved-data aggregation must preserve denominators and unmeasured values."""

import csv
import json

import pytest

from scripts import summarize_telegram_evaluation as summary


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def fixture(directory, *, run_id="run", expansion=True):
    directory.mkdir()
    cases = [{"id": "g1", "group": "golden", "question": "First?", "reference": "Fact", "retrieval_contract": None},
             {"id": "g2", "group": "golden", "question": "Second?", "reference": "Fact"},
             {"id": "f1", "group": "followup", "question": "And?"},
             {"id": "r1", "group": "refusal", "question": "Price?"}]
    rows = [{"id": "g1", "status": "completed", "response": "Fact [1]", "sources": [{"id": "chunk"}], "wall_latency_ms": 1000, "ttft_ms": 700},
            {"id": "g2", "status": "failed", "error": {"type": "Timeout"}, "wall_latency_ms": 4000},
            {"id": "f1", "status": "completed", "response": "Fact", "sources": [], "wall_latency_ms": 2000},
            {"id": "r1", "status": "completed", "response": summary.NO_ANSWER, "sources": []}]
    reviews = [{"id": row["id"], "mode": "brief", "correctness": 0 if row["status"] == "failed" else 1,
                "grounding": None if row["id"] in {"g2", "r1"} else .5, "notes": "Manual inspection"} for row in rows]
    if expansion:
        rows[0]["expansion"] = {"status": "completed", "confirmed_latency_ms": 800,
                                "result": {"answer": "Full fact", "sources": [], "message_id": "expanded"}}
        reviews.append({"id": "g1", "mode": "expanded", "correctness": .5, "grounding": 0, "notes": "Partial"})
    run = {"run_id": run_id, "scenarios": cases, "results": rows,
           "options": {"api_url": "http://localhost", "expansion_indices": [1], "no_expansions": False}}
    write(directory / "run.json", run)
    write(directory / "semantic_review.json", {"run_id": run_id, "results": reviews})
    return run, reviews


def test_failed_requests_count_zero_and_absent_measurements_stay_unobserved(tmp_path):
    directory = tmp_path / "run"
    fixture(directory)
    result = summary.aggregate(directory)
    assert result["golden_correctness_and_completeness"] == {
        "denominator": 2, "sum": 1, "mean": .5, "full": 1, "partial": 0, "zero": 1}
    assert result["golden_cited_fragment_grounding"]["denominator"] == 1
    assert result["refusals"]["correct"] == 1
    assert result["timing"]["golden_all_including_failure"]["p95_nearest_rank_s"] == 4
    assert result["timing"]["golden_successful"]["mean_s"] == 1
    assert result["golden_structural_proxies"] is None and result["checkpoints"] is None
    assert result["fresh_search_after_followup"]["counts"]["f1"] is None
    assert result["source_transport"]["sources_match_persisted"] == {"observed": 0, "passed": 0}
    assert result["expansion_contract"]["replay_equal"] == {"observed": 0, "passed": 0}
    assert (directory / "answers.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    with (directory / "answers.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
        assert len(rows) == 5
        assert next(row for row in rows if row["mode"] == "expanded")["latency_seconds"] == "0.8"


def test_idempotent_outputs_do_not_hash_themselves(tmp_path):
    directory = tmp_path / "run"
    fixture(directory)
    raw = (directory / "run.json").read_bytes()
    first = summary.aggregate(directory)
    second = summary.aggregate(directory)
    assert first == second
    assert (directory / "run.json").read_bytes() == raw
    assert "summary.json" not in second["artifact_sha256"]
    assert "comparison.json" not in second["artifact_sha256"]
    assert second["artifact_sha256"]["answers.csv"] == summary.sha(directory / "answers.csv")


@pytest.mark.parametrize("with_sources", [False, True])
def test_freeform_refusal_without_facts_can_leave_grounding_unmeasured(tmp_path, with_sources):
    directory = tmp_path / "run"
    run, _ = fixture(directory, expansion=False)
    refusal = run["results"][-1]
    refusal["response"] = "Не могу определить SLA: договор не предоставлен."
    refusal["sources"] = [{"id": "chunk"}] if with_sources else []
    write(directory / "run.json", run)
    if with_sources:
        with pytest.raises(ValueError, match="grounding"):
            summary.aggregate(directory)
    else:
        result = summary.aggregate(directory)
        assert result["refusals"]["correct"] == 1
        assert result["refusals"]["canonical_no_answer_without_sources"] == 0


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra", "score", "boolean", "failed", "null", "started", "missing_field"])
def test_invalid_review_or_unfinished_run_is_rejected_before_writes(tmp_path, change):
    directory = tmp_path / "run"
    run, reviews = fixture(directory)
    if change == "missing":
        reviews.pop()
    elif change == "duplicate":
        reviews.append(reviews[0])
    elif change == "extra":
        reviews.append({**reviews[0], "id": "extra"})
    elif change == "score":
        reviews[0]["grounding"] = .75
    elif change == "boolean":
        reviews[0]["correctness"] = True
    elif change == "failed":
        reviews[1]["correctness"] = 1
    elif change == "null":
        reviews[0]["grounding"] = None
    elif change == "missing_field":
        reviews[-1].pop("grounding")
    else:
        run["results"][0]["status"] = "started"
    write(directory / "run.json", run)
    write(directory / "semantic_review.json", {"run_id": "run", "results": reviews})
    with pytest.raises(ValueError):
        summary.aggregate(directory)
    assert not (directory / "summary.json").exists()


def test_nearest_rank_p95_and_empty_measurement():
    assert summary.timing([{"wall_latency_ms": index * 1000} for index in range(1, 21)])["p95_nearest_rank_s"] == 19
    assert summary.timing([{}])["mean_s"] is None


def test_checkpoint_hash_and_observed_search_calls(tmp_path):
    directory = tmp_path / "run"
    fixture(directory)
    cp = {"run_id": "run", "run_sha256": summary.sha(directory / "run.json"), "threads": {
        "thread": {"status": "exported", "owner": "followup", "scenario_ids": ["g1", "f1"],
                   "messages": [{"type": "human"}, {"type": "ai", "tool_calls": [{"name": "search_knowledge_base"}]},
                                {"type": "tool", "content": json.dumps({"sources": [], "contexts": []})},
                                {"type": "human"}, {"type": "ai", "tool_calls": [{"name": "search_knowledge_base"}]},
                                {"type": "tool", "content": json.dumps({"evidence": []})}]}}}
    write(directory / "checkpoints.json", cp)
    result = summary.aggregate(directory)
    assert result["fresh_search_after_followup"]["counts"] == {"f1": 1}
    assert result["checkpoints"]["tool_calls_after_latest_human"] == {"followup": ["search_knowledge_base"]}
    cp["run_sha256"] = "wrong"
    write(directory / "checkpoints.json", cp)
    with pytest.raises(ValueError, match="hash"):
        summary.aggregate(directory)


@pytest.mark.parametrize("mismatch", [None, "scenario", "options", "config", "docstore", "dataset", "threshold_alias", "threshold_value"])
def test_comparison_requires_unchanged_inputs_and_preserves_baseline(tmp_path, monkeypatch, mismatch):
    datasets = tmp_path / "tests" / "eval"
    datasets.mkdir(parents=True)
    write(datasets / "golden_dataset.json", {"items": []})
    monkeypatch.setattr(summary, "ROOT", tmp_path)
    fingerprint = {"golden_dataset.json": summary.sha(datasets / "golden_dataset.json")}
    baseline, current = tmp_path / "baseline", tmp_path / "current"
    fixture(baseline, run_id="before", expansion=False)
    run, _ = fixture(current, run_id="after")
    manifest = {"dataset_sha256": fingerprint, "config": {"rag_top_k": 10, "rag_score_threshold": .45}, "docstore": {"sha256": "same"}}
    write(baseline / "runtime_manifest.json", manifest)
    write(current / "runtime_manifest.json", manifest)
    before = {path.name: path.read_bytes() for path in baseline.iterdir()}
    if mismatch in {"scenario", "options"}:
        if mismatch == "scenario":
            run["scenarios"][0]["question"] = "Changed"
        else:
            run["options"]["expansion_indices"] = [2]
        write(current / "run.json", run)
    elif mismatch in {"threshold_alias", "threshold_value"}:
        manifest["config"] = {"rag_top_k": 10, "rag_min_score": .45 if mismatch == "threshold_alias" else .5}
        write(current / "runtime_manifest.json", manifest)
    elif mismatch:
        manifest[mismatch if mismatch != "dataset" else "dataset_sha256"] = {"different": "value"}
        write(current / "runtime_manifest.json", manifest)
    if mismatch and mismatch != "threshold_alias":
        with pytest.raises(ValueError):
            summary.aggregate(current, baseline)
    else:
        summary.aggregate(current, baseline)
        comparison = summary.read(current / "comparison.json")
        assert comparison["paired_cohort"]["denominator"] == 4
        assert comparison["headline"]["expanded"]["baseline"]["denominator"] == 0
    assert before == {path.name: path.read_bytes() for path in baseline.iterdir()}
