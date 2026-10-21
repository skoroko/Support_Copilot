from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import benchmark_manual_models as bench


def usage():
    return {"input_tokens": 10000, "output_tokens": 1000,
            "input_tokens_details": {"cached_tokens": 4000, "cache_creation_tokens": 2000},
            "output_tokens_details": {"reasoning_tokens": 700}}


def case():
    return {"id": "golden-01", "question": "Вопрос", "reference": "GOLD SECRET REFERENCE",
            "scoring_notes": "SCORING ONLY", "records": [{"marker": "[7]", "citation_number": 7,
                "source": {"file_name": "manual.pdf", "page": 3}, "text": "Документ"}]}


def test_output_includes_reasoning_once_and_mixed_cache_is_priced():
    assert bench.cost("gpt-6-luna", usage()) == pytest.approx((4000 * .1 + 4000 * .01 + 2000 * .125 + 1000 * .5) / 1e6)
    assert bench.cost("gpt-6-luna", usage(), uncached=True) == pytest.approx(.0015)


@pytest.mark.parametrize("bad", [{}, {"input_tokens": 10000},
    {"input_tokens": 2, "output_tokens": 1, "input_tokens_details": {"cached_tokens": 3}},
    {"input_tokens": -1, "output_tokens": 1}])
def test_missing_or_invalid_usage_never_becomes_free(bad):
    with pytest.raises(ValueError):
        bench.cost("gpt-6-luna", bad)


def test_ambiguous_cache_write_fields_rejected():
    data = usage()
    data["input_tokens_details"]["cache_write_tokens"] = 999
    with pytest.raises(ValueError):
        bench.cost("gpt-6-luna", data)


def test_gold_and_notes_never_reach_generation_but_reach_blind_judge():
    selected = case()
    prompt = bench.model_input(selected)
    assert "GOLD SECRET" not in prompt and "SCORING ONLY" not in prompt
    row = {"slot": 0, "model": "gpt-6-luna", "answer": "Ответ [7]", "status": "completed"}
    mapping, payload = bench.blind_payload(selected, [row])
    assert mapping == {"V01": 0}
    assert payload["reference_scoring_only"] == selected["reference"]
    assert "gpt-6-luna" not in str(payload)
    assert payload["documents"][0]["marker"] == "[7]"


def test_schedule_balances_every_model_position_and_repeats():
    items = bench.schedule([str(i) for i in range(20)], 2)
    assert len(items) == 200
    assert len({(i["case_id"], i["repeat"], i["model"]) for i in items}) == 200
    for model in bench.MODELS:
        assert Counter(i["position"] for i in items if i["model"] == model) == {p: 8 for p in range(5)}


def test_repeated_or_missing_judge_label_is_rejected():
    score = {"label": "V01", "correctness_completeness": 1, "grounding": None}
    bench.validate_scores([score], {"V01": 0})
    with pytest.raises(ValueError):
        bench.validate_scores([score, deepcopy(score)], {"V01": 0, "V02": 1})


def test_failed_attempt_is_not_sent_to_judge_and_has_zero_utility():
    rows = [{"slot": 0, "status": "failed", "error_type": "TimeoutError"},
            {"slot": 1, "status": "completed", "answer": "Ответ [7]"}]
    mapping, payload = bench.blind_payload(case(), rows)
    assert mapping == {"V01": 1} and len(payload["answers"]) == 1
    scores = bench.empty_answer_scores(rows)
    assert set(scores) == {0}
    assert scores[0]["correctness_completeness"] == 0 and scores[0]["grounding"] is None
    assert scores[0]["assessment_origin"] == "no_answer_rule"


def test_unknown_long_context_pricing_not_silently_applied():
    data = usage()
    data["input_tokens"] = 272001
    with pytest.raises(ValueError):
        bench.cost("gpt-6-luna", data)


@pytest.mark.parametrize("with_usage", [True, False])
def test_stream_artifact_survives_bad_usage_and_resume_never_repeats(monkeypatch, tmp_path, with_usage):
    from contextlib import asynccontextmanager

    calls = []
    final = SimpleNamespace(
        usage=SimpleNamespace(model_dump=lambda: usage()) if with_usage else None,
        status="completed", output_text="Подтверждённый ответ [7]", model="gpt-6-luna",
        service_tier="default", id="resp_test", incomplete_details=None,
    )

    class FakeStream:
        async def __aiter__(self):
            yield SimpleNamespace(type="response.created")
            yield SimpleNamespace(type="response.output_text.delta", delta="Ответ")

        async def get_final_response(self):
            return final

    @asynccontextmanager
    async def stream(**kwargs):
        calls.append(kwargs)
        yield FakeStream()

    @asynccontextmanager
    async def fake_client():
        yield SimpleNamespace(responses=SimpleNamespace(stream=stream))

    manifest = {"schedule": [{"slot": 0, "case_id": "golden-01", "repeat": 0,
                              "position": 0, "model": "gpt-6-luna"}], "generation_budget_usd": 3}
    monkeypatch.setattr(bench, "verify", lambda _: (manifest, [case()]))
    monkeypatch.setattr(bench, "async_client", fake_client)
    args = SimpleNamespace(output=tmp_path, limit=None)
    bench.run(args)
    row = bench.load(tmp_path / "samples/000.json")
    assert row["answer"] == final.output_text and row["response_id"] == "resp_test"
    assert row["ttft_s"] is not None
    if with_usage:
        assert row["status"] == "completed" and row["cost_usd"] > 0
    else:
        assert row["status"] == "cost_unknown" and "cost_usd" not in row
        assert row["reservation_usd"] > 0
    bench.run(args)
    assert len(calls) == 1
    assert calls[0]["reasoning"] == {"effort": "medium"}
    assert calls[0]["max_output_tokens"] == 8992
    assert calls[0]["store"] is False


def test_hard_wall_deadline_cancels_open_stream(monkeypatch, tmp_path):
    import asyncio
    from contextlib import asynccontextmanager

    cancelled = []

    async def stalled(*_args):
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.append(True)

    @asynccontextmanager
    async def fake_client():
        yield object()

    manifest = {"schedule": [{"slot": 0, "case_id": "golden-01", "repeat": 0,
                              "position": 0, "model": "gpt-6-luna"}],
                "generation_budget_usd": 3, "wall_deadline_s": .01}
    monkeypatch.setattr(bench, "verify", lambda _: (manifest, [case()]))
    monkeypatch.setattr(bench, "async_client", fake_client)
    monkeypatch.setattr(bench, "stream_response", stalled)
    bench.run(SimpleNamespace(output=tmp_path, limit=None))
    row = bench.load(tmp_path / "samples/000.json")
    assert row["status"] == "failed" and row["error_type"] == "TimeoutError"
    assert cancelled == [True]
    assert "cost_usd" not in row and row["reservation_usd"] > 0


def test_question_cluster_bootstrap_has_correct_direction_and_unit():
    from scripts.analyze_manual_models import paired_interval

    result = paired_interval([.25] * 20)
    assert result["mean_delta"] == .25
    assert result["ci95_low"] == result["ci95_high"] == .25
    assert result["n_questions"] == 20


@pytest.mark.parametrize("prior_reserve", [0.21, 1.99])
def test_streamed_judge_resume_and_unknown_cost_budget(monkeypatch, tmp_path, prior_reserve):
    import json
    from contextlib import asynccontextmanager

    selected = case()
    planned = bench.schedule([selected["id"]], 1)
    rows = [{**item, "status": "completed", "answer": "Ответ [7]"} for item in planned]
    for row in rows:
        bench.save(tmp_path / "samples" / f"{row['slot']:03}.json", row)
    mapping, _ = bench.blind_payload(selected, rows)
    raw = [{"label": label, "correctness_completeness": 1, "grounding": 1,
            "critical_unsupported": False, "citation_error": False, "notes": "Подтверждено"}
           for label in mapping]
    final = SimpleNamespace(usage=SimpleNamespace(model_dump=lambda: usage()),
                            output_text=json.dumps({"scores": raw}), status="completed", id="judge_test",
                            model="gpt-6.1-sol", service_tier="default")
    calls = []

    @asynccontextmanager
    async def fake_client():
        yield object()

    async def fake_stream(_api, kwargs, _started):
        calls.append(kwargs)
        return final, .1

    manifest = {"schedule": planned, "judge_budget_usd": 2,
                "prior_judge_unknown_reserve_usd": prior_reserve, "judge_wall_deadline_s": 240}
    monkeypatch.setattr(bench, "verify", lambda _: (manifest, [selected]))
    monkeypatch.setattr(bench, "async_client", fake_client)
    monkeypatch.setattr(bench, "stream_response", fake_stream)
    args = SimpleNamespace(output=tmp_path)
    if prior_reserve > 1:
        with pytest.raises(ValueError, match="budget exhausted"):
            bench.judge(args)
        assert not calls
    else:
        bench.judge(args)
        saved = bench.load(tmp_path / "reviews" / (selected["id"] + ".json"))
        assert saved["status"] == "completed" and saved["raw_scores"] == final.output_text
        assert saved["cost_usd"] == pytest.approx(bench.cost("gpt-6.1-sol", usage()))
        bench.judge(args)
        assert len(calls) == 1
        assert calls[0]["reasoning"] == {"effort": "high"}
        assert calls[0]["text"]["format"]["strict"] is True
        saved["status"] = "failed"
        bench.save(tmp_path / "reviews" / (selected["id"] + ".json"), saved)
        with pytest.raises(ValueError, match="Judge failed"):
            bench.judge(args)
        assert len(calls) == 1  # Saved failures are reported, never replayed.


@pytest.mark.parametrize("budget", [2.0, 0.3])
def test_parallel_judge_reserves_inflight_calls_before_dispatch(monkeypatch, tmp_path, budget):
    import asyncio
    import json
    from contextlib import asynccontextmanager

    cases = [case(), {**case(), "id": "golden-02"}]
    planned = bench.schedule([c["id"] for c in cases], 1)
    for item in planned:
        bench.save(tmp_path / "samples" / f"{item['slot']:03}.json",
                   {**item, "status": "completed", "answer": "Ответ [7]"})
    active = peak = calls = 0

    @asynccontextmanager
    async def fake_client():
        yield object()

    async def fake_stream(_api, kwargs, _started):
        nonlocal active, peak, calls
        active += 1
        calls += 1
        peak = max(peak, active)
        await asyncio.sleep(.01)
        raw = [{"label": a["label"], "correctness_completeness": 1, "grounding": 1,
                "critical_unsupported": False, "citation_error": False, "notes": "Подтверждено"}
               for a in json.loads(kwargs["input"])["answers"]]
        active -= 1
        return SimpleNamespace(usage=SimpleNamespace(model_dump=lambda: usage()),
                               output_text=json.dumps({"scores": raw}), status="completed", id="judge_parallel",
                               model="gpt-6.1-sol", service_tier="default"), .1

    manifest = {"schedule": planned, "judge_budget_usd": budget, "judge_concurrency": 2}
    monkeypatch.setattr(bench, "verify", lambda _: (manifest, cases))
    monkeypatch.setattr(bench, "async_client", fake_client)
    monkeypatch.setattr(bench, "stream_response", fake_stream)
    if budget < 1:
        with pytest.raises(ValueError, match="budget exhausted"):
            bench.judge(SimpleNamespace(output=tmp_path))
        assert calls == peak == 1
    else:
        bench.judge(SimpleNamespace(output=tmp_path))
        assert calls == peak == 2
    assert all(bench.load(p)["status"] == "completed" for p in (tmp_path / "reviews").glob("*.json"))


@pytest.mark.parametrize("tamper", ["answer", "score", "judge_cost", "failed_reservation"])
def test_publication_audit_rejects_drift(monkeypatch, tmp_path, tamper):
    import json

    from scripts import analyze_manual_models as analysis

    selected = case()
    planned = bench.schedule([selected["id"]], 1)
    input_hash = bench.digest(bench.model_input(selected).encode())
    manifest = {"schedule": planned, "input_sha256": {selected["id"]: input_hash},
                "judge_model": "gpt-6.1-sol", "nonreasoning_output_cap": 800, "luna_total_output_cap": 8992}
    monkeypatch.setattr(analysis, "verify", lambda _: (manifest, [selected]))
    rows = []
    for item in planned:
        row = {**item, "status": "completed", "input_sha256": input_hash,
               "served_tier": "default", "served_model": item["model"], "usage": usage(),
               "cost_usd": bench.cost(item["model"], usage()), "answer": "Ответ [7]"}
        if tamper == "failed_reservation" and item["model"] == "gpt-4.1-mini":
            row = {**item, "status": "failed", "input_sha256": input_hash, "error_type": "TimeoutError",
                   "latency_s": 180, "reservation_usd": bench.reservation(item["model"], bench.model_input(selected), 800)}
        rows.append(row)
        bench.save(tmp_path / "samples" / f"{item['slot']:03}.json", row)
    mapping, payload = bench.blind_payload(selected, rows)
    raw = [{"label": label, "correctness_completeness": 1, "grounding": 1,
            "critical_unsupported": False, "citation_error": False, "notes": "Подтверждено"}
           for label in mapping]
    review = {"case_id": selected["id"], "status": "completed", "mapping": mapping,
              "served_tier": "default", "served_model": "gpt-6.1-sol", "usage": usage(),
              "cost_usd": bench.cost("gpt-6.1-sol", usage()),
              "input_sha256": bench.digest(json.dumps(payload, ensure_ascii=False).encode()),
              "raw_scores": json.dumps({"scores": raw}),
              "scores": [{**s, "slot": mapping[s["label"]]} for s in raw]}
    bench.save(tmp_path / "reviews" / (selected["id"] + ".json"), review)
    analysis.audit(tmp_path)
    if tamper == "answer":
        rows[0]["answer"] = "Подменённый ответ"
        bench.save(tmp_path / "samples/000.json", rows[0])
    elif tamper == "score":
        review["scores"][0]["correctness_completeness"] = 0
        bench.save(tmp_path / "reviews" / (selected["id"] + ".json"), review)
    elif tamper == "judge_cost":
        review["cost_usd"] = 0
        bench.save(tmp_path / "reviews" / (selected["id"] + ".json"), review)
    else:
        rows[1]["reservation_usd"] *= 2
        bench.save(tmp_path / "samples/001.json", rows[1])
    with pytest.raises(ValueError):
        analysis.audit(tmp_path)
