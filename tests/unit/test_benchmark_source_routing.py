"""Offline experiment invariants; no provider/corpus/credentials are needed."""
import hashlib
import json
from copy import deepcopy

import httpx
import pytest

from app.services.table_selection import TableSelectionService
from scripts import benchmark_source_routing as bench


@pytest.fixture
def fixture():
    return bench.load_fixture(bench.DEFAULT_FIXTURE)


V2_PATH = bench.DEFAULT_FIXTURE.with_name("source_routing_v2.json")
V2_HASH = "8c49cffb5ecf03133db73be65279047ab29b1cd95b1253cd9fccb35966762dd8"
V2_INPUT_HASHES = [
    "ec0eb332d7704fe22d20726588162dec6ad046b4fac92161521516a7b4f5436b",
    None,
    "d26a57968ac51a6f24c1cbde6672670475dbef4cdd881552b32dc0ced0045890",
    "f3ade62a664cf8dd99db08baff1c98de21a4c7defb5f198775508fe71ae2a470",
    "ca761b9196caffe2a7603835b911b2e4f608e38891abbaf0af9d6bffae2a664d",
    "410d08f864e9668ec283a14c487392443e01609fdd2dbb3ac912db025264c676",
    "8766a225adcde5612ebb1788735de1191305937145954e22fa5b3fe4650b675d",
    "11cbe2873b1f9334e7e22502ffa4741d4815e0e9f341f796522d4d915b474562",
]


def reseal(fixture):
    fixture["fixture_sha256"] = bench.digest({key: value for key, value in fixture.items() if key != "fixture_sha256"})


def test_compact_fixture_preserves_all_original_production_input_hashes(fixture):
    assert len(fixture["candidates"]) == 96 and len(fixture["cases"]) == 8
    assert len(bench.encoded(fixture)) < 100_000
    assert all(set(candidate) == {"id", "description"} for candidate in fixture["candidates"])
    assert fixture["provenance"]["original_fixture_sha256"] == "c8232beddeb1df734814b94c29c5362e46c03ffa694adaee33c8951db50a3dc1"
    for case in fixture["cases"]:
        if case["semantic_invoked"]:
            data = bench.payload(fixture, case)
            shared, _ = TableSelectionService._input(**data)
            assert hashlib.sha256(shared.encode()).hexdigest() == case["input_sha256"]
            assert json.loads(shared) == data
    assert sum(case["semantic_invoked"] for case in fixture["cases"]) == 7


def test_recorded_results_reproduce_review_without_live_calls(fixture):
    summary = bench.summarize(fixture, fixture["recorded_results"])
    modes = {row["mode"]: row for row in summary["modes"]}
    assert [(modes[mode]["correct"], modes[mode]["rows"]) for mode in bench.MODES] == [(4, 16), (4, 16), (14, 16), (16, 16)]
    assert (modes["decisions"]["provider_correct"], modes["decisions"]["provider_calls"]) == (12, 14)
    assert (modes["responses"]["provider_correct"], modes["responses"]["provider_calls"]) == (14, 14)
    assert modes["decisions"]["lexical_skips"] == modes["responses"]["lexical_skips"] == 2
    assert modes["decisions"]["service_seconds"]["median"] == pytest.approx(.913098)
    assert modes["responses"]["service_seconds"]["median"] == pytest.approx(2.620379)
    assert summary["paired_service_timing"] == {"pairs": 14, "decisions_faster": 14}


@pytest.mark.parametrize("change", ["hash", "payload", "duplicate_candidate", "duplicate_order", "oversized_description"])
def test_fixture_drift_and_contract_violations_fail_closed(fixture, change):
    if change == "hash":
        fixture["cases"][0]["question"] += "?"
    else:
        if change == "payload":
            fixture["cases"][0]["question"] += "?"
        elif change == "duplicate_candidate":
            fixture["candidates"][1] = deepcopy(fixture["candidates"][0])
        elif change == "duplicate_order":
            fixture["cases"][0]["candidate_order"][1] = fixture["cases"][0]["candidate_order"][0]
        else:
            fixture["candidates"][0]["description"] = "x" * 1501
        reseal(fixture)
    with pytest.raises(bench.BenchmarkError):
        bench.validate_fixture(fixture)


@pytest.mark.parametrize("change", ["order", "hash", "cache", "selection", "scoring", "missing", "nan"])
def test_result_validation_does_not_accept_false_pairs_or_false_successes(fixture, change):
    results = deepcopy(fixture["recorded_results"])
    row = next(row for row in results["rows"] if row["provider_called"])
    if change == "order":
        results["rows"][2], results["rows"][3] = results["rows"][3], results["rows"][2]
    elif change == "hash":
        row["input_sha256"] = "wrong"
    elif change == "cache":
        row["cache"] = "hit"
    elif change == "selection":
        row["selected_id"] = "invented"
    elif change == "scoring":
        negative = next(item for item in results["rows"] if item["provider_called"] and item["human_expected_id"] is None)
        negative.update(error_type="ReadTimeout", selected_id=None, correct_choice=True)
    elif change == "nan":
        row["service_duration_ms"] = float("nan")
    else:
        results["rows"].pop()
    with pytest.raises(bench.BenchmarkError):
        bench.validate_results(fixture, results)


@pytest.mark.parametrize("prospective", [False, True])
def test_fresh_clients_ab_ba_and_shared_request_bytes_with_mock_transport(fixture, prospective, tmp_path, capsys):
    if prospective:
        fixture = bench.load_fixture(V2_PATH)
    clients, requests = [], []
    expected = {case["question"]: case["expected_id"] or "none" for case in fixture["cases"]}

    def handler(request):
        body = json.loads(request.content)
        data = json.loads(body["input"])
        choice = expected[data["question"]]
        requests.append((request.url.path, body["input"]))
        if request.url.path.endswith("decisions"):
            values = [row["value"] for row in body["questions"][0]["choices"]]
            reply = {"answers": [{"type": "choice", "name": "table", "choice": choice, "confidence": .7,
                "probabilities": [{"value": value, "probability": 1. if value == choice else 0.} for value in values]}]}
        else:
            reply = {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": json.dumps({"table": choice})}]}]}
        return httpx.Response(200, json=reply)

    def factory(**kwargs):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        clients.append(client)
        service = TableSelectionService(**kwargs, client=client)
        service._owns_client = True
        return service

    result = bench.run_benchmark(fixture, model="mock-model", base_url="https://provider.invalid/v1",
                                 api_key="test-only", service_factory=factory)
    assert len(clients) == len(requests) == 28 and all(client.is_closed for client in clients)
    assert all(row["cache"] == "miss" for row in result["rows"] if row["provider_called"])
    for first, second in zip(requests[::2], requests[1::2]):
        assert first[1] == second[1] and first[0] != second[0]
    assert [path for path, _ in requests[:2]] == ["/v1/decisions", "/v1/responses"]
    assert [path for path, _ in requests[14:16]] == ["/v1/responses", "/v1/decisions"]
    assert all(row["duration_ms"] is None for row in result["rows"] if not row["provider_called"])
    assert bench.summarize(fixture, result)["modes"][2]["provider_correct"] == 14
    if prospective:
        result_path = tmp_path / "new-results.json"
        bench.save(result_path, result)
        assert bench.main(["report", "--fixture", str(V2_PATH), "--results", str(result_path)]) == 0
        assert json.loads(capsys.readouterr().out)["fixture_sha256"] == V2_HASH


def test_prospective_fixture_preserves_controls_and_exact_new_input_hashes(fixture):
    v2 = bench.load_fixture(V2_PATH)
    assert fixture["fixture_sha256"] == "afbac046436291e39519756023e7aecc2816aa19a63da2511a1d51081aff1b2a"
    assert v2["fixture_sha256"] == V2_HASH and v2["recorded_results"] is None
    assert v2["scope"] == fixture["scope"]
    assert [row["id"] for row in v2["candidates"]] == [row["id"] for row in fixture["candidates"]]
    controls = ("id", "question", "expected_id", "lexical_selected_id", "semantic_invoked",
                "requires_zero_anchor_fallback", "candidate_order")
    for previous, current, input_hash in zip(fixture["cases"], v2["cases"], V2_INPUT_HASHES, strict=True):
        assert {key: previous[key] for key in controls} == {key: current[key] for key in controls}
        assert current["input_sha256"] == input_hash
        if not current["semantic_invoked"]:
            continue
        data = bench.payload(v2, current)
        shared, _ = TableSelectionService._input(**data)
        assert hashlib.sha256(shared.encode()).hexdigest() == input_hash
        assert len(shared) <= 40_000 and len(data["candidates"]) == 96
        previews = [json.loads(row["description"]) for row in data["candidates"]]
        assert all(row["preview"] and row["not_citable"] and row["examples"] for row in previews)
        material = next(row["description"] for row in data["candidates"] if row["id"] == current["expected_id"]) if current["expected_id"] else None
        if material:
            assert "Elastic" in material and "Tresca" in material
        assert any("Geogrid" in row["description"] and "Truss" in row["description"] for row in data["candidates"])
    assert len(v2["cases"][4]["description_overrides"]) == 33


@pytest.mark.parametrize("change", ["out_of_set", "oversized", "missing_override"])
def test_prospective_description_overrides_are_bound_and_validated(change):
    fixture = bench.load_fixture(V2_PATH)
    case = fixture["cases"][4]
    if change == "out_of_set":
        case["description_overrides"]["invented"] = "preview"
    elif change == "oversized":
        case["description_overrides"][next(iter(case["description_overrides"]))] = "x" * 1501
    else:
        case["description_overrides"].pop(next(iter(case["description_overrides"])))
    reseal(fixture)
    with pytest.raises(bench.BenchmarkError):
        bench.validate_fixture(fixture)


def test_prospective_report_requires_new_results_and_refuses_original_observations(fixture, tmp_path, capsys):
    assert bench.main(["validate", "--fixture", str(V2_PATH)]) == 0
    assert json.loads(capsys.readouterr().out)["fixture_sha256"] == V2_HASH
    assert bench.main(["report", "--fixture", str(V2_PATH)]) == 1
    assert json.loads(capsys.readouterr().out) == {"error_type": "BenchmarkError"}
    with pytest.raises(bench.BenchmarkError, match="ResultsRequired"):
        bench.summarize(bench.load_fixture(V2_PATH), None)
    old_results = deepcopy(fixture["recorded_results"])
    old_results["fixture_sha256"] = fixture["fixture_sha256"]
    result_path = tmp_path / "original-results.json"
    bench.save(result_path, old_results)
    assert bench.main(["report", "--fixture", str(V2_PATH), "--results", str(result_path)]) == 1
    assert json.loads(capsys.readouterr().out) == {"error_type": "BenchmarkError"}


@pytest.mark.parametrize("failure", ["cache", "hash", "selection", "exception"])
def test_endpoint_contract_failure_always_closes_fresh_service(fixture, failure):
    closed = []

    class Service:
        def __init__(self, **_kwargs):
            pass

        def select(self, question, candidates, diagnostics):
            diagnostics.update(input_sha256=bench.digest({"question": question, "candidates": candidates}),
                               cache="miss", error_type=None, duration_ms=1.)
            if failure == "cache":
                diagnostics["cache"] = "hit"
            elif failure == "hash":
                diagnostics["input_sha256"] = "wrong"
            elif failure == "selection":
                return "invented"
            else:
                raise RuntimeError("provider detail must not be printed")
            return None

        def close(self):
            closed.append(True)

    with pytest.raises((bench.BenchmarkError, RuntimeError)):
        bench.run_benchmark(fixture, model="mock-model", base_url="https://provider.invalid/v1",
                            api_key="test-only", service_factory=Service)
    assert closed == [True]


def test_timeout_none_is_never_a_successful_negative(fixture):
    class Service:
        def __init__(self, **_kwargs):
            pass

        def select(self, question, candidates, diagnostics):
            diagnostics.update(input_sha256=bench.digest({"question": question, "candidates": candidates}),
                               cache="miss", error_type="ReadTimeout", duration_ms=1.)

        def close(self):
            pass

    result = bench.run_benchmark(fixture, model="mock-model", base_url="https://provider.invalid/v1",
                                 api_key="test-only", service_factory=Service)
    assert all(row["correct_choice"] is None for row in result["rows"] if row["provider_called"])
    modes = {row["mode"]: row for row in bench.summarize(fixture, result)["modes"]}
    assert modes["decisions"]["errors"] == modes["responses"]["errors"] == 14
    assert modes["decisions"]["correct"] == modes["responses"]["correct"] == 2


def test_cli_offline_default_and_live_gate_do_not_construct_services(monkeypatch, capsys):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("No calls permitted")
    monkeypatch.setattr(bench, "run_benchmark", forbidden)
    assert bench.main(["validate"]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]
    assert bench.main(["report"]) == 0
    assert json.loads(capsys.readouterr().out)["source_choice_only"]
    assert bench.main(["run"]) == 1
    assert json.loads(capsys.readouterr().out) == {"error_type": "BenchmarkError"}


def test_cli_errors_never_print_credentials_or_provider_messages(tmp_path, monkeypatch, capsys):
    sentinel = "test-only-must-never-be-printed"
    monkeypatch.setenv("UNIT_ROUTING_KEY", sentinel)

    def fail(*_args, **_kwargs):
        raise RuntimeError(sentinel)

    monkeypatch.setattr(bench, "run_benchmark", fail)
    assert bench.main(["run", "--allow-paid-calls", "--model", "mock-model", "--api-key-env", "UNIT_ROUTING_KEY",
                       "--output", str(tmp_path / "results.json")]) == 1
    output = capsys.readouterr().out
    assert sentinel not in output and json.loads(output) == {"error_type": "RuntimeError"}
