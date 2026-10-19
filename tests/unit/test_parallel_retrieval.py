import time
from threading import Barrier, Lock

from app.services.parallel_retrieval import independent_searches


def test_queries_run_together_but_evidence_order_is_stable():
    barrier = Barrier(2, timeout=2)
    active = 0
    peak = 0
    lock = Lock()

    def search(query):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait()
        with lock:
            active -= 1
        return [query]

    result = independent_searches(["first", "second"], search)
    assert peak == 2
    assert [row.items for row in result] == [["first"], ["second"]]
    assert all(row.error_type is None for row in result)


def test_failed_search_does_not_erase_successful_sibling_or_expose_exception():
    def search(query):
        if query == "fail":
            raise RuntimeError("sensitive provider response")
        return ["real evidence"]

    result = independent_searches(["fail", "works"], search)
    assert result[0].error_type == "RuntimeError" and result[0].items == []
    assert result[1].items == ["real evidence"]
    assert "sensitive" not in repr(result)


def test_expired_deadline_starts_no_provider_calls():
    calls = []
    result = independent_searches(["one", "two"], lambda query: calls.append(query), deadline=time.perf_counter()-1)
    assert calls == []
    assert all(row.error_type == "TimeoutError" for row in result)
