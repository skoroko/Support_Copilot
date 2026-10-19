from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from app.eval.acceptance_budget import (
    BudgetExceeded,
    BudgetLedger,
    metered_create,
    upper_cost,
    usage_cost,
)


def test_persistent_unknown_request_consumes_its_full_reservation(tmp_path):
    path = tmp_path / "budget.sqlite3"
    ledger = BudgetLedger(path, limit_usd=1)
    ident = ledger.reserve(kind="generation", model="gpt-6-luna", amount=.8)
    ledger.settle(ident, error=True)
    resumed = BudgetLedger(path, limit_usd=1)
    with pytest.raises(BudgetExceeded):
        resumed.reserve(kind="review", model="gpt-6.1-sol", amount=.3)
    assert resumed.summary()["unknown_cost_calls"] == 1
    assert resumed.summary()["remaining_usd"] == .2
    with pytest.raises(ValueError, match="implicitly"):
        BudgetLedger(path, limit_usd=2)


def test_concurrent_reservations_cannot_overspend(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.sqlite3", limit_usd=1)
    def reserve(_):
        try:
            ledger.reserve(kind="embedding", model="text-embedding-3-large", amount=.1)
            return True
        except BudgetExceeded:
            return False
    with ThreadPoolExecutor(max_workers=8) as workers:
        assert sum(workers.map(reserve, range(40))) == 10
    assert ledger.summary()["accounted_upper_usd"] == 1


def test_usage_reasoning_not_double_counted_and_cache_write_reserved():
    usage = {"input_tokens": 1000, "output_tokens": 100,
             "input_token_details": {"cache_read": 200, "cache_creation": 100},
             "output_token_details": {"reasoning": 80}}
    assert usage_cost("gpt-6-luna", usage) == pytest.approx(.0001345)
    assert upper_cost("gpt-6-luna", 1000, 100) == pytest.approx(.000175)
    with pytest.raises(ValueError):
        usage_cost("gpt-6-luna", {})


def test_request_rejected_before_provider_and_failure_not_charged_zero(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.sqlite3", limit_usd=.001)
    calls = []
    def provider(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("secret-like exception must not be persisted")
    with pytest.raises(BudgetExceeded):
        metered_create(provider, ledger, kind="generation", model="gpt-6.1-sol", input="x", max_output_tokens=1000)
    assert not calls
    with pytest.raises(RuntimeError):
        metered_create(provider, ledger, kind="generation", model="gpt-6-luna", input="x", max_output_tokens=100)
    assert ledger.summary()["unknown_cost_calls"] == 1


def test_success_settles_provider_usage_and_duplicate_settle_rejected(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.sqlite3", limit_usd=1)
    def provider(**kwargs):
        return SimpleNamespace(usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 10, "output_tokens": 20}))
    metered_create(provider, ledger, kind="review", model="gpt-6.1-sol", input="x", max_output_tokens=30)
    assert ledger.summary()["accounted_upper_usd"] == .00022
    ident = ledger.reserve(kind="review", model="gpt-6.1-sol", amount=.1)
    ledger.settle(ident, usage={"input_tokens": 0, "output_tokens": 0})
    with pytest.raises(ValueError, match="already"):
        ledger.settle(ident, error=True)
