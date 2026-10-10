"""STS-4: the leaderboard's pure daily-return transform, and the full
build_strategy_leaderboard round trip against a fake DB connection and a
synthetic SPY series -- same fake-connection pattern as
tests/test_community_ideas_eval.py.
"""

import asyncio
from datetime import date, datetime, timedelta

import pandas as pd

import services.strategy_leaderboard as leaderboard_mod
from services.strategy_leaderboard import MIN_LEADERBOARD_DAYS, MIN_LEADERBOARD_TRADES, _daily_returns_from_cumulative


def test_daily_returns_from_cumulative_first_day_is_its_own_cumulative_value():
    assert _daily_returns_from_cumulative([5.0]) == [5.0]


def test_daily_returns_from_cumulative_day_over_day_change():
    # Day 0: +10% (equity 1.10). Day 1: cumulative +21% (equity 1.21) --
    # a further +10% day-over-day, i.e. 1.21/1.10 - 1 = 10%.
    out = _daily_returns_from_cumulative([10.0, 21.0])
    assert out[0] == 10.0
    assert round(out[1], 6) == 10.0


def test_daily_returns_from_cumulative_empty_input_is_empty_output():
    assert _daily_returns_from_cumulative([]) == []


class _FakeConn:
    def __init__(self, strategies, snapshots_by_id):
        self._strategies = strategies
        self._snapshots_by_id = snapshots_by_id

    async def fetch(self, sql, *args):
        if "FROM published_strategy_forward_snapshots" in sql:
            return self._snapshots_by_id.get(args[0], [])
        assert "FROM published_strategies p" in sql
        return self._strategies


class _FakeConnCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *a):
        return False


def _snapshot_rows(n_days: int, final_trades: int):
    rows = []
    for i in range(n_days):
        rows.append({
            "as_of_date": date.today() - timedelta(days=n_days - i),
            "cumulative_return_pct": round(0.2 * (i + 1), 2),
            "trades": max(1, round(final_trades * (i + 1) / n_days)),
        })
    rows[-1]["trades"] = final_trades
    return rows


def _flat_spy_history(days: int):
    idx = pd.date_range(date.today() - timedelta(days=days + 5), periods=days + 10, freq="D")
    close = pd.Series([500.0 + 0.1 * i for i in range(len(idx))], index=idx)
    return pd.DataFrame({"Close": close})


def test_leaderboard_ranks_eligible_above_not_enough_data(monkeypatch):
    eligible = {
        "id": 1, "name": "Momentum v1", "display_name": "alice",
        "published_at": datetime.now() - timedelta(days=MIN_LEADERBOARD_DAYS + 10),
    }
    ineligible = {
        "id": 2, "name": "Too New", "display_name": "bob",
        "published_at": datetime.now() - timedelta(days=10),
    }
    snapshots_by_id = {
        1: _snapshot_rows(n_days=95, final_trades=MIN_LEADERBOARD_TRADES + 5),
        2: _snapshot_rows(n_days=5, final_trades=3),
    }
    conn = _FakeConn([ineligible, eligible], snapshots_by_id)
    monkeypatch.setattr(leaderboard_mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(leaderboard_mod, "get_cached_history", lambda *a, **k: _flat_spy_history(100))

    result = asyncio.run(leaderboard_mod.build_strategy_leaderboard())

    assert [r["id"] for r in result] == [1, 2]
    assert result[0]["eligible"] is True
    assert result[0]["risk_adjusted_excess_return"] is not None
    assert result[1]["eligible"] is False
    assert result[1]["reason"] == "not enough data yet"
    assert result[1]["risk_adjusted_excess_return"] is None


def test_leaderboard_gates_on_trades_even_with_enough_elapsed_days(monkeypatch):
    strategy = {
        "id": 3, "name": "Few Trades", "display_name": None,
        "published_at": datetime.now() - timedelta(days=MIN_LEADERBOARD_DAYS + 30),
    }
    snapshots_by_id = {3: _snapshot_rows(n_days=100, final_trades=MIN_LEADERBOARD_TRADES - 1)}
    conn = _FakeConn([strategy], snapshots_by_id)
    monkeypatch.setattr(leaderboard_mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(leaderboard_mod, "get_cached_history", lambda *a, **k: _flat_spy_history(110))

    result = asyncio.run(leaderboard_mod.build_strategy_leaderboard())

    assert result[0]["eligible"] is False
    assert result[0]["reason"] == "not enough data yet"
