from datetime import date

from web.backend.routers.trading_agent import _worst_month


def _snap(day: date, equity: float) -> dict:
    return {"as_of_date": day, "equity": equity}


def test_worst_month_none_with_fewer_than_two_finished_months():
    # Only this month's own (in-progress) data -- nothing finished yet.
    today = date.today()
    snaps = [_snap(today.replace(day=1), 100_000.0), _snap(today, 99_000.0)]
    assert _worst_month(snaps) is None


def test_worst_month_picks_the_largest_loss_among_finished_months():
    snaps = [
        _snap(date(2026, 7, 1), 100_000.0), _snap(date(2026, 7, 31), 103_000.0),   # July: +3%
        _snap(date(2026, 8, 1), 103_000.0), _snap(date(2026, 8, 31), 93_000.0),    # August: -9.7%
        _snap(date(2026, 9, 1), 93_000.0), _snap(date(2026, 9, 30), 95_000.0),     # September: +2.2%
    ]
    worst = _worst_month(snaps)
    assert worst is not None
    assert worst["month"] == "2026-08"
    assert worst["return_pct"] < -9.0


def test_worst_month_ignores_a_single_snapshot_month():
    # A month with only one snapshot has no start/end to compute a return from.
    snaps = [
        _snap(date(2026, 7, 1), 100_000.0), _snap(date(2026, 7, 31), 90_000.0),  # July: -10%
        _snap(date(2026, 8, 15), 85_000.0),  # August: only one snapshot, skipped
    ]
    worst = _worst_month(snaps)
    assert worst["month"] == "2026-07"
