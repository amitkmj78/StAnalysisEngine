import pandas as pd

from services import sp500_membership as m


def _fake(monkeypatch):
    changes = pd.DataFrame({
        "date": pd.to_datetime(["2021-12-20", "2025-09-22", "2026-03-01"]),
        "added": ["HOOD", "NEWCO", None],
        "removed": [None, "OLDCO", "GONE"],
    })
    monkeypatch.setattr(m, "fetch_changes", lambda: changes)
    monkeypatch.setattr(m, "current_members", lambda: frozenset({"AAPL", "HOOD", "NEWCO"}))


def test_a_later_addition_is_excluded_before_its_add_date(monkeypatch):
    _fake(monkeypatch)
    assert "HOOD" not in m.members_on("2021-10-08")
    assert "HOOD" in m.members_on("2021-12-31")
    assert "HOOD" in m.members_on(pd.Timestamp.today())


def test_a_removed_stock_is_a_member_before_it_left(monkeypatch):
    _fake(monkeypatch)
    before = m.members_on("2021-10-08")
    assert "OLDCO" in before and "GONE" in before
    assert "OLDCO" not in m.members_on("2025-12-31")


def test_removals_after_the_start_date_are_listed(monkeypatch):
    _fake(monkeypatch)
    assert m.removed_after("2021-10-08") == ["GONE", "OLDCO"]


def test_member_flags_follow_the_changes_day_by_day(monkeypatch):
    _fake(monkeypatch)
    index = pd.bdate_range("2025-09-18", periods=8)
    flags = m.member_flags("NEWCO", index, start_members=set())
    assert not flags.iloc[0] and flags.iloc[-1]


def test_blank_cells_in_the_changes_table_do_not_become_members(monkeypatch):
    import math
    changes = pd.DataFrame({
        "date": pd.to_datetime(["2021-12-20", "2025-09-22"]),
        "added": [float("nan"), "NEWCO"],
        "removed": ["OLDCO", float("nan")],
    })
    monkeypatch.setattr(m, "fetch_changes", lambda: changes)
    monkeypatch.setattr(m, "current_members", lambda: frozenset({"AAPL", "NEWCO"}))
    members = m.members_on("2021-10-08")
    assert all(isinstance(x, str) for x in members)
    assert not any(isinstance(x, float) and math.isnan(x) for x in members)
    assert sorted(members)  # sortable: this is the call that failed in production
    assert all(isinstance(x, str) for x in m.removed_after("2021-10-08"))
