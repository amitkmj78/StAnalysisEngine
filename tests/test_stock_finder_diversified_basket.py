from unittest.mock import patch

import pandas as pd

from services.stock_finder_service import build_diversified_basket


def _ranked_df():
    # 3 sectors, 4 tickers each, already sorted by Score descending within
    # and across sectors -- matches rank_stocks' own documented contract.
    rows = []
    for sector, base_score in [("Tech", 90), ("Health", 80), ("Energy", 70)]:
        for i in range(4):
            rows.append(
                {
                    "Ticker": f"{sector[:2].upper()}{i}",
                    "Name": f"{sector} Co {i}",
                    "Sector": sector,
                    "Price": 100.0 + i,
                    "Score": base_score - i,
                }
            )
    return pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)


def test_no_cap_returns_picks_per_sector_times_sector_count():
    with patch("services.stock_finder_service.rank_stocks", return_value=_ranked_df()):
        basket = build_diversified_basket("Long Term", "All", picks_per_sector=2)
    assert len(basket) == 6  # 2 sectors picks x 3 sectors
    assert set(basket["Sector"]) == {"Tech", "Health", "Energy"}


def test_max_stocks_caps_total_while_preserving_sector_spread():
    with patch("services.stock_finder_service.rank_stocks", return_value=_ranked_df()):
        basket = build_diversified_basket("Long Term", "All", picks_per_sector=4, max_stocks=6)
    assert len(basket) == 6
    # Round-robin over 3 sectors for 6 slots -> exactly 2 per sector, not
    # a flat top-6-by-score that could all land in Tech.
    counts = basket["Sector"].value_counts().to_dict()
    assert counts == {"Tech": 2, "Health": 2, "Energy": 2}


def test_max_stocks_keeps_best_scorers_per_sector():
    with patch("services.stock_finder_service.rank_stocks", return_value=_ranked_df()):
        basket = build_diversified_basket("Long Term", "All", picks_per_sector=4, max_stocks=3)
    assert len(basket) == 3
    # One per sector, and it must be each sector's top scorer (index 0 in
    # that sector's own ranking, i.e. TE0/HE0/EN0-style top ticker).
    for sector in ["Tech", "Health", "Energy"]:
        sector_rows = basket[basket["Sector"] == sector]
        assert len(sector_rows) == 1
        assert sector_rows.iloc[0]["Score"] == _ranked_df()[_ranked_df()["Sector"] == sector]["Score"].max()


def test_max_stocks_larger_than_basket_is_a_no_op():
    with patch("services.stock_finder_service.rank_stocks", return_value=_ranked_df()):
        basket = build_diversified_basket("Long Term", "All", picks_per_sector=2, max_stocks=50)
    assert len(basket) == 6


def test_max_stocks_none_or_zero_does_not_trim():
    with patch("services.stock_finder_service.rank_stocks", return_value=_ranked_df()):
        basket_none = build_diversified_basket("Long Term", "All", picks_per_sector=2, max_stocks=None)
        basket_zero = build_diversified_basket("Long Term", "All", picks_per_sector=2, max_stocks=0)
    assert len(basket_none) == 6
    assert len(basket_zero) == 6
