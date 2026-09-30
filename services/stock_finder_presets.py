"""
SCN-4: a small, hardcoded set of preset screens the frontend can load with
one click. Deliberately NOT rows in the saved_screens table -- that table
is for user-owned, user-named, deletable screens; presets are none of
those, and shipping them as code means they can't be accidentally edited
or deleted by a user.

Each `filters` dict uses the exact same keys as the Stock Finder page's own
FilterState (web/frontend/app/stock-finder/page.tsx) and saved_screens.filters
column, so a preset loads through the identical "apply these filters" path
a saved screen already uses -- no separate preset-filtering logic anywhere.
"""

PRESET_SCREENS: list[dict] = [
    {
        "key": "quality_fair_price",
        "name": "High quality at a fair price",
        "rules": (
            "Forward P/E of 20 or under, a Long-Term Score of 50 or higher, "
            "and a Long-Term Signal of Buy or Hold (never Trim)."
        ),
        "goal": "Long Term",
        "universe": "All",
        "filters": {
            "forwardPeMax": "20",
            "longScoreMin": "50",
            "longSignal": ["Buy", "Hold"],
        },
    },
    {
        "key": "rising_estimates",
        "name": "Rising estimates",
        "rules": (
            "Earnings growth of 10% or higher and a Short-Term Signal of Buy -- "
            "names the market is actively upgrading, not just cheap ones."
        ),
        "goal": "Short Term",
        "universe": "All",
        "filters": {
            "earningsGrowthMin": "10",
            "shortSignal": ["Buy"],
        },
    },
    {
        "key": "low_vol_dividend",
        "name": "Low volatility dividend",
        "rules": (
            "6-month volatility of 20% or under and a dividend yield of at least "
            "2% -- steadier names that also pay you to hold them."
        ),
        "goal": "Long Term",
        "universe": "All",
        "filters": {
            "volatilityMax": "20",
            "dividendYieldMin": "2",
        },
    },
]
