"""
SCN-4: PRESET_SCREENS is a static list, not user data -- these tests guard
against a preset silently referencing a filter key that was renamed or
removed from the frontend's FilterState, or from ALLOWED_GOALS.
"""

from services.stock_finder_presets import PRESET_SCREENS

VALID_FILTER_KEYS = {
    "marketCapMin", "marketCapMax",
    "forwardPeMin", "forwardPeMax",
    "volumeStrengthMin",
    "sectors",
    "dividendYieldMin",
    "volatilityMax",
    "momentumMin", "momentumMax",
    "earningsGrowthMin", "earningsGrowthMax",
    "shortScoreMin", "shortScoreMax",
    "longScoreMin", "longScoreMax",
    "shortSignal", "longSignal",
    "owned", "watchlisted",
}
ALLOWED_GOALS = {"Short Term", "Long Term"}


def test_exactly_three_presets():
    assert len(PRESET_SCREENS) == 3


def test_every_preset_has_a_non_empty_rules_explanation():
    for preset in PRESET_SCREENS:
        assert isinstance(preset["rules"], str) and preset["rules"].strip()


def test_every_preset_filter_key_is_a_valid_filter_state_key():
    for preset in PRESET_SCREENS:
        assert set(preset["filters"].keys()) <= VALID_FILTER_KEYS, preset["key"]


def test_every_preset_goal_is_valid():
    for preset in PRESET_SCREENS:
        assert preset["goal"] in ALLOWED_GOALS


def test_preset_keys_are_unique():
    keys = [p["key"] for p in PRESET_SCREENS]
    assert len(keys) == len(set(keys))
