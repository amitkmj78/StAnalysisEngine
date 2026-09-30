import numpy as np
import pandas as pd

from services.market_internals_service import compute_internals_score
from services.market_regime_service import REGIME_GATE_DISCLOSURE, _compute_regime_frame

_PERSIST_COLUMNS = [
    "as_of_date", "internals_score", "mds", "regime_raw", "regime_confirmed",
    "data_completeness", "conflict_flag",
    "breadth_50dma", "vix", "vix3m", "xly_xlp", "hyg_ief", "rsp_spy",
]


def _dates(n, start="2020-01-01"):
    return pd.bdate_range(start, periods=n)


def _flat_internals(n, breadth=50.0, vix=18.0, vix3m=19.0, xly_xlp=1.0, hyg_ief=1.0, rsp_spy=1.0):
    idx = _dates(n)
    return pd.DataFrame(
        {
            "breadth_50dma": breadth,
            "vix": vix,
            "vix3m": vix3m,
            "xly_xlp": xly_xlp,
            "hyg_ief": hyg_ief,
            "rsp_spy": rsp_spy,
        },
        index=idx,
    )


# Regression guard: this is the crux fact the whole feature's disclosure
# exists to convey — it must never be softened, paraphrased away, or
# dropped by a future edit to market_regime_service.py.
def test_regime_gate_disclosure_contains_the_gfc_finding():
    assert "-7.91" in REGIME_GATE_DISCLOSURE
    assert "2008" in REGIME_GATE_DISCLOSURE or "2007" in REGIME_GATE_DISCLOSURE
    assert "gate" in REGIME_GATE_DISCLOSURE.lower()


def test_compute_regime_frame_skips_warmup_rows_entirely():
    # Fewer rows than compute_internals_score's z-score window -> every
    # value is NaN -> nothing scoreable, no bogus rows written.
    df = _flat_internals(100)
    frame = _compute_regime_frame(df)
    assert frame.empty
    assert list(frame.columns) == _PERSIST_COLUMNS


def test_compute_regime_frame_produces_one_row_per_scoreable_date():
    n = 300
    rng = np.random.default_rng(5)
    df = _flat_internals(n)
    df["breadth_50dma"] = 50 + rng.normal(0, 3, n)
    df.index = _dates(n)

    expected_scoreable = int(compute_internals_score(df).notna().sum())
    frame = _compute_regime_frame(df)

    assert expected_scoreable > 0
    assert len(frame) == expected_scoreable
    assert list(frame.columns) == _PERSIST_COLUMNS
    assert frame["regime_raw"].notna().all()
    assert frame["regime_confirmed"].notna().all()
    # Raw component columns are the real same-day inputs, not derived —
    # vix was never varied in this fixture.
    assert (frame["vix"] == 18.0).all()


def test_compute_regime_frame_internals_only_mds_equals_internals_score():
    # With only the internals pillar ever populated (P1, forever, until
    # P2/P3 exist), compute_composite_score's weighted average over one
    # present pillar reduces to that pillar's own value exactly.
    n = 300
    rng = np.random.default_rng(6)
    df = _flat_internals(n)
    df["breadth_50dma"] = 50 + rng.normal(0, 3, n)
    df.index = _dates(n)

    frame = _compute_regime_frame(df)
    # mds is rounded to 2dp inside compute_composite_score itself;
    # internals_score isn't rounded at all -- compare at that precision.
    assert (frame["mds"].round(2) == frame["internals_score"].round(2)).all()
    # And data_completeness is the permanent 1/3 this phase implies — see
    # services/market_internals_service.py::compute_composite_score.
    assert (frame["data_completeness"].round(4) == round(1 / 3, 4)).all()


def test_compute_regime_frame_applies_hysteresis_across_the_whole_series():
    n = 280
    rng = np.random.default_rng(9)
    df = _flat_internals(n)
    df["breadth_50dma"] = 50 + rng.normal(0, 2, n)
    df.index = _dates(n)
    df.iloc[-1, df.columns.get_loc("breadth_50dma")] = 99.0  # one extreme, noisy day

    frame = _compute_regime_frame(df)
    assert not frame.empty
    # SR-5: a single noisy day can't flip the *confirmed* regime on its
    # own, even if that day's *raw* label differs from the day before —
    # apply_hysteresis itself is already tested directly in
    # test_market_internals_service.py; this confirms the orchestration
    # function actually applies it across the whole backfilled series
    # rather than per-day in isolation.
    assert frame["regime_confirmed"].iloc[-1] == frame["regime_confirmed"].iloc[-2]
