from unittest.mock import patch

import pandas as pd

from services.momentum_backtest_service import _download_universe_history


def _fake_frame(ticker: str, n_rows: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n_rows)
    return pd.DataFrame({"Close": range(n_rows), "Volume": range(n_rows)}, index=idx)


def test_splits_into_batches_of_the_configured_size():
    tickers = [f"T{i}" for i in range(120)]  # 3 batches at the default size of 50
    calls = []

    def fake_download(chunk, **kwargs):
        calls.append(list(chunk))
        return {t: _fake_frame(t) for t in chunk}

    with patch("services.momentum_backtest_service.yf.download", side_effect=fake_download), patch(
        "services.momentum_backtest_service.time.sleep"
    ):
        frames = _download_universe_history(tickers, period="4y")

    assert len(calls) == 3
    assert [len(c) for c in calls] == [50, 50, 20]
    assert len(frames) == 120


def test_a_failed_batch_is_skipped_not_fatal():
    tickers = [f"T{i}" for i in range(60)]  # 2 batches

    def fake_download(chunk, **kwargs):
        if chunk[0] == "T0":
            raise Exception("rate limited")
        return {t: _fake_frame(t) for t in chunk}

    with patch("services.momentum_backtest_service.yf.download", side_effect=fake_download), patch(
        "services.momentum_backtest_service.time.sleep"
    ):
        frames = _download_universe_history(tickers, period="4y")

    # First batch (T0-T49) failed entirely (even after fetch_with_backoff's
    # own retries); second batch (T50-T59) succeeded.
    assert len(frames) == 10
    assert "T50" in frames
    assert "T0" not in frames


def test_pauses_between_batches_but_not_after_the_last_one():
    tickers = [f"T{i}" for i in range(100)]  # exactly 2 batches

    def fake_download(chunk, **kwargs):
        return {t: _fake_frame(t) for t in chunk}

    # fetch_with_backoff has its own internal pacing sleep -- bypassed
    # here (call the wrapped fn directly) so this test isolates just the
    # batch-pause sleep this function adds itself.
    with patch("services.momentum_backtest_service.yf.download", side_effect=fake_download), patch(
        "services.momentum_backtest_service.fetch_with_backoff", side_effect=lambda fn: fn()
    ), patch("services.momentum_backtest_service.time.sleep") as mock_sleep:
        _download_universe_history(tickers, period="4y")

    # 2 batches -> exactly 1 pause between them, none trailing.
    assert mock_sleep.call_count == 1


def test_single_ticker_final_batch_indexes_frame_directly():
    """When the last batch has exactly 1 ticker, yf.download's own return
    shape for a single ticker isn't dict-like/indexable by that ticker --
    the frame itself IS that ticker's data (same quirk the old
    single-bulk-call code already handled via `len(tickers) > 1`)."""
    tickers = [f"T{i}" for i in range(51)]  # batch of 50, then a lone T50

    def fake_download(chunk, **kwargs):
        if len(chunk) == 1:
            return _fake_frame(chunk[0])
        return {t: _fake_frame(t) for t in chunk}

    with patch("services.momentum_backtest_service.yf.download", side_effect=fake_download), patch(
        "services.momentum_backtest_service.time.sleep"
    ):
        frames = _download_universe_history(tickers, period="4y")

    assert len(frames) == 51
    assert "T50" in frames
