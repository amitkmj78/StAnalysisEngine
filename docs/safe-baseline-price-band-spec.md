# Safe Baseline Price Band — Requirements Specification

**Feature:** Safe Baseline (Two-Sided Excursion Band)
**Epic:** Quant Model — Risk & Entry Planning
**Version:** 0.2 (Draft) **Supersedes:** v0.1 **Status:** Draft for review
**Target release:** TBD
**Owner:** Amit Kumar Maharaj

---

## Changes from v0.1

v0.1 was reviewed against the existing codebase before any implementation began.
The findings below are incorporated; requirement IDs are preserved for
traceability, with `a`-suffixed IDs added where a v0.1 requirement needed
splitting rather than replacing.

| Change | Rationale |
|---|---|
| Renamed `fair` → `median_path`, `buy_zone_hi`/`buy_zone_lo` → `accumulation_zone_hi`/`accumulation_zone_lo`, `sell_zone_lo`/`sell_zone_hi` → `distribution_zone_lo`/`distribution_zone_hi` (API fields, UI labels, AC-01.1) | DR-03 explicitly bars presenting any level as a recommendation or fair-value estimate — the v0.1 field/label names did exactly that, independent of any disclaimer shown elsewhere on the page |
| Clarified winsorization (FR-09) applies to the **aggregated H-day MAE/MFE distribution**, not to daily excursions prior to aggregation | Winsorizing at the daily level first can clip the single worst day inside a window — exactly the day that legitimately determines that window's MAE. That distorts the tail behavior a floor/ceiling exists to capture. Winsorizing after aggregation guards against a handful of outlier *windows* instead, which is the intended protection |
| Added **FR-04a** — a minimum independent-window count, distinct from FR-04's raw bar-count floor | FR-04's `H + 2` bars is a technical minimum, not a statistical one: at H = 90 it yields exactly one non-overlapping window, which cannot support any confidence statement even though FR-04 is satisfied |
| Added **DR-06** — relationship to existing Predict confidence intervals and Entry Signals stop/target | The app already shows two other "how far might price move" numbers for the same ticker (a residual-based forecast CI on `/predict`, an ATR-heuristic stop/target on Entry Signals). Without an explicit note, a user can reasonably conclude the app disagrees with itself |
| Resolved Open Questions 1, 2, 4 (see §9) | Each has a clear answer once checked against how the rest of the app already works — no need to leave them open |
| Added a parallel/batched OHLC fetch capability as an explicit dependency (§10) | No existing code path in this app fetches multiple tickers' history concurrently; every current multi-ticker feature loops sequentially. NFR-02 cannot be met without building this |
| NFR-02 reworded to state its warm-cache assumption explicitly | The original wording didn't say whether the 5-second budget included a cold-cache fetch. It doesn't — a cold 50-ticker fetch is not achievable at that latency with any fetch pattern that exists in the codebase today |

---

## 1. Purpose

Close-to-close returns describe where a security ended, not the path it took to
get there. A position can finish flat having traded 6% lower intraday. Users
setting limit orders, stops, or accumulation zones need the path.

The Safe Baseline feature derives a five-level price band for any ticker from
the historical distribution of intraday excursions — the maximum dip below and
maximum rise above a reference price — aggregated across a forward horizon.

### 1.1 Relationship to existing features

Two other features already answer an adjacent question for the same ticker,
by different methods, and this spec does not merge or replace either:

- **Predict's confidence interval** (`services/prediction_service.py`) is a
  ±1.96σ band around a *point forecast*, where σ comes from in-sample
  residuals of the prediction model — it answers "how wrong has this specific
  forecast been historically," not "how far has price actually traveled."
- **Entry Signals' stop-loss / first-target** (`services/entry_strategy_service.py`)
  is a rule-based heuristic — 20-day rolling support/resistance and ATR, with
  a fixed 2:1 reward/risk multiple — not derived from a realized excursion
  distribution at all.

The Safe Baseline band is the only one of the three built directly from
realized MAE/MFE. See **DR-06**: the UI must not let a user infer these three
numbers should agree.

---

## 2. Definitions

| Term | Definition |
|---|---|
| **Excursion** | Log travel from a reference price to an intraday extreme within a bar |
| **Dip** | `log(low / prior_close)` — downside travel, gap-inclusive |
| **Rise** | `log(high / prior_close)` — upside travel, gap-inclusive |
| **MAE** | Maximum Adverse Excursion: deepest point below entry across a forward window |
| **MFE** | Maximum Favorable Excursion: highest point above entry across a forward window |
| **Horizon (H)** | Forward trading days evaluated (10 / 30 / 60 / 90) |
| **Confidence** | Probability the band is expected to contain price over H |
| **Floor** | Price below which the security did not trade in `confidence` of historical H-day windows |
| **Breach rate** | Observed frequency of price violating a computed floor |
| **First-touch** | Which of two barriers (stop or target) was reached first on a path |

---

## 3. User Stories

### US-01 — View the safe baseline band
**As a** retail investor evaluating a position
**I want** to see a price band showing how far this security typically dips and rises over my holding period
**So that** I can set an entry price I am comfortable owning at rather than guessing.

**Acceptance criteria**
- **AC-01.1** Given a ticker with sufficient history, when I open the ticker detail page, then a five-level ladder displays: ceiling, distribution zone, median path, accumulation zone, and floor.
- **AC-01.2** Each level displays as both an absolute price and a percentage from last close.
- **AC-01.3** The band reflects the currently selected horizon and confidence level.
- **AC-01.4** The as-of date of the underlying price data is visible on the component.

### US-02 — Adjust horizon and confidence
**As a** user with a specific holding period in mind
**I want** to switch the band between 10, 30, 60, and 90 day horizons and adjust confidence
**So that** the levels match how long I actually intend to hold.

**Acceptance criteria**
- **AC-02.1** Horizon selector offers 10 / 30 / 60 / 90 trading days; default 30.
- **AC-02.2** Confidence selector offers 75% / 90% / 95%; default 90%.
- **AC-02.3** Changing either control recalculates and re-renders the band without a full page reload.
- **AC-02.4** Selections persist for the user's session.

### US-03 — See limit-order fill probability
**As a** user placing a limit buy
**I want** to know how often an order at a given depth would have filled within my horizon
**So that** I can trade off a better price against the risk of never getting filled.

**Acceptance criteria**
- **AC-03.1** A fill-probability curve displays for depths of 1%, 2%, 3%, 5%, 7.5%, 10%, 15%, and 20% below last close.
- **AC-03.2** Each depth shows the percentage of historical H-day windows in which that level was touched.
- **AC-03.3** The curve is monotonically non-increasing as depth increases.

### US-04 — Understand dip/rise sequencing
**As a** user considering a bracket order
**I want** to know whether the upside or downside leg tends to arrive first
**So that** I am not misled by a target that is frequently reached only after my stop would have triggered.

**Acceptance criteria**
- **AC-04.1** The band displays a reward-to-risk ratio (median rise ÷ median dip).
- **AC-04.2** The band displays the rate at which the high is set before the low within the window.
- **AC-04.3** Where a target level is quoted, the display distinguishes "ever touched" from "touched first" against the user's stop.

### US-05 — Compare baselines across a watchlist
**As a** user managing multiple positions
**I want** the band summarized for every ticker on my watchlist
**So that** I can rank candidates by how much downside I would be accepting.

**Acceptance criteria**
- **AC-05.1** The watchlist supports columns for floor %, accumulation-zone %, ceiling %, and reward-to-risk ratio.
- **AC-05.2** Columns are sortable.
- **AC-05.3** Tickers with insufficient history render a clear "insufficient data" state, not a zero or blank.

> **Implementation note:** `/watchlist` today is alert CRUD only (ticker,
> condition, threshold) — there is no existing tickers-with-computed-columns
> table on that page to extend. US-05 means building a new table view, not
> adding columns to one that exists. Scoped out of the first implementation
> pass regardless (core band only); noting here so it isn't re-discovered as
> a surprise when it is picked up.

### US-06 — Trust the numbers
**As a** user relying on these levels
**I want** to see how often the stated floor actually held historically
**So that** I can judge whether the band is calibrated or merely plausible.

**Acceptance criteria**
- **AC-06.1** Observed breach rate displays alongside the expected breach rate (1 − confidence).
- **AC-06.2** Effective independent sample size displays, not just the raw window count.
- **AC-06.3** Where observed breach rate diverges from expected by more than 5 percentage points, a calibration warning displays.

---

## 4. Functional Requirements

### 4.1 Data input
- **FR-01** The engine shall accept daily OHLC bars with a date index and open, high, low, close columns.
- **FR-02** The engine shall use split- and dividend-adjusted prices.
- **FR-03** The engine shall reject bars containing non-positive or non-finite prices.
- **FR-04** The engine shall require a minimum of `H + 2` bars and shall fail explicitly, with the shortfall stated, when history is insufficient.
- **FR-04a** Meeting FR-04 is necessary but not sufficient to display a band: the engine shall additionally require at least 3 independent (non-overlapping) H-day windows of history before returning levels, and shall fail explicitly, stating the shortfall, when this is not met. (`H + 2` bars technically satisfies FR-04 at H = 90 while providing exactly one non-overlapping window — not enough to support any confidence statement.)
- **FR-05** The engine shall default to three years of daily history where available.

### 4.2 Excursion calculation
- **FR-06** All excursions shall be computed in log space so they are additive, and converted back to price for display.
- **FR-07** Dip and rise shall be measured against prior close by default, thereby including the overnight gap.
- **FR-08** The engine shall support an alternative open-referenced mode measuring intraday travel only.
- **FR-09** The **aggregated H-day MAE/MFE distribution** (not daily excursions prior to aggregation) shall be winsorized at the 0.5% and 99.5% percentiles to limit distortion from a small number of outlier windows, while preserving each window's realized worst/best day intact.

### 4.3 Horizon aggregation
- **FR-10** The engine shall provide three aggregation methods: `empirical`, `sqrt`, and `sum`.
- **FR-11** `empirical` shall be the default and shall derive levels from realized MAE and MFE across every rolling H-day window, winsorized per FR-09.
- **FR-12** `sqrt` shall scale a daily excursion quantile by √H, for use where history is thin.
- **FR-13** `sum` shall stack H daily excursion quantiles additively and shall be presented **only** as a stress rail, never as an expected level.
- **FR-14** The UI shall not surface `sum` as a selectable primary method. Additive stacking overstates realistic drawdown by approximately √H — measured at −47.6% versus a realized −9.0% on a 30-day test case — and is misleading if presented as a forecast.

### 4.4 Recency weighting
- **FR-15** Observations shall be weighted by an exponentially decaying function of age, with a configurable half-life defaulting to 126 trading days.
- **FR-16** Recency weighting shall be disableable, yielding equal weights.
- **FR-17** Any breach-rate diagnostic shall be computed under the same weights as the level it validates.

### 4.5 Two-sided sequencing
- **FR-18** The engine shall compute first-touch outcomes for a configurable grid of stop and target pairs.
- **FR-19** For each pair, the engine shall report probability of target-first, stop-first, and neither, plus expected return net of a configurable cost assumption.
- **FR-20** Where a single daily bar breaches both barriers, the engine shall resolve the outcome adversely, since intraday ordering is unrecoverable from OHLC.
- **FR-21** First-touch probabilities for any pair shall sum to 1.0 within a tolerance of 0.001.

### 4.6 Validation and safeguards
- **FR-22** The engine shall report observed breach rate under three views: weighted, full-history unweighted, and most-recent-quartile.
- **FR-23** The engine shall report effective independent sample size, accounting for window overlap, rather than raw window count.
- **FR-24** Any bracket recommendation derived from the barrier grid shall be fitted on an in-sample partition and scored on a held-out partition before display.
- **FR-25** Bracket recommendations failing out-of-sample validation shall be suppressed from the UI rather than shown with a caveat.
- **FR-26** The barrier grid shall be available as an internal research tool and shall remain feature-flagged off for end users until FR-24 is satisfied.
- **FR-27** Bracket ranking shall use return per unit of risk, not raw expected return or win rate.

---

## 5. Non-Functional Requirements

- **NFR-01** A single-ticker band across all four horizons shall compute in under 500 ms on three years of daily data, once that data is in memory (i.e. this budget covers the statistics, not a cold-cache fetch).
- **NFR-02** A 50-ticker watchlist refresh shall complete in under 5 seconds **assuming each ticker's OHLC history is already cached**. Meeting this on a cold cache requires new parallel/batched fetch infrastructure — see Dependencies (§10) — since no existing code path in this app fetches multiple tickers' history concurrently today.
- **NFR-03** Results shall be cached per ticker, horizon, confidence, and as-of date, invalidating on new market data.
- **NFR-04** The engine shall be a pure function of its inputs — no hidden state, deterministic for identical inputs.
- **NFR-05** The engine shall be importable independently of the web layer to support the licensing API path.
- **NFR-06** All outputs shall be JSON-serializable.
- **NFR-07** Insufficient-history and calculation failures shall return a structured error, never a silent null or zero.

---

## 6. API Contract

**Endpoint:** `GET /api/v1/baseline/{ticker}`

**Query parameters**

| Parameter | Type | Default | Constraint |
|---|---|---|---|
| `horizon` | int | 30 | one of 10, 30, 60, 90 |
| `confidence` | float | 0.90 | 0.50 ≤ c < 1.00 |
| `method` | string | `empirical` | `empirical` \| `sqrt` |
| `half_life` | int \| null | 126 | > 0 or null |

**Response fields**

| Field | Type | Description |
|---|---|---|
| `ticker`, `as_of`, `last_price` | — | Identity and reference price |
| `horizon_days`, `confidence`, `method` | — | Echo of parameters applied |
| `floor`, `accumulation_zone_hi`, `median_path`, `distribution_zone_lo`, `ceiling` | float | The five band levels, in price |
| `*_pct` | float | Each level as percent from last close |
| `rr_ratio`, `skew`, `upside_first_rate` | float | Two-sided shape metrics |
| `samples`, `effective_samples` | int | Raw and overlap-adjusted counts |
| `breach_rate`, `breach_rate_full`, `breach_rate_recent`, `expected_breach` | float | Calibration diagnostics |
| `fill_curve` | object | Depth to fill-probability mapping |

**Error responses**

| Code | Condition |
|---|---|
| 400 | Parameter outside permitted range |
| 404 | Unknown ticker |
| 422 | Insufficient history (FR-04 or FR-04a); response states which, and the shortfall |

---

## 7. Display and Disclosure

- **DR-01** The band shall be labelled as derived from historical price paths only.
- **DR-02** The UI shall state that the band incorporates no forward-looking information — no earnings, guidance, corporate actions, or news.
- **DR-03** The UI shall not present any level as a prediction, recommendation, target, or fair-value estimate in the valuation sense.
- **DR-04** Where confidence is 95% or higher, the UI shall note that the level is indicative rather than measured, given limited independent samples in the tail.
- **DR-05** Win rate shall never be displayed as a standalone quality metric without the corresponding risk-reward ratio.
- **DR-06** Where this ticker's page also shows Predict's confidence interval or Entry Signals' stop/target, the UI shall not imply the numbers should reconcile. Each reflects a different methodology (realized historical excursion vs. forecast-residual error vs. an ATR-based heuristic) and shall be labeled distinctly enough that a user encountering different levels for the same ticker on different pages understands why, rather than concluding the app disagrees with itself.

---

## 8. Out of Scope

- Intraday and tick-level data; daily OHLC only
- Options-implied volatility as an input or cross-check
- Fundamental, sentiment, or news-derived inputs
- Order routing or execution
- Portfolio-level aggregation of bands across positions
- Regime detection or explicit volatility forecasting models

---

## 9. Open Questions

1. ~~Should the band be volatility-normalized to allow cross-ticker comparison in the watchlist ranking, or shown in raw percentage terms?~~ **Resolved:** raw percentage for v1 (matches AC-05.1 literally, simpler to reason about); revisit if watchlist ranking quality turns out to need it.
2. ~~For funds and ETFs, should the band be computed on NAV or on market price where they diverge?~~ **Resolved:** market price. No existing data source in this app provides fund NAV; adding one is a separate, unscoped effort.
3. ~~What is the minimum history threshold below which the feature is hidden entirely rather than shown with a warning?~~ **Resolved by FR-04a** — hidden (structured 422, per NFR-07) below 3 independent H-day windows.
4. ~~Should the horizon selector offer calendar days as an alternative to trading days for non-professional users?~~ **Resolved:** trading days only. Every other horizon/lookback concept in this app (momentum backtest, entry signals, portfolio insights) is already trading-day-based; a calendar-day alternate here would be the first inconsistency of its kind for marginal benefit.
5. Does the licensing API path require a different confidence granularity than the three fixed retail options? **Still open** — depends on actual licensing customer requirements, which aren't known yet.

---

## 10. Dependencies

- Adjusted daily OHLC history, three years, for all covered tickers
- Existing ticker detail and watchlist pages
- Caching layer
- Feature-flag mechanism for FR-26
- **New:** a parallel/batched historical-OHLC fetch capability. Every existing multi-ticker code path in this app (`services/stock_finder_service.get_stock_finder_table`) fetches sequentially; even its ~24-ticker universe is cached for a full hour, which is itself evidence that a live sequential refresh at that scale is already costly. NFR-02 cannot be met without this being built first.
