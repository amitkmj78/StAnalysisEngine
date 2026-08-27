# Safe Baseline Price Band — Requirements Specification

**Feature:** Safe Baseline (Two-Sided Excursion Band)
**Epic:** Quant Model — Risk & Entry Planning
**Target release:** TBD
**Owner:** Amit Kumar Maharaj
**Status:** Draft for review

---

## 1. Purpose

Close-to-close returns describe where a security ended, not the path it took to
get there. A position can finish flat having traded 6% lower intraday. Users
setting limit orders, stops, or accumulation zones need the path.

The Safe Baseline feature derives a five-level price band for any ticker from
the historical distribution of intraday excursions — the maximum dip below and
maximum rise above a reference price — aggregated across a forward horizon.

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
- **AC-01.1** Given a ticker with sufficient history, when I open the ticker detail page, then a five-level ladder displays: ceiling, sell-zone floor, fair value, buy-zone ceiling, and floor.
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
- **AC-05.1** The watchlist supports columns for floor %, buy-zone %, ceiling %, and reward-to-risk ratio.
- **AC-05.2** Columns are sortable.
- **AC-05.3** Tickers with insufficient history render a clear "insufficient data" state, not a zero or blank.

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
- **FR-05** The engine shall default to three years of daily history where available.

### 4.2 Excursion calculation
- **FR-06** All excursions shall be computed in log space so they are additive, and converted back to price for display.
- **FR-07** Dip and rise shall be measured against prior close by default, thereby including the overnight gap.
- **FR-08** The engine shall support an alternative open-referenced mode measuring intraday travel only.
- **FR-09** Daily excursions shall be winsorized at the 0.5% and 99.5% percentiles to limit single-print distortion.

### 4.3 Horizon aggregation
- **FR-10** The engine shall provide three aggregation methods: `empirical`, `sqrt`, and `sum`.
- **FR-11** `empirical` shall be the default and shall derive levels from realized MAE and MFE across every rolling H-day window.
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

- **NFR-01** A single-ticker band across all four horizons shall compute in under 500 ms on three years of daily data.
- **NFR-02** A 50-ticker watchlist refresh shall complete in under 5 seconds.
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
| `floor`, `buy_zone_hi`, `fair`, `sell_zone_lo`, `ceiling` | float | The five band levels, in price |
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
| 422 | Insufficient history; response states bars required and bars available |

---

## 7. Display and Disclosure

- **DR-01** The band shall be labelled as derived from historical price paths only.
- **DR-02** The UI shall state that the band incorporates no forward-looking information — no earnings, guidance, corporate actions, or news.
- **DR-03** The UI shall not present any level as a prediction, recommendation, target, or fair-value estimate in the valuation sense.
- **DR-04** Where confidence is 95% or higher, the UI shall note that the level is indicative rather than measured, given limited independent samples in the tail.
- **DR-05** Win rate shall never be displayed as a standalone quality metric without the corresponding risk-reward ratio.

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

1. Should the band be volatility-normalized to allow cross-ticker comparison in the watchlist ranking, or shown in raw percentage terms?
2. For funds and ETFs, should the band be computed on NAV or on market price where they diverge?
3. What is the minimum history threshold below which the feature is hidden entirely rather than shown with a warning?
4. Should the horizon selector offer calendar days as an alternative to trading days for non-professional users?
5. Does the licensing API path require a different confidence granularity than the three fixed retail options?

---

## 10. Dependencies

- Adjusted daily OHLC history, three years, for all covered tickers
- Existing ticker detail and watchlist pages
- Caching layer
- Feature-flag mechanism for FR-26
