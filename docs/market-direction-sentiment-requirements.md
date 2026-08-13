# Market Direction & Sentiment Engine — Requirements

**Feature:** Market Direction Dashboard (`/market`)
**Product:** Stock/Fund Prediction Platform
**Version:** Draft v0.1 — **Phase 1 (Internals) attempted twice and gated both times, see §9a and §9b**
**Purpose:** Establish a top-down market regime ("which way is the tape leaning?") from news, earnings, and price internals, so that ticker-level buy/sell decisions are taken *with* the market rather than against it.

---

## 1. Problem Statement

Today the platform produces bottom-up, per-ticker predictions. A model can be right about a stock and still lose money because the whole market was risk-off that week. There is no layer that answers: *should I be adding risk, trimming risk, or standing still today?*

This feature adds that layer as an explicit, scored, auditable input — not a vibe.

**Design principle:** market sentiment is used as a **gate / position-size modifier**, not as a standalone entry signal. News sentiment at daily horizons is noisy and mean-reverting; treating it as an alpha source directly is the most common way these systems fail. See §9 (Validation) — the feature does not ship until it is proven on this point.

---

## 2. Scope

### In scope (v1)
- Composite Market Direction Score for US equities (broad market + 11 GICS sectors)
- Three input pillars: News Sentiment, Earnings Sentiment, Market Internals
- `/market` dashboard page
- Regime label surfaced on existing `/predictions`, `/watchlist`, and `/strategies` pages
- Historical score series + backtest harness

### Out of scope (v1)
- Non-US markets, FX, crypto, fixed income
- Intraday (sub-daily) regime updates
- Social media / Reddit / X sentiment (deferred to v2 — high noise, high spam risk)
- Any automated order routing off the score

---

## 3. Definitions

| Term | Definition |
|---|---|
| **MDS** | Market Direction Score. Composite, range −100 to +100. |
| **Regime** | Bucketed MDS: Risk-Off, Cautious, Neutral, Constructive, Risk-On. |
| **Pillar** | One of the three sub-scores (News, Earnings, Internals), each normalized −100 to +100. |
| **Point-in-time (PIT)** | A value as it was knowable at timestamp T, with no later revisions folded in. |
| **Breadth** | % of index constituents above their 50-day moving average. |

---

## 4. Data Requirements

### 4.1 News
| Requirement | Detail |
|---|---|
| DR-N1 | Ingest headlines + summaries for: broad market keywords, each of 11 sectors, and every ticker in the user universe (S&P 500 + user watchlists). |
| DR-N2 | Every article stored with `published_at` (UTC, source-provided) **and** `ingested_at`. Backtests must use `published_at`. |
| DR-N3 | Deduplicate syndicated copies. Cluster by title similarity (cosine on embeddings, threshold configurable, default 0.85) within a 48h window; one cluster = one event, weighted once. |
| DR-N4 | Source credibility weight per publisher (tier 1 wires = 1.0, aggregators = 0.5, promotional/PR wires = 0.2). Weights stored in config, editable without redeploy. |
| DR-N5 | Retain 3 years of history to match existing backtest depth. |
| DR-N6 | Handle provider rate limits and outages gracefully — stale data must be *labeled* stale, never silently reused as fresh. |

### 4.2 Earnings
| Requirement | Detail |
|---|---|
| DR-E1 | Earnings calendar: upcoming reports for next 30 days, with confirmed vs. estimated date flag. |
| DR-E2 | For each reported quarter: EPS actual vs. consensus, revenue actual vs. consensus, surprise %, and guidance direction (raised / maintained / lowered / withdrawn / none). |
| DR-E3 | Post-earnings 1-day price reaction captured — the market's verdict often contradicts the headline beat. |
| DR-E4 | Aggregate beat-rate and median surprise % rolling over the trailing 30 days, at market and sector level. |
| DR-E5 | Guidance direction must be parsed from the call/release text where not provided structurally; low-confidence parses flagged, not guessed. |

### 4.3 Market Internals
| Requirement | Detail |
|---|---|
| DR-I1 | Index levels and returns: SPY/SPX, QQQ, IWM (1d, 5d, 21d). |
| DR-I2 | Breadth: % of S&P 500 above 50-DMA and 200-DMA; advance/decline line. |
| DR-I3 | Volatility: VIX level, VIX 5-day change, VIX term structure slope (VIX vs VIX3M) as a stress flag. |
| DR-I4 | Risk appetite spreads: XLY/XLP ratio, HYG/IEF ratio, equal-weight vs cap-weight (RSP/SPY). |
| DR-I5 | Sector relative strength: 21-day return of each sector ETF vs SPY. |

---

## 5. Scoring Model

### 5.1 Pillar computation

**News Pillar**
1. Score each unique article cluster on a −1..+1 scale using an LLM classifier with a fixed rubric (materiality, direction, whether the news is already priced in).
2. Weight each cluster by: `source_credibility × recency_decay × cluster_size_dampener`.
   - `recency_decay = exp(−age_hours / 36)` (half-life ≈ 25h, configurable).
   - `cluster_size_dampener = log(1 + n_articles)` — prevents one loud story dominating.
3. Sum, then convert to a z-score against the trailing 250-day distribution of the same statistic. Scale to −100..+100, clipped at ±3σ.

**Earnings Pillar**
- Inputs: rolling 30d beat rate (vs. its own 3-year mean), median revenue surprise, net guidance ratio (raised − lowered) / total, and median 1-day post-earnings reaction.
- Each z-scored on 3-year history, then equally weighted, scaled to −100..+100.
- Coverage rule: if fewer than 15 reports in the trailing 30 days (off-season), the pillar's weight is reduced proportionally and redistributed to the other two. This must be visible in the UI, not silent.

**Internals Pillar**
- Inputs: breadth level, breadth 5-day change, VIX z-score (inverted), term-structure inversion flag, risk-appetite ratio momentum.
- Same z-score → weight → scale treatment.

### 5.2 Composite

```
MDS = w_news × News + w_earn × Earnings + w_int × Internals
```

- SR-1: Default weights `w_news = 0.25`, `w_earn = 0.25`, `w_int = 0.50`. Internals gets the largest weight because price is the only pillar that reflects positioning, not narrative.
- SR-2: Weights are configurable per environment and must be versioned. Any score displayed is stored with the `model_version` that produced it.
- SR-3: MDS is smoothed with a 3-day EMA for display, but the raw daily value is also persisted. Smoothing prevents whipsaw in the UI; the raw value is what the backtest consumes.
- SR-4: A **conflict flag** is raised when any two pillars disagree by more than 60 points. Disagreement is information (e.g. good news into weak tape = distribution) and must be surfaced, not averaged away.

### 5.3 Regime mapping

| MDS | Regime | Suggested posture |
|---|---|---|
| +60 to +100 | Risk-On | Full model position sizing |
| +20 to +59 | Constructive | Normal sizing |
| −19 to +19 | Neutral | Reduced sizing; require stronger per-ticker conviction |
| −59 to −20 | Cautious | Half sizing; new longs need confirmation |
| −100 to −60 | Risk-Off | Defensive; model longs suppressed or flagged |

- SR-5: Regime changes require the smoothed MDS to hold the new band for 2 consecutive sessions (hysteresis) to avoid flip-flopping on a single headline.

---

## 6. Page & UI Requirements

Route: `/market`, nav label **"Market Direction"**.

| ID | Requirement |
|---|---|
| UI-1 | Hero: current MDS gauge (−100..+100), regime label, direction arrow vs. yesterday, and "as of" timestamp with data-freshness state (Live / Delayed / Stale). |
| UI-2 | Pillar breakdown: three cards with each pillar's score, its 5-day sparkline, and the top 3 contributing items (headline / metric) with their individual scores. **A score with no visible drivers is not shippable** — every number must be clickable down to its inputs. |
| UI-3 | Historical chart: MDS overlaid on SPY, 3-year range, zoomable. Regime bands shaded. |
| UI-4 | Sector heatmap: MDS per sector, sortable, click-through to sector detail with its own news/earnings feed. |
| UI-5 | Earnings calendar strip: next 10 trading days, count of reports per day, marked with any watchlist tickers. |
| UI-6 | News feed: deduplicated clusters, sorted by absolute impact score, each showing direction, sources, and published time. |
| UI-7 | Conflict banner when SR-4 fires, in plain language ("News is positive but internals are deteriorating — historically a poor entry environment"). |
| UI-8 | Regime badge component reused on `/predictions`, `/watchlist`, `/strategies` — one compact pill linking to `/market`. |
| UI-9 | On any prediction detail view, show the MDS *as of the prediction date* alongside the current MDS. This lets the user see whether a bad prediction was a model failure or a regime failure. |
| UI-10 | Prominent, non-dismissible disclaimer: informational/research tool, not investment advice, no guarantee of accuracy. Placement per §11. |

---

## 7. API Requirements

| Endpoint | Method | Returns |
|---|---|---|
| `/api/market/sentiment/current` | GET | MDS, regime, pillar scores, conflict flag, freshness, model_version |
| `/api/market/sentiment/history?from=&to=&granularity=` | GET | Time series of MDS + pillars |
| `/api/market/sentiment/sectors` | GET | Per-sector MDS + rank + 5d change |
| `/api/market/news?scope=&limit=` | GET | Scored, deduplicated clusters |
| `/api/market/earnings/calendar?days=` | GET | Upcoming reports + surprise history |
| `/api/market/sentiment/explain?date=` | GET | Full driver breakdown for one date (audit) |

- API-1: All responses include `as_of`, `model_version`, and `data_completeness` (0–1).
- API-2: Cached with TTL matched to refresh cadence (§8); cache key includes model_version.
- API-3: These endpoints must be designed to be externally exposable later — versioned path (`/api/v1/...`), auth-ready, rate-limit-ready — given the licensing direction for the platform.

---

## 8. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-1 | Refresh cadence: internals every 15 min during market hours; news every 30 min; earnings hourly and within 15 min of a scheduled release. Full recompute at 6:00 AM ET pre-open. |
| NFR-2 | `/market` page loads in under 2s at p95; cached score served, never computed on request. |
| NFR-3 | LLM scoring cost capped — batch article scoring, cache by content hash, never re-score an unchanged cluster. Target under a defined daily ceiling; alert at 80%. |
| NFR-4 | Every stored score is reproducible: inputs, weights, and model_version persisted so any historical MDS can be recomputed and verified. |
| NFR-5 | Graceful degradation: if a pillar's data is unavailable, compute MDS from the remaining pillars, reduce `data_completeness`, and show a degraded-state badge. Never display a confident score built on missing data. |
| NFR-6 | Timezone: all storage UTC, all display ET with explicit label. |
| NFR-7 | Audit log of every score change and every config/weight change, with actor. |

---

## 9. Validation & Backtest Requirements — *gating*

This section is a release gate, not documentation.

| ID | Requirement |
|---|---|
| V-1 | Reconstruct MDS daily over 3 years using strictly PIT data. Any use of a revised or late-arriving value invalidates the run. |
| V-2 | Measure forward SPY returns at 1, 5, 21 days conditioned on regime. Report mean, median, hit rate, and t-stat — including when results are unflattering. |
| V-3 | Compare the existing per-ticker model's performance **with and without** the regime gate: hit rate, average return, max drawdown, Sharpe. The gate ships only if it improves risk-adjusted return or drawdown; a gate that merely reduces trade count is not a win. |
| V-4 | Test each pillar in isolation to identify which is actually carrying signal. Expect internals to dominate. If news sentiment adds nothing incremental, reduce its weight rather than keeping it for narrative appeal. |
| V-5 | Sensitivity analysis on weights (±0.1) and on the decay half-life. If results collapse under small perturbations, the model is overfit and must be simplified. |
| V-6 | Out-of-sample holdout: tune on years 1–2, evaluate untouched on year 3. |
| V-7 | Document known limitations explicitly: 3 years covers a limited set of regimes; news coverage is survivorship-biased toward large caps; LLM sentiment scoring drifts with model updates and must be pinned by version. |

---

## 9a. Phase 1 Attempt — Result: Gated, Not Shipped

Phase 1 (Internals pillar only, per §13's phasing) was implemented and backtested against this section's own gate. **It did not pass and was not shipped.** Documented here so the attempt isn't silently re-litigated later without this context.

**What was built:** `compute_internals_score` (breadth level, breadth 5-day change, VIX inverted, VIX/VIX3M term-structure slope inverted, and a 3-ratio risk-appetite momentum composite — XLY/XLP, HYG/IEF, RSP/SPY — each z-scored against a trailing 250-day window per §5.1, equally weighted, scaled to ±100), the 5-band regime mapper, SR-5 hysteresis, and a real V-2 forward-return backtest harness. Data: 5 years of daily closes for all 503 S&P 500 constituents (breadth) plus VIX/VIX3M/sector-ratio ETFs, fetched via yfinance. 789 scored trading days after the 250-day z-score warm-up window (2022-05 through 2026-08).

**V-2 result (forward SPY returns conditioned on regime):**

| Horizon | Constructive mean | Neutral mean | Cautious mean | Risk-Off mean |
|---|---|---|---|---|
| 1d | +0.066% | +0.091% (p=0.010) | +0.095% | +0.178% |
| 5d | +0.304% (p=0.008) | +0.386% (p<0.001) | +0.529% (p=0.069) | **+1.668%** (78% hit rate, p=0.065) |
| 21d | +1.761% (p<0.001) | +1.280% (p<0.001) | **+3.207%** (p<0.001) | +2.672% (p=0.137, n=18) |

At every horizon, forward SPY returns were **higher after more "Cautious"/"Risk-Off" readings than after "Constructive" ones** — the reverse of what the regime labels are designed to mean ("Risk-On → add risk, Risk-Off → get defensive"). Several buckets are individually significant at p<0.01. Zero days in the sample reached the Risk-On band (+60), so that band is untested in either direction.

**Interpretation:** the internals inputs as specified (inverted VIX, inverted term-structure slope, positive breadth/risk-appetite) are behaving as a **contrarian / mean-reversion** signal over this sample, not a trend-confirmation signal — consistent with well-documented market behavior (VIX spikes and breadth washouts are classic oversold/bounce setups), but the opposite of how §5.3's regime postures ("Risk-Off → defensive, suppress longs") intend the score to be used. Per V-4's own instruction, this is exactly the kind of finding that must change the plan rather than be shipped anyway for narrative appeal.

**Decision:** per this spec's own gate (§9 intro: *"the feature does not ship until it is proven on this point"*), Phase 1 is **not** wired into any page, endpoint, or scheduled job. The scoring/backtest code is retained (`services/market_internals_service.py`, `services/market_data_service.py`, tested) as a validated-negative research artifact, not deleted, since reworking the signal (or the labels/postures) is a plausible follow-up.

**Not yet investigated** (would need to happen before any retry): whether a differently-signed version validates (e.g. treating a stress reading as a contrarian opportunity signal rather than a defensive one — which would mean rewriting §5.3's postures, not just the math); whether the contrarian pattern is specific to this 2022–2026 sample (dominated by sharp-drawdown-then-V-shaped-recovery episodes per V-7's own caution about limited regime coverage) or holds over a longer/different window; and V-5's weight-sensitivity check, which was never run since V-2 already failed.

---

## 9b. Phase 1 Rework Attempt — Result: Still Gated, Not Shipped

Per §9a's three "not yet investigated" leads, this attempt (1) tested a sign-flipped version of the score (treating internals stress as an opportunity signal rather than a danger signal), (2) extended the sample from 5 to 10 years (2017–2026, adding the 2018 Q4 selloff and the COVID crash to the regime mix, tested as independent first/second halves), and (3) ran V-5 weight-sensitivity on both orientations. It also fixed a real methodology gap surfaced in the process: `run_forward_return_backtest`'s significance test (`stats.ttest_1samp`) treated each day's H-day forward return as an independent draw, when consecutive days' forward-return windows overlap by H−1 days — this inflates apparent significance for any horizon beyond 1 day. `run_forward_return_backtest` now also reports `t_stat_hac`/`p_value_hac`, a Newey-West (Bartlett kernel, maxlags = horizon − 1) corrected pair, alongside the original naive fields (`services/market_internals_service.py`, `_newey_west_mean_test`, tested in `tests/test_market_internals_service.py`).

**Lead 1 result — the sign flip is not a new signal.** Because `REGIME_BANDS` is symmetric around zero, flipping the score's sign before mapping to a regime exactly swaps which days get which label — e.g. original's Risk-Off bucket (n=81, mean +2.17%) becomes flipped's Risk-On bucket, same 81 days, same number. It is a relabeling of §9a's finding, not independent evidence. Framed honestly, testing it means asking "should the bands be renamed and §5.3's postures rewritten (stress = opportunity, not danger)," not "does flipping fix the math."

**Lead 2 result — the pattern is directionally consistent across independent sub-periods, once relabeled.** Testing 2017–2021 and 2021–2026 as independent halves, the same direction holds in both (extreme-stress days precede above-average forward returns in each half on its own) — it is not solely an artifact of the 2022–2026 V-shaped-recovery sample flagged as a limitation in §9a.

**Lead 3 (V-5) result — direction is stable under weight perturbation** (up-weighting or dropping any single sub-signal by 0.5 kept the flipped orientation's monotonicity at 2–3/4 adjacent steps across every variant tested) — not fragile to reasonable reweighting.

**But the HAC-corrected significance test — the actual point of this rework — fails the bucket that matters:**

| Window | Risk-Off/extreme bucket n | p-value (naive) | p-value (HAC-corrected) |
|---|---|---|---|
| Full 10y (2017–2026) | 81 | 0.0531 | **0.3525** |
| First half (2017–2021) | 41 | 0.2682 | **0.6165** |
| Second half (2021–2026) | 18 | 0.1369 | **0.0584** |
| Last 5y (§9a's original sample) | 18 | 0.1369 | **0.0584** |

The extreme bucket — the one that would actually be interesting or actionable under either orientation — is not statistically distinguishable from zero in the full sample or in either independent half, once the overlapping-window autocorrelation the naive test ignored is properly corrected for. What looked like borderline-significant (p≈0.05) evidence in the naive test was substantially the naive test overstating confidence on a small, autocorrelated sample. The well-populated middle buckets (Neutral, and Cautious in most windows) remain HAC-significant, but that's just "the market has a positive expected return most of the time" — not something to build a regime-timing product around.

**Decision:** still gated, still not shipped — now on firmer methodological ground rather than a single naive-test result. Phase 1 has now failed its own release gate twice: once as originally specified (contrarian, §9a), and once in its most plausible reworked form (relabeled-contrarian/opportunity framing, properly significance-tested, §9b). The Newey-West correction is kept permanently in `run_forward_return_backtest` regardless of this outcome — it's a real methodology fix that would matter for any future attempt at this or a similar backtest, not specific to this result.

**What would change this:** a sample that includes a genuine multi-year structural bear market (this 2017–2026 window, like the original 5y one, doesn't have one — V-7's caution about limited regime coverage still applies), or a fundamentally different set of internals inputs rather than a relabeling of the same five. Absent either, further Phase 1 iteration on this exact input set is not a good use of time.

---

## 10. Data Model (indicative)

```
market_sentiment_daily
  date, scope (market|sector|ticker), scope_id,
  mds_raw, mds_smoothed, regime,
  news_score, earnings_score, internals_score,
  conflict_flag, data_completeness, model_version, computed_at

news_cluster
  cluster_id, canonical_title, first_published_at,
  article_count, avg_credibility, sentiment_score,
  materiality, scope, scope_id, model_version

news_article
  article_id, cluster_id, source, url, published_at,
  ingested_at, title, summary, content_hash

earnings_event
  ticker, fiscal_period, report_date, report_time (bmo|amc),
  eps_actual, eps_consensus, rev_actual, rev_consensus,
  guidance_direction, guidance_confidence,
  next_day_return, is_confirmed

internals_daily
  date, breadth_50dma, breadth_200dma, ad_line,
  vix, vix_5d_chg, term_slope, xly_xlp, hyg_ief, rsp_spy
```

---

## 11. Compliance & Disclosure

| ID | Requirement |
|---|---|
| C-1 | Every page and every API response carries: research/informational purposes only, not investment advice, not a recommendation to buy or sell. |
| C-2 | No language anywhere in UI or copy that promises, projects, or implies guaranteed returns. Regime labels describe conditions, never instruct ("Risk-Off" not "Sell now"). |
| C-3 | News content displayed as headline + link + short summary only, with attribution — no full-article reproduction. |
| C-4 | Data provider terms of service reviewed and recorded per source before ingestion, especially regarding redistribution — this matters more once the API is licensed externally. |
| C-5 | Retain the audit trail per NFR-7 for a defined period to support any later regulatory review. |

*Note: as this moves toward a product sold to outside users, the line between "research tool" and "advice" becomes a real regulatory question (investment adviser registration). Worth a securities attorney's read before external launch — I'm not a lawyer and this doc is not legal guidance.*

---

## 12. User Stories & Acceptance Criteria

**US-1 — See market direction before acting**
> As a user, I want a single market direction score with a plain-language regime label so I know whether conditions favor adding risk today.

*Accept:* `/market` shows MDS, regime, and timestamp; value matches the stored daily record; freshness state is accurate.

**US-2 — Understand why**
> As a user, I want to see what is driving the score so I can judge whether I agree with it.

*Accept:* Each pillar expands to its top 3 drivers with individual contributions; every driver links to its source article or metric.

**US-3 — Regime-aware predictions**
> As a user, I want each prediction annotated with the market regime at the time it was made so I can separate model error from market error.

*Accept:* Prediction detail shows MDS at creation and current MDS; `/predictions` list is filterable by regime-at-creation.

**US-4 — Sector rotation view**
> As a user, I want sector-level sentiment so I can see where strength is concentrating.

*Accept:* Heatmap of 11 sectors ranked by MDS, with 5-day change and click-through detail.

**US-5 — Earnings awareness**
> As a user, I want to know which of my holdings report soon so I am not blindsided.

*Accept:* Calendar strip marks watchlist tickers; alert configurable at T−3 days.

**US-6 — Historical context**
> As a user, I want to see how the score behaved in past drawdowns so I can calibrate my trust in it.

*Accept:* 3-year MDS vs SPY chart with shaded regimes; regime-conditional forward-return stats displayed from V-2.

**US-7 — Honest degradation**
> As a user, I want to be told when the score is running on incomplete data.

*Accept:* Degraded badge and `data_completeness` shown whenever any pillar is missing or stale.

---

## 13. Phasing

| Phase | Contents | Exit criteria |
|---|---|---|
| **P1 — Internals only** | Internals pillar, MDS, `/market` hero + history chart, regime badge on existing pages | V-1, V-2 pass on internals alone |
| **P2 — News** | Ingestion, clustering, LLM scoring, news feed, News pillar into composite | V-4 shows news adds incremental value at chosen weight |
| **P3 — Earnings** | Calendar, surprise/guidance aggregation, Earnings pillar, US-5 alerts | Coverage rule handles off-season correctly |
| **P4 — Integration** | Regime gate wired into position sizing, US-3, sector heatmap | V-3 passes: gated model beats ungated on risk-adjusted return |
| **P5 — Externalize** | API versioning, auth, rate limits, docs | Endpoints stable under load; ToS cleared per C-4 |

Build P1 first and resist the urge to start with news. Internals are cheap, reliable, and carry most of the signal; if P1 does not improve the existing model, adding news will not rescue it — and you will have learned that for a fraction of the cost.

---

## 14. Open Questions

1. Which news and fundamentals providers, and at what tier? Cost drives the clustering/scoring budget in NFR-3.
2. Is the ticker universe S&P 500, or S&P 1500, or watchlist-only? Breadth calculations depend on this.
3. Should the regime gate *suppress* model signals or only *resize* them? Suppression is cleaner to test; resizing is likelier to be right.
4. Retail-facing copy: how directive can regime language be without crossing into advice? Ties to C-2.
5. Does the sentiment score become a licensed API product on its own, separate from the per-ticker signals?
6. Fixed weights, or weights fit on the backtest? Fitted weights raise the overfitting risk in V-5 considerably — recommend fixed for v1.
