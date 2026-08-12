# Stock Screener Improvements — Requirements Specification

**Feature:** Stock Screener (`/stock-finder`) — Filtering, Customization, Saved Screens, Row Actions
**Epic:** Discovery & Research Tools
**Version:** 0.1 (Draft) **Status:** Draft for review
**Target release:** TBD
**Owner:** Amit Kumar Maharaj

---

## 1. Purpose

The screener ranks a fixed universe of tickers by a composite Score and shows a
9-column table. It has no way to narrow that universe by the criteria users
actually screen on (valuation, size, sector, liquidity), no way to save a
screening configuration for reuse, and no path from a row to the rest of the
app's tools (Watchlist, Predict, Safe Baseline). This spec closes those gaps
without changing how the Score itself is computed.

### 1.1 Relationship to existing code

- **`services/stock_finder_service.py`** (`rank_stocks`, `score_stock_ticker`)
  is the only scoring engine the current web app (`web/backend/routers/stock_finder.py`,
  `web/frontend/app/stock-finder/page.tsx`) uses. This spec extends it; it does
  not replace it.
- **`services/screener_service.py`**'s `get_top_bullish_stocks` (a second,
  independent "Bullish Score"/Rating engine) is called only from
  `services/app.py` — the legacy Streamlit entry point, not the FastAPI/Next.js
  web app (confirmed by grep: no reference anywhere under `web/`). **Finding:**
  there is no "two engines disagreeing" problem inside the current web app to
  reconcile; `screener_service`'s scoring is simply unused there. Untangling it
  from the deprecated Streamlit app is a separate, unscoped cleanup — see §8.
- **`_build_stock_row`** (in `stock_finder_service.py`) already fetches Sector,
  Market Cap, Forward P/E, Revenue Growth, Earnings Growth, and every return/
  technical column alongside the ones currently displayed. **Finding:** the
  data this spec's filters need already exists in the cached per-universe
  DataFrame (`get_stock_finder_table`, 1hr TTL) — filtering requires no new
  data source, only new query parameters and post-fetch filtering.
- **`web/backend/utils.records_safe`** already serializes every DataFrame
  column to the API response, not just the frontend's hardcoded
  `DISPLAY_COLUMNS` subset. **Finding:** column customization (US-02) is a
  frontend-only change — the data is already on the wire.

---

## 2. Definitions

| Term | Definition |
|---|---|
| **Universe** | A named, fixed list of tickers (`STOCK_UNIVERSES`) the screener ranks within |
| **Goal** | `Short Term` or `Long Term` — selects which metric weights compute the Score |
| **Score** | 0–100 composite, min-max normalized *within the current result set* (relative, not absolute — see existing `COLUMN_INFO["Score"]` disclosure) |
| **Filter** | A user-supplied constraint (range or category) narrowing the ranked/scored rows before display |
| **Screen** | A named, saved combination of goal + universe + filters + visible columns + sort, reloadable later |
| **Row action** | A one-click operation from a result row into another feature (Watchlist, Predict, Safe Baseline) |

---

## 3. User Stories

### US-01 — Filter results by criteria
**As a** user screening for candidates that fit my strategy
**I want** to narrow ranked results by market cap, forward P/E, sector, and volume strength
**So that** I only see tickers that already fit my basic constraints, without manually scanning the full table.

**Acceptance criteria**
- **AC-01.1** Filters available: Market Cap ($B) min/max, Forward P/E min/max, Sector (multi-select), Volume Strength % min.
- **AC-01.2** Filters apply to both `rank` mode results and are combinable (AND, not OR).
- **AC-01.3** A ticker missing a value for a filtered field (e.g. no Forward P/E) is excluded when that filter is active, and the UI states this rather than silently dropping rows unexplained.
- **AC-01.4** Filters can be cleared individually or all at once without re-running the scan.
- **AC-01.5** The result count updates live as filters change, without a new network request (filtering happens client-side against the already-fetched full result set).

### US-02 — Customize visible columns
**As a** user who cares about specific metrics
**I want** to choose which columns show in the results table
**So that** I can see Forward P/E or Revenue Growth without those columns being hidden by a fixed default set.

**Acceptance criteria**
- **AC-02.1** A column picker lists every field present in the result set (all ~20 already returned by the API, not just the current 9).
- **AC-02.2** At least one column must remain visible (Ticker cannot be hidden).
- **AC-02.3** The chosen column set persists across page reloads (localStorage, matching the existing `PortfolioSwitcher` pattern).

### US-03 — Sort by multiple columns
**As a** user comparing close-scoring candidates
**I want** to sort by more than one column (e.g. Score, then Market Cap as a tiebreaker)
**So that** ties resolve in an order I choose rather than an arbitrary one.

**Acceptance criteria**
- **AC-03.1** Shift-click (or an equivalent explicit control) adds a column as a secondary/tertiary sort key.
- **AC-03.2** Active sort keys and their order display visibly (e.g. numbered indicators), not just direction arrows.
- **AC-03.3** Single-click sort behavior (replace, not add) is preserved as the default for users who don't need multi-sort.

### US-04 — Save and reload a screen
**As a** user with a repeatable screening strategy
**I want** to save my goal, universe, filters, columns, and sort as a named screen
**So that** I can re-run it later in one click instead of re-entering every control.

**Acceptance criteria**
- **AC-04.1** "Save this screen" captures goal, universe, all active filters, visible columns, and sort order under a user-supplied name.
- **AC-04.2** A saved-screens list shows name and saved-at date; selecting one restores every captured control and re-runs the scan.
- **AC-04.3** Saved screens can be deleted.
- **AC-04.4** Saved screens are private per user (RLS-isolated, matching every other saved-* table in this app).

### US-05 — Compare a saved screen's results over time
**As a** user tracking whether my strategy's picks are stable
**I want** to compare a saved screen's top results against a fresh run of the same screen
**So that** I can see whether the leaderboard has actually changed, not just re-read a table and guess.

**Acceptance criteria**
- **AC-05.1** "Save this screen" (US-04) also snapshots the top 10 ranked rows (ticker, Score, Price) at save time.
- **AC-05.2** Reloading a saved screen and re-running it offers a "Compare to saved" view showing rank/Score movement per ticker (new entrant, dropped out, moved up/down) — the same before/after framing already used by the Safe Baseline Band's `CompareSnapshot`.
- **AC-05.3** This is explicitly a snapshot comparison, not a backtest — no claim is made about why a ticker's rank changed.

### US-06 — Act on a result row
**As a** user who found a promising ticker in the screener
**I want** to add it to my watchlist or jump to its forecast/price band without retyping the ticker
**So that** the screener is a starting point for research, not a dead end.

**Acceptance criteria**
- **AC-06.1** Each row has an "Add to Watchlist" action that opens the existing alert-creation flow pre-filled with the ticker.
- **AC-06.2** Each row has a "View in Predict" link to `/predict?ticker=...` and a "View in Safe Baseline" link to the same, both pre-filling the ticker.
- **AC-06.3** Row actions do not require leaving the results table (e.g. a small menu or icon buttons per row), so a user can act on several rows without losing their filtered/sorted state.

---

## 4. Functional Requirements

### 4.1 Filtering
- **FR-01** The `/rank` endpoint shall accept optional query parameters: `market_cap_min`, `market_cap_max` (float, $B), `forward_pe_min`, `forward_pe_max` (float), `sector` (repeatable string), `volume_strength_min` (float, %).
- **FR-02** Filters shall be applied to the already-cached per-universe DataFrame (`get_stock_finder_table`) after scoring, not by re-fetching data — filtering must not add new yfinance calls or bypass the existing 1hr TTL cache.
- **FR-03** A row with a null value for a field under an active min/max filter shall be excluded from that filter's results.
- **FR-04** Sector values offered as filter options shall be derived from the distinct `Sector` values actually present in the current universe's cached table, not a hardcoded list (universes vary in which sectors they contain).

### 4.2 Column customization
- **FR-05** The frontend shall support toggling visibility of any field already present in a result row; no backend change is required since `records_safe` already returns every column.
- **FR-06** Selected columns shall persist in `localStorage`, scoped independently of goal/universe/mode selection.

### 4.3 Multi-column sort
- **FR-07** Sorting shall support up to 3 active keys, each independently ascending/descending, applied client-side against the already-fetched result set (matching the existing single-column client-side sort already in `stock-finder/page.tsx`).

### 4.4 Saved screens
- **FR-08** A new `saved_screens` table shall store: `user_id`, `name`, `goal`, `universe`, `filters` (jsonb), `visible_columns` (jsonb array), `sort_keys` (jsonb array), `snapshot_top10` (jsonb, per US-05), `saved_at`.
- **FR-09** `POST /api/v1/stock-finder/screens/save`, `GET /api/v1/stock-finder/screens`, `DELETE /api/v1/stock-finder/screens/{id}` shall follow the same RLS-isolation, `user_conn`, and `RETURNING *`/plain-dict-response pattern as `saved_baseline_snapshots` (`web/backend/routers/baseline.py`) — no new backend pattern needed.
- **FR-10** Reloading a saved screen shall re-run `rank_stocks` fresh (respecting the existing 1hr cache) rather than replaying stored results, except for the US-05 comparison view, which explicitly uses the stored `snapshot_top10`.

### 4.5 Row actions
- **FR-11** "Add to Watchlist" shall call the existing `POST /api/v1/watchlist` endpoint with the row's ticker pre-filled; condition type/threshold still require user input (screener has no default threshold to assume).
- **FR-12** "View in Predict" / "View in Safe Baseline" shall be plain links using existing page routes with a `?ticker=` query param; both target pages must read that param on load (currently `/predict` defaults to a hardcoded `"AAPL"` — this requires a small change there to seed `ticker` state from the URL if present).

---

## 5. Non-Functional Requirements

- **NFR-01** Adding filters, column customization, and multi-sort shall not increase `/rank`'s response latency beyond current levels — all three operate on data already being fetched/returned today.
- **NFR-02** Saved-screen CRUD shall follow the same per-user RLS isolation as every other `saved_*` table (`saved_predictions`, `saved_narratives`, `saved_baseline_snapshots`).
- **NFR-03** The screener page shall remain usable with zero saved screens and zero filters applied — every addition in this spec is opt-in, not a required step before seeing ranked results (preserves current first-run behavior).

---

## 6. API Contract

**Endpoint (extended):** `GET /api/v1/stock-finder/rank`

**New query parameters** (all optional; existing `goal`, `universe` unchanged)

| Parameter | Type | Constraint |
|---|---|---|
| `market_cap_min` / `market_cap_max` | float | ≥ 0, $B |
| `forward_pe_min` / `forward_pe_max` | float | — |
| `sector` | string, repeatable | must be a sector present in the current universe |
| `volume_strength_min` | float | — |

**New endpoints**

| Method & Path | Purpose |
|---|---|
| `POST /api/v1/stock-finder/screens/save` | Save a named screen (FR-08/09) |
| `GET /api/v1/stock-finder/screens` | List the current user's saved screens |
| `DELETE /api/v1/stock-finder/screens/{id}` | Delete a saved screen |

**Error responses**

| Code | Condition |
|---|---|
| 422 | `sector` value not present in the resolved universe; min > max on any range filter |
| 404 | Saved screen not found (delete) or doesn't belong to the requesting user (RLS) |

---

## 7. Out of Scope

- Reconciling or removing `screener_service.get_top_bullish_stocks` — unused by the web app (§1.1); touching it only affects the legacy Streamlit app, a separate effort.
- Expanding universes beyond the current hardcoded `INDEX_MAP` lists (e.g. full S&P 500, dynamic index membership) — a data-sourcing problem, not a filtering/UX one.
- Server-side pagination — current universes are small enough (~20–40 tickers) that client-side filtering/sorting of the full result set is sufficient; revisit if universes grow substantially.
- Fund/ETF-specific filter criteria (expense ratio, AUM) — `stock-finder` only covers the stock universes; `index-fund` is the separate existing feature for funds.
- Sharing a saved screen with other users — saved screens are private per user, matching every other `saved_*` table in this app.

---

## 8. Open Questions

1. Should "Add to Watchlist" (US-06/AC-06.1) offer a sensible default threshold (e.g. current price ± 5%) instead of requiring the user to type one, given the screener already knows the current price? Leaning yes, but not blocking — can default to the existing empty-threshold flow and revisit.
2. Does `services/app.py` (legacy Streamlit) still need to be maintained at all, given the FastAPI/Next.js app has superseded it for every feature checked in this spec? Out of scope to decide here, but worth raising separately — `screener_service.get_top_bullish_stocks` being unused outside it is one more data point.

---

## 9. Dependencies

- Existing `stock_finder_service.get_stock_finder_table` cache (no changes needed, just new consumers of its already-fetched columns)
- Existing `saved_baseline_snapshots`/`baseline.py` save/history/delete pattern (template for FR-08/09)
- Existing `watchlist_alerts`/`POST /api/v1/watchlist` endpoint (target of US-06's row action)
- `/predict` page needs to read an optional `?ticker=` URL param on load (currently always defaults to `"AAPL"` on mount) for US-06/FR-12 deep links to work
