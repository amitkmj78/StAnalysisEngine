-- Economic calendar awareness banner: known, publicly-scheduled macro
-- data releases (CPI, PPI, GDP, Retail Sales, Jobs Report) surfaced a few
-- days in advance, plus FOMC meeting dates. Most types are auto-fetched
-- from the free FRED API at request time (see services/economic_calendar_
-- service.py); this table holds admin-entered dates, used as the only
-- source for FOMC (FRED's forward-date coverage for FOMC meetings is
-- unconfirmed) and as a manual corrective for any other type FRED misses.
-- Admin-only (service_conn/app_service), same posture as compliance_signoffs.
create table if not exists economic_event_overrides (
    id bigint generated always as identity primary key,
    event_type text not null check (event_type in
        ('CPI', 'PPI', 'GDP', 'RETAIL_SALES', 'JOBS_REPORT', 'FOMC')),
    event_date date not null,
    title text not null,
    note text,
    created_at timestamptz not null default now(),
    unique (event_type, event_date)
);
create index if not exists economic_event_overrides_date_idx on economic_event_overrides(event_date);
grant select, insert, update, delete on economic_event_overrides to app_service;

-- New user-visible feature -- a deploy must not itself turn the banner on;
-- an admin opts in once FRED_API_KEY is configured and FOMC override rows
-- (if needed) are entered. Same convention as every other toggle in
-- app_settings.py (see ECONOMIC_CALENDAR_ENABLED_KEY there).
insert into app_settings (key, value) values ('economic_calendar_enabled', 'false')
    on conflict (key) do nothing;
