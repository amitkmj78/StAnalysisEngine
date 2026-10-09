-- Short-/Long-Term Plan text (services/portfolio_strategy.py) was keyed
-- off a risk_profile/risk_factor that was hardcoded to "Balanced"/5 on
-- the frontend for every user, with no way to change it -- this is why
-- almost every position's plan read identically. Now a real per-portfolio
-- setting, edited via PUT /portfolio/{id}/risk-profile. Safe to re-run.

alter table portfolios add column if not exists risk_profile text not null default 'Balanced';
alter table portfolios add column if not exists risk_factor smallint not null default 5;
