-- COM-1/2/3: a user-settable, unique public display name -- there was
-- no "show this user to others" concept in this app beyond
-- services/challenge_service.py::mask_email (deliberately low-fidelity,
-- wrong for a persistent public author identity). Nullable: existing
-- users have none until they set one, and publishing an idea requires
-- one already set (enforced in the router, not here).
alter table users add column if not exists display_name text unique;

-- COM-1: a published, timestamped, LOCKED idea (no edit path exists or
-- ever will -- only COM-6's moderation can hide one later). author_user_id
-- is nullable + is_model distinguishes the app's own model "author"
-- (COM-7) from a real user -- mirrors services/quant_model_service.py's
-- MODEL_MEMBER_LABEL pattern for the challenges leaderboard (a synthetic
-- entry with user_id=None, is_model=True, no real users row at all)
-- rather than inventing a fake sentinel user row.
create table if not exists community_ideas (
  id bigint generated always as identity primary key,
  author_user_id uuid references users(id) on delete cascade,
  is_model boolean not null default false,
  ticker text not null,
  direction text not null,
  horizon_days int not null,
  target real,
  stop real,
  entry_price real not null,
  -- COM-6: required at publish, never optional -- the real, honest
  -- moderation mechanism this round builds (see the migration's own
  -- disclosure in the tracker note: no automated pump-and-dump/paid-
  -- promotion detection exists or can realistically exist in code).
  has_position boolean not null,
  disclosure_note text,
  attested_no_promotion boolean not null,
  hidden boolean not null default false,
  -- COM-2: null until the horizon elapses and the nightly job scores
  -- it once -- never rescored, never guessed at early.
  realized_return_pct real,
  excess_vs_spy_pct real,
  outcome text,
  scored_at timestamptz,
  created_at timestamptz not null default now(),
  constraint community_ideas_direction_check check (direction in ('LONG', 'SHORT')),
  constraint community_ideas_author_check check (
    (is_model and author_user_id is null) or (not is_model and author_user_id is not null)
  )
);
create index if not exists community_ideas_author_idx on community_ideas(author_user_id, created_at desc);
create index if not exists community_ideas_unscored_idx on community_ideas(scored_at) where scored_at is null;

-- No RLS: ideas are public data (feed/profile/leaderboard readable by
-- everyone), the same posture signal_outcomes/published_signals
-- already have -- author checks happen in the router, not a policy.
grant select, insert, update on community_ideas to app_user;
grant select, insert, update on community_ideas to app_service;
