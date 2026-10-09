-- BEG-4: a question is just a post with post_type = 'question' -- widen
-- the existing CHECK rather than a new table. An answer is just a
-- post_comment; is_accepted + the partial unique index enforce "at most
-- one accepted answer per post" at the DB level, not just in application
-- code. Safe to re-run.
alter table posts drop constraint if exists posts_post_type_check;
alter table posts add constraint posts_post_type_check
  check (post_type in ('note', 'performance_claim', 'question'));

alter table post_comments add column if not exists is_accepted boolean not null default false;
create unique index if not exists post_comments_one_accepted_per_post
  on post_comments (post_id) where is_accepted;
-- post_comments was previously select/insert-only -- marking an answer
-- accepted needs update too.
grant update on post_comments to app_user;
grant update on post_comments to app_service;

-- BEG-5: mentor_badge follows the exact same nightly-recompute shape as
-- verified_badge (web/backend/social_badges.py) -- never user-settable.
-- group_sessions is new: groups today are pure membership + a shared post
-- feed, with no scheduled-event concept at all.
alter table users add column if not exists mentor_badge boolean not null default false;

create table if not exists group_sessions (
  id bigint generated always as identity primary key,
  group_id bigint not null references groups(id) on delete cascade,
  host_user_id uuid not null references users(id) on delete cascade,
  title text not null,
  description text,
  scheduled_at timestamptz not null,
  created_at timestamptz not null default now()
);
create index if not exists group_sessions_group_idx on group_sessions(group_id, scheduled_at);

-- BEG-6: a challenge can opt into being beginner-only, which forces the
-- 'diversified' scoring method (services/challenge_service.py) rather than
-- leaving diversification optional.
alter table challenges drop constraint if exists challenges_scoring_check;
alter table challenges add constraint challenges_scoring_check
  check (scoring in ('return','sharpe','sortino','calmar','excess_spy','diversified'));
alter table challenges add column if not exists beginner_only boolean not null default false;
