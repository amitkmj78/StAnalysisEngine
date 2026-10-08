-- COM-5: follow an author, get alerted on their new ideas. No existing
-- follow/social-graph table anywhere in this app (confirmed) --
-- genuinely new. No RLS: a follow relationship isn't sensitive the way
-- a private alert/portfolio row is, and the leaderboard/profile pages
-- need to show follower counts across users.
create table if not exists author_follows (
  id bigint generated always as identity primary key,
  follower_user_id uuid not null references users(id) on delete cascade,
  followed_user_id uuid not null references users(id) on delete cascade,
  created_at timestamptz not null default now(),
  constraint author_follows_unique unique (follower_user_id, followed_user_id),
  constraint author_follows_no_self_follow check (follower_user_id <> followed_user_id)
);
create index if not exists author_follows_follower_idx on author_follows(follower_user_id);
create index if not exists author_follows_followed_idx on author_follows(followed_user_id);

-- COM-5's in-app alert: "an author you follow published a new idea" --
-- an 8th branch in web/backend/routers/alerts_inbox.py's UNION, the
-- exact ALX-1 condition_alerts pattern. user_id here is the FOLLOWER
-- (the alert's recipient), same convention watchlist_alerts/
-- condition_alerts already use for "whose inbox this shows up in".
-- UNLIKE community_ideas/author_follows above, this one DOES need RLS
-- -- alerts_inbox.py's UNION runs through user_conn (RLS-scoped), and
-- without a policy here every follower's alert would leak to every
-- user who ever opens their inbox.
create table if not exists followed_author_alerts (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  author_user_id uuid not null references users(id) on delete cascade,
  idea_id bigint not null references community_ideas(id) on delete cascade,
  ticker text not null,
  created_at timestamptz not null default now(),
  seen_at timestamptz
);
create index if not exists followed_author_alerts_user_idx on followed_author_alerts(user_id, created_at desc);
alter table followed_author_alerts enable row level security;
drop policy if exists followed_author_alerts_isolation on followed_author_alerts;
create policy followed_author_alerts_isolation on followed_author_alerts for all
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

grant select, insert, delete on author_follows to app_user;
grant select, insert, delete on author_follows to app_service;
grant select, update on followed_author_alerts to app_user;
grant select, insert, update on followed_author_alerts to app_service;
