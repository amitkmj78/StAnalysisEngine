-- SOC-1: profile fields. experience_level/interests are self-settable
-- (PUT /api/v1/social/profile). verified_badge is NEVER user-settable
-- -- only the nightly job in web/backend/social_badges.py sets it,
-- backed by real COM-3/COM-4 scored-idea track record, so the badge
-- means something (not a self-claimed flag).
alter table users add column if not exists experience_level text
  check (experience_level in ('beginner', 'intermediate', 'experienced'));
alter table users add column if not exists interests jsonb;
alter table users add column if not exists verified_badge boolean not null default false;

-- SOC-2: follow a ticker or topic for the feed -- distinct from
-- watchlist_alerts (a price-threshold-alert table, not a feed-follow
-- concept) and from author_follows (COM-5, people). No RLS, same "a
-- follow relationship isn't sensitive" reasoning as author_follows.
create table if not exists ticker_follows (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  ticker text not null,
  created_at timestamptz not null default now(),
  constraint ticker_follows_unique unique (user_id, ticker)
);
create index if not exists ticker_follows_user_idx on ticker_follows(user_id);

create table if not exists topic_follows (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  topic text not null,
  created_at timestamptz not null default now(),
  constraint topic_follows_unique unique (user_id, topic)
);
create index if not exists topic_follows_user_idx on topic_follows(user_id);

grant select, insert, delete on ticker_follows to app_user;
grant select, insert, delete on ticker_follows to app_service;
grant select, insert, delete on topic_follows to app_user;
grant select, insert, delete on topic_follows to app_service;

-- SOC-6: groups, public or private, by topic/ticker/strategy, with
-- moderators. No RLS -- public-group listing/content must be readable
-- across users, and a PRIVATE group's content is gated in the router
-- (web/backend/routers/social.py), the same app-layer-gated posture
-- idea_reports (COM-6) already uses for admin-only data, not a
-- Postgres policy keyed off a membership join.
create table if not exists groups (
  id bigint generated always as identity primary key,
  name text not null unique,
  slug text not null unique,
  description text,
  topic text,
  ticker text,
  is_private boolean not null default false,
  created_by_user_id uuid not null references users(id) on delete cascade,
  created_at timestamptz not null default now()
);
create index if not exists groups_topic_idx on groups(topic);
create index if not exists groups_ticker_idx on groups(ticker);

create table if not exists group_members (
  id bigint generated always as identity primary key,
  group_id bigint not null references groups(id) on delete cascade,
  user_id uuid not null references users(id) on delete cascade,
  role text not null default 'member' check (role in ('member', 'moderator', 'owner')),
  joined_at timestamptz not null default now(),
  constraint group_members_unique unique (group_id, user_id)
);
create index if not exists group_members_group_idx on group_members(group_id);
create index if not exists group_members_user_idx on group_members(user_id);

grant select, insert, update, delete on groups to app_user;
grant select, insert, update, delete on groups to app_service;
grant select, insert, update, delete on group_members to app_user;
grant select, insert, update, delete on group_members to app_service;

-- SOC-4: a frozen, shareable copy of a user's chart_drawings (CHT-5)
-- for one ticker, taken at the moment they attach it to a post.
-- SOC-4's "keep working on it" means fork-to-edit, not live multi-
-- user co-editing: a reader opening a shared chart gets this payload
-- loaded into THEIR OWN chart_drawings as a fresh, editable copy
-- (disclosed in the tracker, not a silent downgrade). No RLS: shared
-- by definition once attached to a post.
create table if not exists chart_snapshots (
  id bigint generated always as identity primary key,
  ticker text not null,
  created_by_user_id uuid not null references users(id) on delete cascade,
  drawings jsonb not null,
  created_at timestamptz not null default now()
);
grant select, insert on chart_snapshots to app_user;
grant select, insert on chart_snapshots to app_service;

-- SOC-2/3/4/5: the central social "post" -- a note or (SOC-5) a
-- performance claim, optionally scoped to a ticker/topic/group,
-- optionally carrying an attached chart snapshot (SOC-4). SOC-5:
-- there is no way to detect "a performance claim" from free text, so
-- the author explicitly marks post_type at compose time; a
-- performance_claim is `verified` only when it links one of the
-- author's own community_ideas (claim_reference_id) -- otherwise the
-- UI renders "unverified", never silently treated as verified. No
-- RLS: public feed data, same posture as community_ideas.
create table if not exists posts (
  id bigint generated always as identity primary key,
  author_user_id uuid not null references users(id) on delete cascade,
  post_type text not null default 'note' check (post_type in ('note', 'performance_claim')),
  body text not null,
  ticker text,
  topic text,
  group_id bigint references groups(id) on delete cascade,
  chart_snapshot_id bigint references chart_snapshots(id) on delete set null,
  claim_reference_id bigint references community_ideas(id) on delete set null,
  verified boolean not null default false,
  hidden boolean not null default false,
  created_at timestamptz not null default now(),
  constraint posts_claim_check check (post_type = 'performance_claim' or claim_reference_id is null)
);
create index if not exists posts_author_idx on posts(author_user_id, created_at desc);
create index if not exists posts_ticker_idx on posts(ticker, created_at desc) where ticker is not null;
create index if not exists posts_topic_idx on posts(topic, created_at desc) where topic is not null;
create index if not exists posts_group_idx on posts(group_id, created_at desc) where group_id is not null;
grant select, insert, update on posts to app_user;
grant select, insert, update on posts to app_service;

create table if not exists post_comments (
  id bigint generated always as identity primary key,
  post_id bigint not null references posts(id) on delete cascade,
  author_user_id uuid not null references users(id) on delete cascade,
  body text not null,
  created_at timestamptz not null default now()
);
create index if not exists post_comments_post_idx on post_comments(post_id, created_at);
grant select, insert on post_comments to app_user;
grant select, insert on post_comments to app_service;

-- SOC-7: permissioned DMs ("only from people the user follows or
-- allows"). direct_messages IS genuinely private (only sender/
-- recipient should ever see a row) and is queried through user_conn
-- -- RLS required, same reasoning as followed_author_alerts (COM-5).
create table if not exists dm_allowed_senders (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  allowed_sender_user_id uuid not null references users(id) on delete cascade,
  created_at timestamptz not null default now(),
  constraint dm_allowed_senders_unique unique (user_id, allowed_sender_user_id),
  constraint dm_allowed_senders_no_self check (user_id <> allowed_sender_user_id)
);
alter table dm_allowed_senders enable row level security;
drop policy if exists dm_allowed_senders_isolation on dm_allowed_senders;
create policy dm_allowed_senders_isolation on dm_allowed_senders for all
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);
grant select, insert, delete on dm_allowed_senders to app_user;
grant select, insert, delete on dm_allowed_senders to app_service;

create table if not exists direct_messages (
  id bigint generated always as identity primary key,
  sender_user_id uuid not null references users(id) on delete cascade,
  recipient_user_id uuid not null references users(id) on delete cascade,
  body text not null,
  created_at timestamptz not null default now(),
  read_at timestamptz,
  constraint direct_messages_no_self check (sender_user_id <> recipient_user_id)
);
create index if not exists direct_messages_thread_idx on direct_messages(sender_user_id, recipient_user_id, created_at);
alter table direct_messages enable row level security;
drop policy if exists direct_messages_isolation on direct_messages;
create policy direct_messages_isolation on direct_messages for all
  using (current_setting('app.user_id', true)::uuid in (sender_user_id, recipient_user_id))
  with check (current_setting('app.user_id', true)::uuid in (sender_user_id, recipient_user_id));
grant select, insert, update on direct_messages to app_user;
grant select, insert, update on direct_messages to app_service;

-- SOC-7: chat rooms (one per ticker, plus 'general') -- polling-based,
-- not WebSocket (no such layer exists anywhere in this app; every
-- other "live" feature already polls on an interval, same posture
-- kept here -- see services/social_chat_service.py). Public within a
-- room, no RLS (same posture as community_ideas); posting is gated by
-- the market-hours + room-name check in the router, not a policy.
create table if not exists chat_messages (
  id bigint generated always as identity primary key,
  room text not null,
  user_id uuid not null references users(id) on delete cascade,
  body text not null,
  created_at timestamptz not null default now()
);
create index if not exists chat_messages_room_idx on chat_messages(room, created_at);
grant select, insert on chat_messages to app_user;
grant select, insert on chat_messages to app_service;

-- SOC-9: one consolidated notifications table (not four near-
-- identical ones) covering new_follower/post_reply/mention/
-- group_activity -- mirrors followed_author_alerts' shape/RLS,
-- generalized across sub-types (reuse over duplication). Queried
-- through alerts_inbox.py's user_conn -- RLS required, same reasoning
-- as followed_author_alerts.
create table if not exists social_notifications (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  notification_type text not null check (notification_type in ('new_follower', 'post_reply', 'mention', 'group_activity')),
  actor_user_id uuid references users(id) on delete set null,
  post_id bigint references posts(id) on delete cascade,
  group_id bigint references groups(id) on delete cascade,
  summary text not null,
  created_at timestamptz not null default now(),
  seen_at timestamptz
);
create index if not exists social_notifications_user_idx on social_notifications(user_id, created_at desc);
alter table social_notifications enable row level security;
drop policy if exists social_notifications_isolation on social_notifications;
create policy social_notifications_isolation on social_notifications for all
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);
grant select, update on social_notifications to app_user;
grant select, insert, update on social_notifications to app_service;
