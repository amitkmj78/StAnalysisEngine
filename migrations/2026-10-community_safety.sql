-- SAF-1: guidelines acceptance (required checkbox at signup, same UX as
-- discoverable_for_challenges) + warn/ban. Ban reuses the existing
-- is_active/deactivate_user mechanism -- no new "banned" status invented.
alter table users add column if not exists guidelines_accepted_at timestamptz;
alter table users add column if not exists warning_count int not null default 0;
alter table users add column if not exists last_warning_note text;
alter table users add column if not exists last_warning_at timestamptz;

-- SAF-8: self-attestation only (not identity verification -- that needs a
-- KYC vendor, out of scope). No date-of-birth stored, less PII than that
-- while still satisfying "confirmed at sign-up."
alter table users add column if not exists age_confirmed_18_plus boolean not null default false;
-- Re-attested a second time immediately before a broker connection
-- (web/backend/routers/plaid_integration.py's link-token step).
alter table users add column if not exists age_reconfirmed_before_broker_at timestamptz;

-- SAF-6: portfolio-sharing is opt-in and off by default; when on, only a
-- percentage-only summary (never dollar amounts) renders on the author's
-- public community profile.
alter table users add column if not exists portfolio_sharing_enabled boolean not null default false;

-- SAF-4: position disclosure on posts (mirrors community_ideas.has_position
-- exactly) + an auto-verified tier when the author has a real Plaid-synced
-- position in that ticker (web/backend/routers/plaid_integration.py).
alter table posts add column if not exists has_position boolean;
alter table posts add column if not exists position_verified boolean not null default false;
-- SAF-5: self-declared paid-promotion label, honor system -- same shape as
-- community_ideas.attested_no_promotion.
alter table posts add column if not exists is_paid_promotion boolean not null default false;

-- SAF-2: generic report queue for posts/comments/groups/profiles --
-- idea_reports already covers community_ideas and is left as-is.
create table if not exists content_reports (
  id bigint generated always as identity primary key,
  content_type text not null check (content_type in ('post', 'comment', 'group', 'profile')),
  -- text, not bigint: a post/comment/group id is a stringified integer, a
  -- profile "id" is a stringified user uuid -- one column handles both
  -- without a second nullable id column and the NULL-uniqueness headache
  -- that would come with it.
  content_id text not null,
  reporter_user_id uuid not null references users(id) on delete cascade,
  reason text not null,
  created_at timestamptz not null default now(),
  reviewed_at timestamptz,
  reviewed_by uuid references users(id) on delete set null,
  resolution text,
  constraint content_reports_unique unique (content_type, content_id, reporter_user_id)
);
create index if not exists content_reports_content_idx on content_reports(content_type, content_id);
create index if not exists content_reports_unreviewed_idx on content_reports(created_at) where reviewed_at is null;
grant select, insert on content_reports to app_user;
grant select, insert, update on content_reports to app_service;

-- SAF-3: automated mention-spike / duplicate-post flags, held for review --
-- never auto-removed, same "flag, don't silently delete" posture as
-- community_ideas' auto-hide-at-threshold.
create table if not exists flagged_content (
  id bigint generated always as identity primary key,
  content_type text not null check (content_type in ('post', 'ticker_mentions')),
  content_id bigint,
  flag_type text not null check (flag_type in ('mention_spike', 'duplicate_post')),
  detail text not null,
  created_at timestamptz not null default now(),
  reviewed_at timestamptz,
  reviewed_by uuid references users(id) on delete set null
);
create index if not exists flagged_content_unreviewed_idx on flagged_content(created_at) where reviewed_at is null;
-- Admin-only (queried via service_conn + require_admin, same as idea_reports'
-- own admin queue) -- no app_user grant, so a regular user can never read
-- the moderation queue even if a bug exposed raw SQL.
grant select, insert, update on flagged_content to app_service;

-- SAF-7: block (mutual visibility/DM) and mute (ticker/keyword/user, feed
-- filtering only -- muting doesn't block DMs or notify the muted party).
create table if not exists user_blocks (
  id bigint generated always as identity primary key,
  blocker_user_id uuid not null references users(id) on delete cascade,
  blocked_user_id uuid not null references users(id) on delete cascade,
  created_at timestamptz not null default now(),
  constraint user_blocks_unique unique (blocker_user_id, blocked_user_id),
  constraint user_blocks_no_self check (blocker_user_id <> blocked_user_id)
);
create index if not exists user_blocks_blocker_idx on user_blocks(blocker_user_id);
create index if not exists user_blocks_blocked_idx on user_blocks(blocked_user_id);
alter table user_blocks enable row level security;
drop policy if exists user_blocks_isolation on user_blocks;
create policy user_blocks_isolation on user_blocks for all
  using (blocker_user_id = current_setting('app.user_id', true)::uuid)
  with check (blocker_user_id = current_setting('app.user_id', true)::uuid);
grant select, insert, delete on user_blocks to app_user;
grant select, insert, delete on user_blocks to app_service;

create table if not exists user_mutes (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  mute_type text not null check (mute_type in ('user', 'ticker', 'keyword')),
  mute_value text not null,
  created_at timestamptz not null default now(),
  constraint user_mutes_unique unique (user_id, mute_type, mute_value)
);
create index if not exists user_mutes_user_idx on user_mutes(user_id);
alter table user_mutes enable row level security;
drop policy if exists user_mutes_isolation on user_mutes;
create policy user_mutes_isolation on user_mutes for all
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);
grant select, insert, delete on user_mutes to app_user;
grant select, insert, delete on user_mutes to app_service;

-- SAF-9: a generalized version of agent_user_settings.compliance_reference
-- -- a place to record a REAL legal sign-off when one happens, for paid
-- strategy subscriptions / copy trading / paid mentoring / performance
-- advertising. This table existing is not a claim that any review has
-- occurred; it starts empty.
create table if not exists compliance_signoffs (
  id bigint generated always as identity primary key,
  feature_key text not null,
  signed_off_by text not null,
  reference text not null,
  signed_off_at timestamptz not null default now()
);
create index if not exists compliance_signoffs_feature_idx on compliance_signoffs(feature_key, signed_off_at desc);
-- Admin-only record-keeping (service_conn + require_admin) -- no app_user grant.
grant select, insert on compliance_signoffs to app_service;
