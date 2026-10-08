-- ALX-3: web push (standard browser Push API + VAPID) as a third alert
-- delivery channel alongside email and the outbound ALR-3 webhook. A
-- user can have more than one subscription (one per browser/device
-- they've granted permission on), so this is its own table, not a
-- column on user_notification_settings.
create table if not exists push_subscriptions (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  endpoint text not null unique,
  p256dh_key text not null,
  auth_key text not null,
  created_at timestamptz not null default now()
);
create index if not exists push_subscriptions_user_idx on push_subscriptions(user_id);
alter table push_subscriptions enable row level security;
drop policy if exists push_subscriptions_isolation on push_subscriptions;
create policy push_subscriptions_isolation on push_subscriptions for all
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

grant select, insert, delete on push_subscriptions to app_user;
grant select, delete on push_subscriptions to app_service;

-- Per-user opt-in, same ALR-2 default-on posture channel_email/
-- channel_inapp already have.
alter table user_notification_settings add column if not exists push_enabled boolean not null default true;
