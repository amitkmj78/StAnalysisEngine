-- ALX-6: a user's default Stock Finder column set, persisted server-side
-- so it follows them across devices. Saved screens already persist their
-- own visible_columns (saved_screens.visible_columns) -- this is only for
-- the ad-hoc, no-screen-loaded case, which was previously localStorage-only.
create table if not exists user_ui_preferences (
  user_id uuid primary key references users(id) on delete cascade,
  default_stock_finder_columns jsonb,
  updated_at timestamptz not null default now()
);
alter table user_ui_preferences enable row level security;
drop policy if exists user_ui_preferences_isolation on user_ui_preferences;
create policy user_ui_preferences_isolation on user_ui_preferences for all
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

grant select, insert, update on user_ui_preferences to app_user;
