-- Admin-created users with roles: a real role column, alongside (not
-- replacing) the existing hardcoded ADMIN_EMAIL -- every admin check is
-- role='admin' OR email=ADMIN_EMAIL, so a bad role value can never lock out
-- the original account.
alter table users add column if not exists role text not null default 'user'
  check (role in ('user', 'admin'));
update users set role = 'admin' where lower(email) = 'amitkmj78@gmail.com';

-- Theme color preference: a per-user override, NULL meaning "use the
-- admin-configured site default" (app_settings.default_theme_accent_color).
-- Mirrors user_ui_preferences' existing default_stock_finder_columns column
-- -- same table, same RLS, same app_user-only grant.
alter table user_ui_preferences add column if not exists theme_accent_color text;
