-- Admin-created accounts get a known default password (Welcome1!) instead
-- of the admin having to invent and relay one -- but must_reset_password
-- forces a real password-reset flow (the same one-time-token mechanism
-- forgot-password already uses) before the account can do anything else.
-- Login never issues a session while this is true; reset_password clears it.
alter table users add column if not exists must_reset_password boolean not null default false;
