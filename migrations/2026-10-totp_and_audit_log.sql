-- PLT-6: TOTP-based two-factor login, and a general account-action audit
-- log (login success/failure, 2FA enable/disable, password reset, role
-- changes, deactivate/reactivate) -- distinct from paper_order_audit_log,
-- which is order-lifecycle-only.
alter table users add column if not exists totp_secret_encrypted bytea;
alter table users add column if not exists totp_enabled boolean not null default false;

create table if not exists account_audit_log (
    id bigint generated always as identity primary key,
    user_id uuid not null references users(id) on delete cascade,
    event_type text not null,
    detail jsonb not null default '{}'::jsonb,
    ip_address text,
    created_at timestamptz not null default now()
);
create index if not exists account_audit_log_user_idx on account_audit_log(user_id, created_at desc);
alter table account_audit_log enable row level security;
drop policy if exists account_audit_log_isolation on account_audit_log;
create policy account_audit_log_isolation on account_audit_log
    using (user_id = current_setting('app.user_id', true)::uuid);
grant select on account_audit_log to app_user;
-- Insert-only by grant, same posture as paper_order_audit_log -- no app
-- role is ever granted update/delete on an audit trail.
grant select, insert on account_audit_log to app_service;
