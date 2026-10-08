-- COM-6: the report button. One report per (idea, reporter) -- a repeat
-- click from the same user never inflates the hide-threshold count.
-- No RLS: an admin needs to see every report across every user for the
-- review queue; the router only ever lets a user insert their OWN
-- report, never read others'.
create table if not exists idea_reports (
  id bigint generated always as identity primary key,
  idea_id bigint not null references community_ideas(id) on delete cascade,
  reporter_user_id uuid not null references users(id) on delete cascade,
  reason text not null,
  created_at timestamptz not null default now(),
  constraint idea_reports_unique unique (idea_id, reporter_user_id)
);
create index if not exists idea_reports_idea_idx on idea_reports(idea_id);

grant select, insert on idea_reports to app_user;
grant select, insert on idea_reports to app_service;
