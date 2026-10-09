-- BEG-2: tracks completion + quiz score per lesson in the Learning Paths
-- feature (web/frontend/lib/lessons.ts holds the static lesson content --
-- this table only tracks progress). BEG-3's paper-trading-first gate reuses
-- a row here for the "risk-and-drawdown" lesson as its risk-quiz check,
-- rather than building a second quiz. Safe to re-run.

create table if not exists lesson_progress (
  user_id uuid not null references users(id) on delete cascade,
  lesson_id text not null,
  completed_at timestamptz not null default now(),
  quiz_score smallint not null,
  quiz_total smallint not null,
  primary key (user_id, lesson_id)
);
create index if not exists lesson_progress_user_idx on lesson_progress(user_id);
