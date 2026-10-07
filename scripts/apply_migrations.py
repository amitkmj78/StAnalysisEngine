"""Apply migrations/*.sql to a database, once each, in name order. Same rules as the deploy step
(aws_deploy.py _apply_migrations): applied names are recorded in schema_migrations, and a failing
file stops the run.

Needs a superuser (or owner) connection, because migrations create tables and grant roles.
The URL is read from MIGRATION_DATABASE_URL so the password is never written to a file:

  set MIGRATION_DATABASE_URL=postgresql://postgres:<password>@127.0.0.1:5432/stanalysisengine
  python scripts/apply_migrations.py
"""

import asyncio
import os
import pathlib
import re
import sys

import asyncpg

MIGRATIONS_DIR = pathlib.Path(__file__).resolve().parent.parent / "migrations"
NAME_RE = re.compile(r"^[A-Za-z0-9._-]+\.sql$")


async def main() -> int:
    url = os.environ.get("MIGRATION_DATABASE_URL")
    if not url:
        print("Set MIGRATION_DATABASE_URL to a superuser connection string first.", file=sys.stderr)
        return 2
    files = sorted(p for p in MIGRATIONS_DIR.glob("*.sql"))
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(
            "create table if not exists schema_migrations (name text primary key, applied_at timestamptz not null default now())"
        )
        applied = 0
        for path in files:
            if not NAME_RE.match(path.name):
                raise SystemExit(f"migration file name not allowed: {path.name}")
            done = await conn.fetchval("select 1 from schema_migrations where name = $1", path.name)
            if done:
                print(f"skip   {path.name} (already applied)")
                continue
            sql = path.read_text(encoding="utf-8")
            try:
                async with conn.transaction():
                    await conn.execute(sql)
                    await conn.execute("insert into schema_migrations (name) values ($1)", path.name)
            except Exception as e:
                print(f"FAILED {path.name}: {e}", file=sys.stderr)
                return 1
            applied += 1
            print(f"applied {path.name}")
        print(f"done: {applied} applied, {len(files) - applied} already up to date")
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
