import datetime
import decimal
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from web.backend.admin import require_admin
from web.backend.db import crawlsearch_conn, crawlsearch_configured, service_conn
from web.backend.rate_limit import limiter

router = APIRouter(
    prefix="/api/v1/admin/sql",
    tags=["admin-sql"],
    dependencies=[Depends(require_admin)],
)

MAX_ROWS = 500

# Both databases run on the same shared Postgres server but are entirely
# separate schemas/products — this console just lets an admin point the
# same read-only query tool at either one instead of needing a second tool.
DATABASES = {"stanalysisengine": service_conn, "crawlsearch": crawlsearch_conn}


def _conn_factory(database: str):
    if database not in DATABASES:
        raise HTTPException(422, f"database must be one of {list(DATABASES)}")
    if database == "crawlsearch" and not crawlsearch_configured():
        raise HTTPException(503, "CrawlSearch database is not configured on this server.")
    return DATABASES[database]


def _jsonable(value):
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    return value


@router.get("/tables")
async def list_tables(database: str = "stanalysisengine"):
    """Schema browser — lets the admin see what's there before writing a
    query, rather than needing to already know the table/column names."""
    conn_factory = _conn_factory(database)
    async with conn_factory() as conn:
        columns = await conn.fetch(
            """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
            """
        )
        row_counts = await conn.fetch(
            """
            SELECT relname AS table_name, n_live_tup AS approx_row_count
            FROM pg_stat_user_tables
            """
        )

    counts = {r["table_name"]: r["approx_row_count"] for r in row_counts}
    tables: dict[str, dict] = {}
    for c in columns:
        t = tables.setdefault(
            c["table_name"],
            {"table_name": c["table_name"], "approx_row_count": counts.get(c["table_name"]), "columns": []},
        )
        t["columns"].append({"name": c["column_name"], "type": c["data_type"]})

    return {"tables": sorted(tables.values(), key=lambda t: t["table_name"])}


class SqlQueryRequest(BaseModel):
    sql: str
    database: str = "stanalysisengine"


@router.post("/query")
@limiter.limit("30/minute")
async def run_query(request: Request, body: SqlQueryRequest):
    """
    Read-only ad-hoc SQL for the admin. Safety is enforced at the database
    level, not by inspecting the query text (a first-keyword check like
    "must start with SELECT" is trivially bypassed by a CTE such as
    `WITH x AS (DELETE FROM users RETURNING *) SELECT * FROM x`):

    1. The query runs inside a genuine Postgres READ ONLY transaction
       (asyncpg's transaction(readonly=True)) — any write, even one
       smuggled inside a CTE, is rejected by Postgres itself.
    2. It's wrapped as a subquery (`SELECT * FROM (<sql>) AS q`), which
       also only accepts SELECT/WITH-shaped statements syntactically.
    3. The transaction is always rolled back, never committed, and
       service_conn() only grants what app_service already has (no DDL,
       no other users' RLS-protected rows bypassed beyond what admin
       tooling already relies on elsewhere in this app).
    4. Results are capped at MAX_ROWS so one query can't return an
       unbounded response.
    """
    sql = body.sql.strip().rstrip(";")
    if not sql:
        raise HTTPException(422, "Query is empty.")
    if ";" in sql:
        raise HTTPException(422, "Only a single statement is allowed (no semicolons).")

    conn_factory = _conn_factory(body.database)
    wrapped = f"SELECT * FROM ({sql}) AS admin_sql_subquery LIMIT {MAX_ROWS + 1}"

    async with conn_factory() as conn:
        tx = conn.transaction(readonly=True)
        await tx.start()
        try:
            rows = await conn.fetch(wrapped)
        except Exception as e:
            await tx.rollback()
            raise HTTPException(422, f"Query failed: {e}")
        await tx.rollback()  # read-only query — never commit, belt and suspenders

    truncated = len(rows) > MAX_ROWS
    rows = rows[:MAX_ROWS]
    columns = list(rows[0].keys()) if rows else []

    return {
        "columns": columns,
        "rows": [[_jsonable(v) for v in r.values()] for r in rows],
        "row_count": len(rows),
        "truncated": truncated,
    }
