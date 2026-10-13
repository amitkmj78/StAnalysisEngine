"""SAF-9: a generalized version of trading_agent.py's
EnableUserRequest.compliance_reference -- a place to record a REAL legal
sign-off when one happens, for paid strategy subscriptions, copy trading,
paid mentoring, and public performance advertising.

This router existing, and a row being insertable, is NOT a claim that any
legal review has occurred. No feature in this codebase is currently gated
behind a row here -- STS-8 (creator subscriptions) stays unbuilt until a
real review happens and is recorded.
"""

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from web.backend.admin import require_admin
from web.backend.db import service_conn

router = APIRouter(
    prefix="/api/v1/admin/compliance-signoffs",
    tags=["admin-compliance"],
    dependencies=[Depends(require_admin)],
)


def _record_to_dict(record) -> dict:
    return {k: record[k] for k in record.keys()}


class ComplianceSignoffRequest(BaseModel):
    feature_key: str = Field(min_length=2, max_length=100)
    reference: str = Field(min_length=3, max_length=500)


@router.get("")
async def list_compliance_signoffs():
    async with service_conn() as conn:
        rows = await conn.fetch("SELECT * FROM compliance_signoffs ORDER BY signed_off_at DESC")
    return {"signoffs": [_record_to_dict(r) for r in rows]}


@router.post("")
async def record_compliance_signoff(request: Request, body: ComplianceSignoffRequest):
    admin_email = request.state.user["email"]
    async with service_conn() as conn:
        row = await conn.fetchrow(
            "INSERT INTO compliance_signoffs (feature_key, signed_off_by, reference) VALUES ($1, $2, $3) RETURNING *",
            body.feature_key.strip(), admin_email, body.reference.strip(),
        )
    return _record_to_dict(row)
