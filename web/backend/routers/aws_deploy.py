import configparser
import io
import json
import os
import secrets
import threading
import time
from datetime import datetime

import boto3
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from web.backend.admin import require_admin
from web.backend.deploy_jobs import cancel_job, finish, get_job, log, new_job

router = APIRouter(prefix="/api/v1/aws-deploy", tags=["aws-deploy"], dependencies=[Depends(require_admin)])

# --- Local-only state. This router runs against the developer's own machine
# (see plan: "local-only, admin-gated dev tool") — never deployed to the EC2
# instance it provisions. AWS credentials live here, not in the app it ships.
_STATE_DIR = os.path.join(os.path.expanduser("~"), ".stanalysisengine")
_CONFIG_PATH = os.path.join(_STATE_DIR, "aws_config.json")

REPO_URL = "https://github.com/amitkmj78/StAnalysisEngine.git"
REMOTE_DIR = "/opt/stanalysisengine"
AMI_FALLBACK = "ami-0866a3c8686eaeeba"  # Ubuntu 24.04 LTS us-east-1, fallback if the live lookup fails

_DEFAULT_CFG = {
    "access_key_id": "",
    "secret_access_key": "",
    "region": "us-east-1",
    "sg_id": "",
    "key_name": "stanalysisengine-key",
}


# ── Config helpers ──────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    if os.path.isfile(_CONFIG_PATH):
        with open(_CONFIG_PATH) as f:
            return {**_DEFAULT_CFG, **json.load(f)}
    # Bootstrap from the AWS CLI's own credentials file, if present.
    creds = configparser.ConfigParser()
    creds.read(os.path.expanduser("~/.aws/credentials"))
    region = "us-east-1"
    try:
        cfg_ini = configparser.ConfigParser()
        cfg_ini.read(os.path.expanduser("~/.aws/config"))
        region = cfg_ini.get("default", "region", fallback="us-east-1")
    except Exception:
        pass
    return {
        **_DEFAULT_CFG,
        "access_key_id": creds.get("default", "aws_access_key_id", fallback=""),
        "secret_access_key": creds.get("default", "aws_secret_access_key", fallback=""),
        "region": region,
    }


def _save_cfg(cfg: dict) -> None:
    os.makedirs(_STATE_DIR, exist_ok=True)
    with open(_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def _boto_kw() -> dict:
    from botocore.config import Config as BotoCfg

    cfg = _load_cfg()
    kw: dict = {
        "region_name": cfg.get("region", "us-east-1"),
        "config": BotoCfg(connect_timeout=5, read_timeout=10, retries={"max_attempts": 1}),
    }
    if cfg.get("access_key_id") and cfg.get("secret_access_key"):
        kw["aws_access_key_id"] = cfg["access_key_id"]
        kw["aws_secret_access_key"] = cfg["secret_access_key"]
    return kw


def _has_credentials() -> bool:
    cfg = _load_cfg()
    return bool(cfg.get("access_key_id") and cfg.get("secret_access_key"))


def _ec2():
    return boto3.client("ec2", **_boto_kw())


def _sts():
    return boto3.client("sts", **_boto_kw())


def _key_name() -> str:
    return _load_cfg().get("key_name", "stanalysisengine-key")


def _pem_path() -> str:
    return os.path.join(_STATE_DIR, f"{_key_name()}.pem")


def _get_ami() -> str:
    try:
        ec2 = _ec2()
        resp = ec2.describe_images(
            Owners=["099720109477"],  # Canonical
            Filters=[
                {"Name": "name", "Values": ["ubuntu/images/hvm-ssd/ubuntu-noble-24.04-amd64-server-*"]},
                {"Name": "state", "Values": ["available"]},
                {"Name": "architecture", "Values": ["x86_64"]},
                {"Name": "virtualization-type", "Values": ["hvm"]},
            ],
        )
        images = sorted(resp.get("Images", []), key=lambda x: x["CreationDate"], reverse=True)
        if images:
            return images[0]["ImageId"]
    except Exception:
        pass
    return AMI_FALLBACK


def _ensure_security_group() -> str:
    ec2 = _ec2()
    cfg = _load_cfg()
    existing = cfg.get("sg_id", "")
    if existing:
        try:
            ec2.describe_security_groups(GroupIds=[existing])
            return existing
        except Exception:
            cfg["sg_id"] = ""

    vpcs = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}]).get("Vpcs", [])
    if not vpcs:
        raise RuntimeError("No default VPC found in this region.")
    vpc_id = vpcs[0]["VpcId"]
    name = "stanalysisengine-sg"
    groups = ec2.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": [name]}, {"Name": "vpc-id", "Values": [vpc_id]}]
    ).get("SecurityGroups", [])
    if groups:
        sg_id = groups[0]["GroupId"]
    else:
        sg_id = ec2.create_security_group(
            GroupName=name, Description="StAnalysisEngine web + SSH access", VpcId=vpc_id
        )["GroupId"]
        ec2.create_tags(Resources=[sg_id], Tags=[{"Key": "Project", "Value": "StAnalysisEngine"}])

    for port in (22, 80):
        try:
            ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[{
                    "IpProtocol": "tcp", "FromPort": port, "ToPort": port,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": f"StAnalysisEngine port {port}"}],
                }],
            )
        except Exception as e:
            if "InvalidPermission.Duplicate" not in str(e):
                raise
    cfg["sg_id"] = sg_id
    _save_cfg(cfg)
    return sg_id


# ── Config endpoints ─────────────────────────────────────────────────────────

class AwsConfigIn(BaseModel):
    access_key_id: str
    secret_access_key: str
    region: str = "us-east-1"
    key_name: str = "stanalysisengine-key"


@router.get("/config")
def get_config():
    cfg = _load_cfg()
    sk = cfg.get("secret_access_key", "")
    return {
        "access_key_id": cfg.get("access_key_id", ""),
        "secret_access_key": ("*" * 8 + sk[-4:]) if len(sk) > 4 else ("*" * len(sk)),
        "region": cfg.get("region", "us-east-1"),
        "key_name": cfg.get("key_name", "stanalysisengine-key"),
        "sg_id": cfg.get("sg_id", ""),
    }


@router.put("/config")
def save_config(req: AwsConfigIn):
    cfg = _load_cfg()
    cfg["access_key_id"] = req.access_key_id.strip()
    if req.secret_access_key and not req.secret_access_key.startswith("*"):
        cfg["secret_access_key"] = req.secret_access_key.strip()
    cfg["region"] = req.region
    cfg["key_name"] = req.key_name.strip()
    _save_cfg(cfg)
    return {"ok": True}


@router.get("/config/verify")
def verify_config():
    if not _has_credentials():
        return {"ok": False, "error": "AWS credentials not configured"}
    try:
        identity = _sts().get_caller_identity()
        sg_id, sg_error = "", ""
        try:
            sg_id = _ensure_security_group()
        except Exception as e:
            sg_error = str(e)[:240]
        return {
            "ok": True, "account": identity["Account"], "arn": identity["Arn"],
            "sg_id": sg_id, "sg_error": sg_error,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}


# ── Status ───────────────────────────────────────────────────────────────────

@router.get("/status")
def status():
    if not _has_credentials():
        return {"key_pair": {"exists": False, "pem_on_server": False}, "instances": [], "ec2_error": "AWS credentials not configured"}

    kn, pem = _key_name(), _pem_path()
    instances, ec2_error, key_exists = [], "", False
    try:
        ec2 = _ec2()
        kp = ec2.describe_key_pairs(Filters=[{"Name": "key-name", "Values": [kn]}])
        key_exists = len(kp["KeyPairs"]) > 0

        r = ec2.describe_instances(Filters=[
            {"Name": "tag:Project", "Values": ["StAnalysisEngine"]},
            {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]},
        ])
        for res in r["Reservations"]:
            for inst in res["Instances"]:
                instances.append({
                    "id": inst["InstanceId"],
                    "type": inst["InstanceType"],
                    "state": inst["State"]["Name"],
                    "public_ip": inst.get("PublicIpAddress", ""),
                    "launched": inst["LaunchTime"].isoformat(),
                })
    except Exception as e:
        ec2_error = str(e)[:200]

    return {
        "key_pair": {"exists": key_exists, "name": kn, "pem_on_server": os.path.isfile(pem)},
        "instances": instances,
        "region": _load_cfg().get("region", "us-east-1"),
        "ec2_error": ec2_error,
    }


# ── Key pair ─────────────────────────────────────────────────────────────────

@router.post("/key-pair")
def create_key_pair():
    ec2 = _ec2()
    kn, pem = _key_name(), _pem_path()
    try:
        ec2.delete_key_pair(KeyName=kn)
    except Exception:
        pass
    kp = ec2.create_key_pair(KeyName=kn, KeyType="rsa")
    os.makedirs(_STATE_DIR, exist_ok=True)
    with open(pem, "w") as f:
        f.write(kp["KeyMaterial"])
    try:
        import stat
        os.chmod(pem, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass
    return {"key_name": kn, "saved_to": pem}


# ── SSH helpers ──────────────────────────────────────────────────────────────

def _load_pkey(pem_path: str, paramiko_mod):
    for key_cls in (paramiko_mod.RSAKey, paramiko_mod.Ed25519Key, paramiko_mod.ECDSAKey):
        try:
            return key_cls.from_private_key_file(pem_path)
        except Exception:
            continue
    raise RuntimeError(f"Could not load private key from {pem_path}")


def _connect_ssh(public_ip: str, username: str, job: dict):
    import paramiko

    pem = _pem_path()
    if not os.path.isfile(pem):
        raise RuntimeError(f"PEM file not found at {pem} — create the key pair first")
    log(job, f"Connecting to {username}@{public_ip}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    pkey = _load_pkey(pem, paramiko)
    client.connect(public_ip, username=username, pkey=pkey, timeout=30)
    log(job, "SSH connected")
    return client


def _ssh_exec(client, cmd: str, timeout: int = 300) -> tuple[str, str, int]:
    """Poll exit_status_ready() instead of paramiko's blocking recv_exit_status(),
    which hangs forever on a wedged remote process — a hard wall-clock timeout instead."""
    _, stdout, stderr = client.exec_command(cmd)
    ch = stdout.channel
    out_parts: list[bytes] = []
    err_parts: list[bytes] = []
    deadline = time.monotonic() + timeout

    while not ch.exit_status_ready():
        if time.monotonic() >= deadline:
            try:
                ch.close()
            except Exception:
                pass
            raise TimeoutError(f"SSH command timed out after {timeout}s: {cmd[:120]}")
        while ch.recv_ready():
            out_parts.append(ch.recv(65536))
        while ch.recv_stderr_ready():
            err_parts.append(ch.recv_stderr(65536))
        time.sleep(0.25)

    while ch.recv_ready():
        out_parts.append(ch.recv(65536))
    while ch.recv_stderr_ready():
        err_parts.append(ch.recv_stderr(65536))

    out = b"".join(out_parts).decode("utf-8", errors="replace")
    err = b"".join(err_parts).decode("utf-8", errors="replace")
    return out, err, ch.recv_exit_status()


# ── EC2 launch ───────────────────────────────────────────────────────────────

_USER_DATA = """#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive
if [ ! -f /swapfile ]; then
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
apt-get update -y
apt-get install -y python3-venv python3-pip nginx git curl postgresql
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
mkdir -p /opt/stanalysisengine
chown ubuntu:ubuntu /opt/stanalysisengine
echo "SETUP_DONE" > /tmp/stanalysisengine_setup_done
"""


class LaunchRequest(BaseModel):
    instance_type: str = "t3.small"
    volume_size_gb: int = 20


def _worker_launch(job_id: str, req: LaunchRequest) -> None:
    job = get_job(job_id)
    try:
        ec2 = _ec2()
        kn, sg = _key_name(), _ensure_security_group()
        ami = _get_ami()
        log(job, f"Launching {req.instance_type} (AMI: {ami}, volume: {req.volume_size_gb}GB)")
        result = ec2.run_instances(
            ImageId=ami, InstanceType=req.instance_type,
            KeyName=kn, SecurityGroupIds=[sg],
            MinCount=1, MaxCount=1, UserData=_USER_DATA,
            BlockDeviceMappings=[{
                "DeviceName": "/dev/sda1",
                "Ebs": {"VolumeSize": max(8, req.volume_size_gb), "VolumeType": "gp3", "DeleteOnTermination": True},
            }],
            TagSpecifications=[{"ResourceType": "instance", "Tags": [
                {"Key": "Name", "Value": "StAnalysisEngine"},
                {"Key": "Project", "Value": "StAnalysisEngine"},
            ]}],
        )
        iid = result["Instances"][0]["InstanceId"]
        log(job, f"✓ Instance created: {iid}")
        log(job, "Waiting for running state (up to ~3 min)...")
        ec2.get_waiter("instance_running").wait(InstanceIds=[iid], WaiterConfig={"Delay": 8, "MaxAttempts": 23})
        desc = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]
        ip = desc.get("PublicIpAddress", "N/A")
        log(job, "✓ Running!")
        log(job, f"  Instance ID : {iid}")
        log(job, f"  Public IP   : {ip}")
        log(job, "Server is now installing nginx/postgres/node via user-data (~2 min) — wait before deploying.")
        finish(job, True)
    except Exception as e:
        log(job, f"✗ {e}")
        finish(job, False)


@router.post("/ec2/launch")
def launch_ec2(req: LaunchRequest):
    if not _has_credentials():
        raise HTTPException(400, "AWS credentials not configured")
    kn = _key_name()
    if not _ec2().describe_key_pairs(Filters=[{"Name": "key-name", "Values": [kn]}])["KeyPairs"]:
        raise HTTPException(400, f"Key pair '{kn}' does not exist — create it first")
    job_id, _ = new_job(f"Launch EC2 {req.instance_type}")
    threading.Thread(target=_worker_launch, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id}


# ── Deploy ───────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """\
create extension if not exists pgcrypto;

create table if not exists users (
  id uuid primary key default gen_random_uuid(),
  email text unique not null,
  password_hash text not null,
  approved boolean not null default false,
  created_at timestamptz not null default now()
);
-- Deactivation is a reversible, admin-triggered suspension distinct from
-- delete: it blocks future logins (checked in /login) and, via
-- session_invalidated_at below, kills any already-open session too —
-- keeps the account and all its data/relationships intact.
alter table users add column if not exists is_active boolean not null default true;
-- NULL means "no revocation in effect" — a token issued (iat) before this
-- timestamp is rejected by verify_bearer_token regardless of its own
-- expiry, so deactivating a user or force-logging them out takes effect
-- on their already-open tabs within seconds (see the short-TTL cache in
-- web/backend/auth.py), not just on their next fresh login.
alter table users add column if not exists session_invalidated_at timestamptz;
-- NULL means "use the admin-configured global default"
-- (app_settings.portfolio_drop_threshold_pct) — most users never touch
-- this, only set once a user opts into a tighter/looser sensitivity
-- than the shared default from the Portfolio page.
alter table users add column if not exists drop_alert_threshold_pct real;
-- Set on every successful /login (and the admin-bootstrap auto-login in
-- /signup) — see web/backend/routers/auth.py. last_login_ip is read from
-- X-Forwarded-For/X-Real-IP (this app sits behind nginx, so
-- request.client.host alone is just nginx's own address), falling back
-- to request.client.host only if neither header is present.
alter table users add column if not exists last_login_at timestamptz;
alter table users add column if not exists last_login_ip text;

-- Forgot-password: only a sha256 hash of the token is ever stored, never
-- the token itself, so a DB read can't be used to reset an account's
-- password. No RLS — this table is only ever touched via service_conn
-- (issuing/consuming a reset token happens before a session exists), not
-- user-scoped app_user queries.
create table if not exists password_reset_tokens (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  token_hash text unique not null,
  expires_at timestamptz not null,
  used_at timestamptz,
  created_at timestamptz not null default now()
);
create index if not exists password_reset_tokens_user_idx on password_reset_tokens(user_id);

create table if not exists trades (
  trade_id text primary key,
  user_id uuid not null references users(id) on delete cascade,
  ticker text not null, direction text, strategy_type text,
  created_at timestamptz not null default now(),
  entry_low real, entry_high real, stop_loss real, target real,
  context text, risk_profile text, risk_factor real, status text default 'OPEN',
  entry_price real, entry_date timestamptz, exit_price real, exit_date timestamptz,
  max_runup_pct real, max_drawdown_pct real, realized_pnl_pct real, days_in_trade real
);
create index if not exists trades_user_idx on trades(user_id);
alter table trades enable row level security;
drop policy if exists trades_isolation on trades;
create policy trades_isolation on trades
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

-- A user can have more than one portfolio (e.g. "Retirement" vs "Trading");
-- portfolio_positions/portfolio_strategies/watchlist_alerts below each get
-- a portfolio_id pointing here.
create table if not exists portfolios (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  name text not null default 'My Portfolio',
  created_at timestamptz not null default now()
);
create index if not exists portfolios_user_idx on portfolios(user_id);
alter table portfolios enable row level security;
drop policy if exists portfolios_isolation on portfolios;
create policy portfolios_isolation on portfolios
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);
-- Same reversible-suspension shape as users.is_active: an admin can hide
-- one specific portfolio (blocked from _resolve_portfolio_id, dropped
-- from GET /list) without deleting its positions/strategies/alerts, distinct
-- from the admin DELETE below which removes it and all of that permanently.
alter table portfolios add column if not exists is_active boolean not null default true;
-- User-controlled, distinct from is_active above: lets someone silence
-- drop-alert scanning for just this one portfolio (e.g. a buy-and-forget
-- retirement account) while keeping it fully visible/active everywhere
-- else. Checked in portfolio_alerts.scan_portfolios_for_drops.
alter table portfolios add column if not exists drop_alerts_enabled boolean not null default true;
-- Money borrowed from the broker against this portfolio (0 = no margin
-- used / cash account). A liability, not a position -- total market
-- value (sum of what's held) minus this is Net Equity, the standard
-- brokerage distinction between account value and what you'd actually
-- walk away with after paying the loan back. User-edited directly, not
-- derived from anything else in this schema.
alter table portfolios add column if not exists margin_balance real not null default 0;
-- Uninvested cash sitting in this portfolio (0 = none). An asset, the
-- mirror image of margin_balance above -- added on top of position market
-- value for Total Value and Net Equity, but deliberately excluded from
-- gain-vs-cost/benchmark-comparison return percentages (idle cash has no
-- cost basis; including it would understate the real return on what's
-- actually invested). User-edited directly, not derived from anything
-- else in this schema.
alter table portfolios add column if not exists cash_balance real not null default 0;

-- Diversified-basket generation metadata -- only populated for
-- portfolios created via "Build a Diversified Basket"; NULL/default for
-- every other portfolio, no behavior change for them.
-- basket_target_weights is refreshed on every generation AND on every
-- applied rebalance (see basket_rebalance_alerts below), so it always
-- reflects "what the weights are supposed to be right now."
alter table portfolios add column if not exists basket_universe text;
alter table portfolios add column if not exists basket_goal text;
alter table portfolios add column if not exists basket_score_as_of date;
alter table portfolios add column if not exists basket_generation_inputs jsonb not null default '{}'::jsonb;
alter table portfolios add column if not exists basket_target_weights jsonb not null default '{}'::jsonb;
alter table portfolios add column if not exists rebalance_frequency text not null default 'none';
alter table portfolios add column if not exists drift_threshold_pct real not null default 5.0;
alter table portfolios add column if not exists last_rebalance_checked_at timestamptz;
do $$
begin
  if not exists (
    select 1 from pg_constraint where conname = 'portfolios_rebalance_frequency_check'
  ) then
    alter table portfolios add constraint portfolios_rebalance_frequency_check
      check (rebalance_frequency in ('none', 'monthly', 'quarterly')) not valid;
  end if;
end $$;

-- HLT-4: tax-loss harvesting is only shown for taxable accounts. Reuses
-- the exact Taxable/Traditional/Roth taxonomy services/million_plan_
-- service.py::ACCOUNT_TYPES already established for strategy_plans, for
-- consistency -- not a new enum. Defaults to 'Taxable' (the more
-- conservative default for an observational feature: a false positive
-- just shows a candidate list on an account that isn't really taxable,
-- while defaulting to a non-taxable type would silently hide a real
-- insight on an account that is).
alter table portfolios add column if not exists account_type text not null default 'Taxable';
do $$
begin
  if not exists (
    select 1 from pg_constraint where conname = 'portfolios_account_type_check'
  ) then
    alter table portfolios add constraint portfolios_account_type_check
      check (account_type in ('Taxable', 'Traditional', 'Roth')) not valid;
  end if;
end $$;

create table if not exists portfolio_positions (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  ticker text, name text, shares real, avg_cost real, current_price real,
  unrealized_pnl_pct real, source text, created_at timestamptz not null default now()
);
create index if not exists portfolio_positions_user_idx on portfolio_positions(user_id);

-- created_at tracks row-insert time, which is NOT a stable "date this
-- position was acquired": every save (including editing one unrelated
-- position) deletes and reinserts every row for the whole portfolio
-- (see _save_and_respond in web/backend/routers/portfolio.py), which
-- would otherwise reset created_at to now() for every untouched
-- position too. acquired_at is a separate, deliberately-preserved date
-- -- set once (from a CSV's real earliest-buy date when available, else
-- today, or a user-supplied date on manual entry) and carried forward
-- by _merge_with_existing on every subsequent save, so it actually
-- means "when this position was first added," not "when this row last
-- happened to be rewritten."
alter table portfolio_positions add column if not exists acquired_at date not null default current_date;

alter table portfolio_positions enable row level security;
drop policy if exists portfolio_positions_isolation on portfolio_positions;
create policy portfolio_positions_isolation on portfolio_positions
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

create table if not exists portfolio_strategies (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  ticker text, shares real, avg_cost real, current_price real, unrealized_pnl_pct real,
  short_term_plan text, long_term_plan text, risk_profile text, risk_factor integer,
  created_at timestamptz not null default now()
);
create index if not exists portfolio_strategies_user_idx on portfolio_strategies(user_id);
alter table portfolio_strategies enable row level security;
drop policy if exists portfolio_strategies_isolation on portfolio_strategies;
create policy portfolio_strategies_isolation on portfolio_strategies
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

-- One row per user/ticker/day a same-day drop was detected — the unique
-- constraint is what keeps the scheduler from re-running the expensive
-- sentiment+LLM analysis (and re-notifying) on every scan tick.
create table if not exists portfolio_drop_alerts (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  ticker text not null,
  alert_date date not null,
  prev_close real not null,
  price_at_check real not null,
  pct_change real not null,
  sentiment_summary text,
  predicted_signal text,
  predicted_expected_return_pct real,
  predicted_target_price real,
  recommended_action text,
  created_at timestamptz not null default now(),
  seen_at timestamptz,
  unique (user_id, ticker, alert_date)
);
alter table portfolio_drop_alerts add column if not exists updated_at timestamptz;
create index if not exists portfolio_drop_alerts_user_idx on portfolio_drop_alerts(user_id, created_at desc);
alter table portfolio_drop_alerts enable row level security;
drop policy if exists portfolio_drop_alerts_isolation on portfolio_drop_alerts;
create policy portfolio_drop_alerts_isolation on portfolio_drop_alerts
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

-- Scheduled rebalance-check output for a diversified basket: a review-
-- and-act alert, never auto-applied. One row per portfolio per
-- check_date -- ON CONFLICT DO UPDATE refreshes it in place on a
-- same-day re-run, same dedup shape as portfolio_drop_alerts above.
create table if not exists basket_rebalance_alerts (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  portfolio_id bigint not null references portfolios(id) on delete cascade,
  check_date date not null,
  score_as_of date not null,
  drift_summary jsonb not null default '[]'::jsonb,
  suggested_swaps jsonb not null default '[]'::jsonb,
  target_weights jsonb not null default '{}'::jsonb,
  max_drift_pct real not null,
  status text not null default 'pending',
  applied_at timestamptz,
  seen_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz,
  unique (portfolio_id, check_date)
);
do $$
begin
  if not exists (
    select 1 from pg_constraint where conname = 'basket_rebalance_alerts_status_check'
  ) then
    alter table basket_rebalance_alerts add constraint basket_rebalance_alerts_status_check
      check (status in ('pending', 'applied', 'dismissed')) not valid;
  end if;
end $$;
create index if not exists basket_rebalance_alerts_user_idx on basket_rebalance_alerts(user_id, created_at desc);
alter table basket_rebalance_alerts enable row level security;
drop policy if exists basket_rebalance_alerts_isolation on basket_rebalance_alerts;
create policy basket_rebalance_alerts_isolation on basket_rebalance_alerts
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

-- Saved goals from the Strategies calculator. monthly_contribution/
-- annual_return_pct are whatever the plan resolved to when saved (the
-- solved figure for the active mode, or the given input for the others),
-- locked in at that point, not recomputed later -- progress tracking
-- compounds this same fixed contribution (stepping up by
-- annual_contribution_increase_pct once every 12 months) forward from
-- created_at and compares it against the user's live portfolio value, so
-- "ahead/behind pace" means "vs. what you'd have if you'd contributed this
-- amount every month since saving." account_type/inflation_pct are stored
-- for display only (what assumptions this goal was saved under).
create table if not exists strategy_plans (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  name text,
  target_amount real not null,
  years real not null,
  starting_capital real not null,
  annual_return_pct real not null,
  monthly_contribution real not null,
  annual_contribution_increase_pct real not null default 0,
  account_type text not null default 'Taxable',
  inflation_pct real not null default 2.5,
  created_at timestamptz not null default now()
);
create index if not exists strategy_plans_user_idx on strategy_plans(user_id, created_at desc);
alter table strategy_plans enable row level security;
drop policy if exists strategy_plans_isolation on strategy_plans;
create policy strategy_plans_isolation on strategy_plans
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

create table if not exists request_log (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  endpoint text not null,
  created_at timestamptz not null default now()
);
create index if not exists request_log_user_created_idx on request_log(user_id, created_at desc);

create table if not exists saved_predictions (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  ticker text not null,
  period text not null,
  predicted_at timestamptz not null default now(),
  last_close real,
  next_price real,
  signal text,
  expected_return_pct real,
  target_price real,
  target_date timestamptz,
  actual_next_price real,
  actual_target_price real,
  actual_target_open real,
  next_price_error_pct real,
  target_price_error_pct real,
  signal_correct boolean,
  verified_at timestamptz
);
create index if not exists saved_predictions_user_idx on saved_predictions(user_id, ticker, predicted_at desc);
alter table saved_predictions enable row level security;
drop policy if exists saved_predictions_isolation on saved_predictions;
create policy saved_predictions_isolation on saved_predictions for all
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

create table if not exists saved_narratives (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  ticker text not null,
  provider text not null,
  period text not null,
  days_ahead integer not null,
  narrative text not null,
  sentiment_context text not null,
  saved_at timestamptz not null default now()
);
create index if not exists saved_narratives_user_idx on saved_narratives(user_id, ticker, saved_at desc);
alter table saved_narratives enable row level security;
drop policy if exists saved_narratives_isolation on saved_narratives;
create policy saved_narratives_isolation on saved_narratives for all
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

create table if not exists saved_baseline_snapshots (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  ticker text not null,
  horizon_days integer not null,
  confidence real not null,
  method text not null,
  as_of date not null,
  last_price real not null,
  floor real not null,
  floor_pct real not null,
  accumulation_zone_hi real not null,
  accumulation_zone_hi_pct real not null,
  median_path real not null,
  distribution_zone_lo real not null,
  distribution_zone_lo_pct real not null,
  ceiling real not null,
  ceiling_pct real not null,
  samples integer not null,
  effective_samples integer not null,
  breach_rate_full real not null,
  saved_at timestamptz not null default now()
);
create index if not exists saved_baseline_snapshots_user_idx on saved_baseline_snapshots(user_id, ticker, saved_at desc);
alter table saved_baseline_snapshots enable row level security;
drop policy if exists saved_baseline_snapshots_isolation on saved_baseline_snapshots;
create policy saved_baseline_snapshots_isolation on saved_baseline_snapshots for all
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

create table if not exists saved_screens (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  name text not null,
  goal text not null,
  universe text not null,
  filters jsonb not null default '{}'::jsonb,
  visible_columns jsonb not null default '[]'::jsonb,
  sort_keys jsonb not null default '[]'::jsonb,
  snapshot_top10 jsonb not null default '[]'::jsonb,
  saved_at timestamptz not null default now()
);
create index if not exists saved_screens_user_idx on saved_screens(user_id, saved_at desc);
alter table saved_screens enable row level security;
drop policy if exists saved_screens_isolation on saved_screens;
create policy saved_screens_isolation on saved_screens for all
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

-- STR-2: custom stress-test scenarios the user builds and reruns. Mirrors
-- saved_screens' shape/RLS exactly -- rerun is entirely client-side (load
-- shock_config into the builder form, call the same compute endpoint a
-- brand-new scenario would use), same precedent as stock-finder's saved
-- screens. Named saved_stress_scenarios, not saved_scenarios -- this app
-- already uses "scenario" for goal-plan-solver modes elsewhere.
create table if not exists saved_stress_scenarios (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  name text not null,
  shock_config jsonb not null default '[]'::jsonb,
  saved_at timestamptz not null default now()
);
create index if not exists saved_stress_scenarios_user_idx on saved_stress_scenarios(user_id, saved_at desc);
alter table saved_stress_scenarios enable row level security;
drop policy if exists saved_stress_scenarios_isolation on saved_stress_scenarios;
create policy saved_stress_scenarios_isolation on saved_stress_scenarios for all
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

-- SCN-3: one row per saved screen per day it's checked, written by the
-- nightly scan (see services/saved_screen_alert_service.py). `membership`
-- is the full set of tickers matching that screen as of check_date --
-- storing the whole set (not just top-10, unlike snapshot_top10 above)
-- means the NEXT day's enter/leave diff is just "most recent row's
-- membership", no mutation of saved_screens itself needed.
create table if not exists saved_screen_alerts (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  screen_id bigint not null references saved_screens(id) on delete cascade,
  check_date date not null,
  entered jsonb not null default '[]'::jsonb,
  left_tickers jsonb not null default '[]'::jsonb,
  membership jsonb not null default '[]'::jsonb,
  emailed_at timestamptz,
  created_at timestamptz not null default now(),
  unique (screen_id, check_date)
);
create index if not exists saved_screen_alerts_screen_idx on saved_screen_alerts(screen_id, check_date desc);
create index if not exists saved_screen_alerts_user_idx on saved_screen_alerts(user_id, created_at desc);
alter table saved_screen_alerts enable row level security;
drop policy if exists saved_screen_alerts_isolation on saved_screen_alerts;
create policy saved_screen_alerts_isolation on saved_screen_alerts for all
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

-- A saved GET /portfolio/goal-plan request — same inputs (target_amount,
-- target_date, optional monthly_amount/compare_universe) re-run live
-- against current prices/signals each time it's loaded, not a frozen
-- snapshot of the plan itself.
create table if not exists saved_portfolio_goals (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  portfolio_id bigint not null references portfolios(id) on delete cascade,
  name text not null default 'Goal',
  target_amount real not null,
  target_date date not null,
  monthly_amount real,
  compare_universe text,
  created_at timestamptz not null default now()
);
create index if not exists saved_portfolio_goals_user_idx on saved_portfolio_goals(user_id, created_at desc);
alter table saved_portfolio_goals enable row level security;
drop policy if exists saved_portfolio_goals_isolation on saved_portfolio_goals;
create policy saved_portfolio_goals_isolation on saved_portfolio_goals for all
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

-- A saved GET /monthly-plan/summary form (both the fund and stock side of
-- the page share one set of inputs, run together by the same "Build Plan"
-- click) — same inputs re-run live against current prices/rankings each
-- time it's loaded, not a frozen snapshot of the plan itself, same
-- rationale as saved_portfolio_goals above.
create table if not exists saved_monthly_plans (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  name text not null default 'Monthly Plan',
  monthly_amount real not null,
  years integer not null,
  fund_goal text not null,
  fund_category text not null,
  stock_goal text not null,
  stock_universe text not null,
  created_at timestamptz not null default now()
);
create index if not exists saved_monthly_plans_user_idx on saved_monthly_plans(user_id, created_at desc);
alter table saved_monthly_plans enable row level security;
drop policy if exists saved_monthly_plans_isolation on saved_monthly_plans;
create policy saved_monthly_plans_isolation on saved_monthly_plans for all
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

-- Portfolio Compare's "Current Signals Across Your Positions" table is
-- expensive to compute (a model trained per ticker per horizon), so it's
-- snapshotted by US trading day rather than recomputed on every page
-- load — one row per (user, portfolio, day), overwritten by an explicit
-- refresh rather than accumulating history.
create table if not exists portfolio_insights_snapshots (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  portfolio_id bigint not null references portfolios(id) on delete cascade,
  as_of_date date not null,
  positions jsonb not null,
  concentration_threshold_pct real,
  predict_period text,
  predict_days_ahead integer,
  lookback_days integer,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (user_id, portfolio_id, as_of_date)
);
create index if not exists portfolio_insights_snapshots_lookup_idx
  on portfolio_insights_snapshots(user_id, portfolio_id, as_of_date desc);
alter table portfolio_insights_snapshots enable row level security;
drop policy if exists portfolio_insights_snapshots_isolation on portfolio_insights_snapshots;
create policy portfolio_insights_snapshots_isolation on portfolio_insights_snapshots for all
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

create table if not exists watchlist_alerts (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  ticker text not null,
  condition_type text not null,
  threshold real not null,
  created_at timestamptz not null default now(),
  active boolean not null default true,
  triggered_at timestamptz,
  triggered_price real,
  seen_at timestamptz,
  source text
);
alter table watchlist_alerts add column if not exists source text;
create index if not exists watchlist_alerts_user_idx on watchlist_alerts(user_id, created_at desc);
alter table watchlist_alerts enable row level security;
drop policy if exists watchlist_alerts_isolation on watchlist_alerts;
create policy watchlist_alerts_isolation on watchlist_alerts for all
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

-- Multi-portfolio migration: add portfolio_id to portfolio_positions,
-- portfolio_strategies, and watchlist_alerts (whose portfolio_auto rows
-- are recreated per-save and would otherwise get wiped across
-- portfolios), backfill every existing row into a per-user "My
-- Portfolio", then lock the two position tables to NOT NULL now that
-- every row has one. Self-terminating: once a user's rows are
-- backfilled, `where portfolio_id is null` finds nothing left for them
-- on the next deploy, so this is safe to leave in place permanently.
alter table portfolio_positions add column if not exists portfolio_id bigint references portfolios(id) on delete cascade;
alter table portfolio_strategies add column if not exists portfolio_id bigint references portfolios(id) on delete cascade;
alter table watchlist_alerts add column if not exists portfolio_id bigint references portfolios(id) on delete cascade;

do $$
declare
  r record;
  new_portfolio_id bigint;
begin
  for r in
    select distinct user_id from portfolio_positions where portfolio_id is null
    union
    select distinct user_id from portfolio_strategies where portfolio_id is null
  loop
    insert into portfolios (user_id, name) values (r.user_id, 'My Portfolio') returning id into new_portfolio_id;
    update portfolio_positions set portfolio_id = new_portfolio_id where user_id = r.user_id and portfolio_id is null;
    update portfolio_strategies set portfolio_id = new_portfolio_id where user_id = r.user_id and portfolio_id is null;
    update watchlist_alerts set portfolio_id = new_portfolio_id where user_id = r.user_id and portfolio_id is null and source = 'portfolio_auto';
  end loop;
end $$;

alter table portfolio_positions alter column portfolio_id set not null;
alter table portfolio_strategies alter column portfolio_id set not null;
-- watchlist_alerts.portfolio_id stays nullable — manually-created alerts
-- (source is not 'portfolio_auto') aren't tied to any portfolio.

create index if not exists portfolio_positions_portfolio_idx on portfolio_positions(portfolio_id);
create index if not exists portfolio_strategies_portfolio_idx on portfolio_strategies(portfolio_id);

create table if not exists app_settings (
  key text primary key,
  value text not null,
  updated_at timestamptz not null default now()
);
insert into app_settings (key, value) values ('verify_predictions_enabled', 'true')
  on conflict (key) do nothing;
insert into app_settings (key, value) values ('publish_signals_enabled', 'false')
  on conflict (key) do nothing;
insert into app_settings (key, value) values ('password_policy_enabled', 'true')
  on conflict (key) do nothing;
insert into app_settings (key, value) values ('pit_price_capture_enabled', 'true')
  on conflict (key) do nothing;
insert into app_settings (key, value) values ('pit_analyst_rating_capture_enabled', 'true')
  on conflict (key) do nothing;
insert into app_settings (key, value) values ('pit_quant_signal_capture_enabled', 'true')
  on conflict (key) do nothing;
insert into app_settings (key, value) values ('portfolio_drop_alerts_enabled', 'false')
  on conflict (key) do nothing;
insert into app_settings (key, value) values ('portfolio_drop_threshold_pct', '1.0')
  on conflict (key) do nothing;
insert into app_settings (key, value) values ('daily_quota', '600')
  on conflict (key) do nothing;
insert into app_settings (key, value) values ('db_backup_enabled', 'true')
  on conflict (key) do nothing;

-- Public track-record ledger (TR-1/TR-2). Not user-scoped, no RLS: this is
-- deliberately a public record, not private data. Rows are never updated or
-- deleted by the app — corrections are new rows with reason_code/corrects_id
-- set, so the append-only history stays intact.
create table if not exists published_signals (
  id bigint generated always as identity primary key,
  published_at_utc timestamptz not null default now(),
  model_version_hash text not null,
  as_of_data_timestamp timestamptz not null,
  target_date date not null,
  universe_id text not null,
  lookback_days integer not null,
  rank integer not null,
  ticker text not null,
  trailing_return_pct real not null,
  data_source text not null default 'live',
  reason_code text,
  corrects_id bigint references published_signals(id)
);
alter table published_signals add column if not exists data_source text not null default 'live';
create index if not exists published_signals_lookup_idx
  on published_signals(target_date, universe_id, lookback_days);

-- TR-4: realized outcomes for already-published signals, once enough
-- trading days have elapsed to know them. Deliberately a separate table
-- from published_signals (and from anything backtest-derived) — live and
-- backtested results must never be combinable in any query or output.
create table if not exists signal_outcomes (
  id bigint generated always as identity primary key,
  evaluated_at_utc timestamptz not null default now(),
  target_date date not null,
  universe_id text not null,
  lookback_days integer not null,
  horizon_days integer not null,
  ticker text not null,
  rank integer not null,
  entry_price real not null,
  exit_price real not null,
  realized_return_pct real not null,
  benchmark_return_pct real not null,
  beat_benchmark boolean not null,
  unique (target_date, universe_id, lookback_days, horizon_days, ticker)
);
create index if not exists signal_outcomes_lookup_idx
  on signal_outcomes(target_date, universe_id, lookback_days, horizon_days);

-- TR-7: every backtest run persists its full parameter set and result,
-- retrievable forever by id — not user-scoped, no RLS, since a backtest
-- configuration/result isn't private data (same treatment as
-- published_signals/signal_outcomes).
create table if not exists backtest_runs (
  id bigint generated always as identity primary key,
  requested_at_utc timestamptz not null default now(),
  asset_type text not null,
  universe text not null,
  lookback_days integer not null,
  top_n integer not null,
  years integer not null,
  horizon_days integer not null default 30,
  slippage_bps real not null,
  commission_bps real not null,
  borrow_cost_bps_annual real not null,
  risk_free_rate_annual real not null,
  result_json jsonb not null
);
alter table backtest_runs add column if not exists horizon_days integer not null default 30;
create index if not exists backtest_runs_lookup_idx
  on backtest_runs(asset_type, universe, lookback_days, top_n, years, horizon_days);

-- NFR-03: one row per backup attempt for the published-record + PIT-store
-- tables. structural_check_passed comes free with every backup (parses
-- the dump's own table of contents, no DB needed); restore_test_passed
-- is only set once an actual restore-into-a-throwaway-database test has
-- run against that specific backup (manually or on the quarterly
-- schedule) — the real "restore-tested" guarantee, not just a file
-- integrity check.
create table if not exists backup_runs (
  id bigint generated always as identity primary key,
  started_at_utc timestamptz not null default now(),
  s3_key text,
  size_bytes bigint,
  tables_verified text[],
  structural_check_passed boolean not null default false,
  restore_test_run boolean not null default false,
  restore_test_passed boolean,
  restore_test_row_counts jsonb,
  error text
);
create index if not exists backup_runs_started_idx on backup_runs(started_at_utc desc);

-- TR-3 Phase 1: append-only point-in-time price store. A row's mere
-- presence proves this exact close was on record at captured_at_utc — the
-- ON CONFLICT DO NOTHING below means a row is never overwritten once
-- captured, so later data-vendor revisions (split/dividend reprocessing,
-- corrections) can never quietly rewrite history out from under it. Not
-- user-scoped, no RLS: internal engine data, same as backtest_runs.
create table if not exists pit_prices (
  id bigint generated always as identity primary key,
  ticker text not null,
  price_date date not null,
  close real not null,
  captured_at_utc timestamptz not null default now(),
  source text not null default 'yfinance',
  unique (ticker, price_date)
);
create index if not exists pit_prices_ticker_date_idx on pit_prices(ticker, price_date desc);

-- TR-3 Phase 2: point-in-time universe membership snapshots. INDEX_MAP /
-- INDEX_FUND_UNIVERSE in code today are static — a delisted, merged, or
-- renamed ticker just vanishes with no record. This starts an honest
-- going-forward history; it cannot backfill what already changed before
-- capture began.
create table if not exists pit_universe_membership (
  id bigint generated always as identity primary key,
  asset_type text not null,
  universe_key text not null,
  ticker text not null,
  snapshot_date date not null,
  captured_at_utc timestamptz not null default now(),
  unique (asset_type, universe_key, ticker, snapshot_date)
);
create index if not exists pit_universe_membership_lookup_idx
  on pit_universe_membership(asset_type, universe_key, snapshot_date desc);

-- TR-3 Phase 3: point-in-time fundamentals for the "Long Term" composite
-- score's fundamental inputs (see stock_finder_service.GOAL_WEIGHTS) — the
-- missing piece blocking an honest walk-forward backtest of Best Stock
-- Finder / Best Index Fund's Long Term ranking, which today can only ever
-- see today's fundamentals no matter what historical date it's asked about.
create table if not exists pit_fundamentals (
  id bigint generated always as identity primary key,
  ticker text not null,
  as_of_date date not null,
  forward_pe real,
  revenue_growth_pct real,
  earnings_growth_pct real,
  captured_at_utc timestamptz not null default now(),
  source text not null default 'yfinance',
  unique (ticker, as_of_date)
);
-- Raw yfinance sector string, added alongside Phase 1's stock-score work
-- so its capture (already fetching .info per ticker here) can supply
-- sector labels without a second full-universe pass. Nullable/backfilled
-- going forward only -- past rows keep sector=null, same as every other
-- additive PIT column in this file.
alter table pit_fundamentals add column if not exists sector text;

-- SCR-1's Quality factor: return_on_equity_pct/profit_margin_pct,
-- captured off the same .info call as everything else in this table (no
-- new fetch). services/stock_score_capture_service.py's blend_quality
-- averages whichever of these two is present into one raw value.
-- Debt-to-equity is deliberately excluded from v1 -- it's lower-is-
-- better, and averaging it in pre-percentile with these two higher-is-
-- better metrics would distort the blend; that needs a percentile-then-
-- blend design instead, out of scope here. Nullable/backfilled going
-- forward only, same convention as sector above.
alter table pit_fundamentals add column if not exists return_on_equity_pct real;
alter table pit_fundamentals add column if not exists profit_margin_pct real;
create index if not exists pit_fundamentals_ticker_date_idx on pit_fundamentals(ticker, as_of_date desc);

-- Point-in-time capture of the same Quant Signal shown on /predict and
-- the Stock Screener — one row per ticker per day, so day-over-day
-- comparison ("did the model's call on this ticker change?") is possible
-- without a live single-ticker call. Not part of the TR-3 backtest chain
-- (services/pit_signal_service.py's PIT ranking is separate and already
-- exists) — this is a plain historical log, not a scoring input.
create table if not exists pit_quant_signal (
  id bigint generated always as identity primary key,
  ticker text not null,
  as_of_date date not null,
  signal text not null,
  expected_return_pct real not null,
  target_price real not null,
  last_close real not null,
  captured_at_utc timestamptz not null default now(),
  source text not null default 'internal-model',
  unique (ticker, as_of_date)
);
create index if not exists pit_quant_signal_ticker_date_idx on pit_quant_signal(ticker, as_of_date desc);

-- The live, out-of-sample counterpart to services/quant_signal_backtest_
-- service.py's simulated walk-forward: one row per Quant Signal call
-- (pit_quant_signal) once horizon_days trading days have actually
-- elapsed and a real exit price (pit_prices) is on record, recording
-- whether the BUY/HOLD/SELL call was actually right. Append-only like
-- every other PIT table — a row's outcome never changes once computed.
create table if not exists quant_signal_outcomes (
  id bigint generated always as identity primary key,
  ticker text not null,
  as_of_date date not null,
  signal text not null,
  expected_return_pct real not null,
  entry_price real not null,
  horizon_days integer not null,
  exit_date date not null,
  exit_price real not null,
  realized_return_pct real not null,
  correct boolean not null,
  evaluated_at_utc timestamptz not null default now(),
  unique (ticker, as_of_date, horizon_days)
);
create index if not exists quant_signal_outcomes_ticker_date_idx on quant_signal_outcomes(ticker, as_of_date desc);

-- Point-in-time capture of the same real, third-party analyst consensus
-- shown on the Stock Screener's "Analyst Rating" column — one row per
-- ticker per day (only for tickers with coverage that day), enabling
-- the same day-over-day comparison as pit_quant_signal above.
create table if not exists pit_analyst_rating (
  id bigint generated always as identity primary key,
  ticker text not null,
  as_of_date date not null,
  consensus text not null,
  analyst_count integer,
  buy_pct real,
  target_mean real,
  target_high real,
  target_low real,
  current_price real,
  captured_at_utc timestamptz not null default now(),
  source text not null default 'yfinance',
  unique (ticker, as_of_date)
);
create index if not exists pit_analyst_rating_ticker_date_idx on pit_analyst_rating(ticker, as_of_date desc);

-- Phase 1 ("Trust") two-score system: one row per (as_of_date, universe_id,
-- ticker), a rules-based composite (weighted percentile-rank of factors),
-- NOT a trained ML model -- computed purely from pit_prices/pit_fundamentals/
-- pit_universe_membership already on record for that day. Append-only like
-- every other PIT-family table -- a day's score never changes once written.
-- factor_detail carries each factor's raw value, percentile, contribution,
-- and source ('pit'|'live', see services/stock_score_capture_service.py's
-- hybrid fallback) so EXP-1/2/3's explanations never need to recompute
-- anything. short_score/long_score already ARE the universe percentile;
-- short_sector_percentile/long_sector_percentile are the separate
-- sector-scoped recomputation SCR-3 needs.
create table if not exists stock_scores (
  id bigint generated always as identity primary key,
  as_of_date date not null,
  universe_id text not null,
  ticker text not null,
  short_score real,
  short_signal text not null,
  short_confidence_score real,
  short_confidence_label text not null default 'unknown',
  long_score real,
  long_signal text not null,
  long_confidence_score real,
  long_confidence_label text not null default 'unknown',
  sector_key text not null,
  short_sector_percentile real,
  long_sector_percentile real,
  factor_detail jsonb not null,
  computed_at_utc timestamptz not null default now(),
  unique (as_of_date, universe_id, ticker)
);
create index if not exists stock_scores_ticker_date_idx on stock_scores(ticker, as_of_date desc);

-- SCR-3: universe-wide percentile ("Top 8% of S&P 500") alongside the
-- existing sector percentile, plus sector rank+count ("top 3 of 22 in
-- Semis") -- an ordinal position, not a percentage, computed by
-- services/stock_score_service.py's sector_rank. Nullable/backfilled
-- going forward only, same convention as every other additive column
-- on this table.
alter table stock_scores add column if not exists short_universe_percentile real;
alter table stock_scores add column if not exists long_universe_percentile real;
alter table stock_scores add column if not exists short_sector_rank int;
alter table stock_scores add column if not exists short_sector_count int;
alter table stock_scores add column if not exists long_sector_rank int;
alter table stock_scores add column if not exists long_sector_count int;

-- Shared, ticker-keyed (not user-scoped) LLM sentiment reading — one row
-- per ticker per day, reused across every user/portfolio holding that
-- ticker, same sharing rationale as the yfinance cache. This is a
-- "current reading" display only, not a 5d/10d forecast: see
-- docs/market-direction-sentiment-requirements.md 9a-9c, where a related
-- predictive sentiment signal failed its own backtest validation gate
-- four separate times. label/reasoning are nullable because a failed
-- LLM call or unparseable response degrades that one ticker to "unknown"
-- rather than blocking the rest of the portfolio.
create table if not exists ticker_sentiment_snapshots (
  id bigint generated always as identity primary key,
  ticker text not null,
  as_of_date date not null,
  label text,
  reasoning text,
  updated_at timestamptz not null default now(),
  unique (ticker, as_of_date)
);
create index if not exists ticker_sentiment_snapshots_ticker_date_idx on ticker_sentiment_snapshots(ticker, as_of_date desc);

-- SUM-1: real SEC 10-K/10-Q filing summaries, straight from SEC EDGAR's
-- free submissions/document APIs (services/edgar_service.py) -- not the
-- LLM-guesses-from-training-knowledge approach Agent/filingAgent.py uses.
-- Shared per ticker like ticker_sentiment_snapshots above, not user-scoped.
-- Unique on (ticker, accession_number) is the idempotency guarantee the
-- daily scheduler job relies on -- re-running it never re-summarizes or
-- duplicates a filing already stored. compared_to_accession_number is
-- null only when there's truly no prior filing of that form type in SEC's
-- own history for this ticker (e.g. a recent IPO's first 10-K).
create table if not exists filing_summaries (
  id bigint generated always as identity primary key,
  ticker text not null,
  cik text not null,
  form_type text not null,
  accession_number text not null,
  filing_date date not null,
  report_date date,
  document_url text not null,
  compared_to_accession_number text,
  summary text not null,
  method text not null,
  created_at timestamptz not null default now(),
  unique (ticker, accession_number)
);
create index if not exists filing_summaries_ticker_form_idx on filing_summaries(ticker, form_type, filing_date desc);

-- SUM-2: real earnings press-release summaries, straight from SEC EDGAR's
-- 8-K Exhibit 99.1 (services/earnings_release_service.py) -- the press
-- release only, NOT a transcript of the earnings call (EDGAR doesn't have
-- one; see that module's NO_QA_CAVEAT). Unique on (ticker, accession_
-- number) is permanent-caching idempotency -- a published release never
-- changes, so once summarized it's summarized for good.
create table if not exists earnings_release_summaries (
  id bigint generated always as identity primary key,
  ticker text not null,
  cik text not null,
  accession_number text not null,
  filing_date date not null,
  report_date date,
  document_url text not null,
  summary text not null,
  method text not null,
  created_at timestamptz not null default now(),
  unique (ticker, accession_number)
);
create index if not exists earnings_release_summaries_ticker_date_idx on earnings_release_summaries(ticker, filing_date desc);

-- Plaid brokerage integration: one row per linked institution (a "Link"
-- connection). access_token_encrypted is Fernet ciphertext (see
-- web/backend/crypto_utils.py) -- never plaintext at rest, and never
-- selected back to the frontend (routers/plaid_integration.py uses an
-- explicit column list on every read, never `select *`, on this table).
create table if not exists plaid_items (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  portfolio_id bigint not null references portfolios(id) on delete cascade,
  plaid_item_id text not null unique,
  access_token_encrypted bytea not null,
  institution_id text,
  institution_name text,
  status text not null default 'active',
  last_sync_at timestamptz,
  last_sync_error text,
  created_at timestamptz not null default now()
);
create index if not exists plaid_items_user_idx on plaid_items(user_id);
alter table plaid_items enable row level security;
drop policy if exists plaid_items_isolation on plaid_items;
create policy plaid_items_isolation on plaid_items
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

-- Append-only history of each sync attempt for a linked item -- audit
-- trail for "why did my holdings change/not change," same shape as
-- backup_runs.
create table if not exists plaid_sync_log (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  plaid_item_id bigint not null references plaid_items(id) on delete cascade,
  started_at timestamptz not null default now(),
  finished_at timestamptz,
  status text not null,
  positions_upserted integer,
  error_detail text,
  triggered_by text not null
);
create index if not exists plaid_sync_log_item_idx on plaid_sync_log(plaid_item_id, started_at desc);
alter table plaid_sync_log enable row level security;
drop policy if exists plaid_sync_log_isolation on plaid_sync_log;
create policy plaid_sync_log_isolation on plaid_sync_log
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

-- Reconciliation scope for a Plaid sync: NULL for every manual/CSV row,
-- set only for a row this item's own sync wrote. A sync's delete+reinsert
-- (web/backend/plaid_sync.py) filters on THIS column, never on the
-- text `source` column, so it can never touch a manual/CSV/other-item
-- row that happens to share the same ticker -- see aws_deploy.py's own
-- history: the pre-existing CSV-import merge path is ticker-keyed, not
-- source-scoped, which is exactly the ambiguity this column avoids for
-- an ongoing, unattended sync.
alter table portfolio_positions add column if not exists plaid_item_id bigint references plaid_items(id) on delete cascade;
alter table portfolio_strategies add column if not exists plaid_item_id bigint references plaid_items(id) on delete cascade;

-- Paper trading (Alpaca): one paper-trading link per user (Alpaca issues
-- one paper account per API key pair, unlike plaid_items' one-row-per-
-- institution shape). api_key_id is not secret (Alpaca shows it in
-- plaintext in its own dashboard) so it's stored in the clear for
-- display; only the secret key is Fernet-encrypted via the same
-- web/backend/crypto_utils.py helper used for Plaid access tokens.
create table if not exists alpaca_paper_accounts (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  portfolio_id bigint not null references portfolios(id) on delete cascade,
  api_key_id text not null,
  api_secret_key_encrypted bytea not null,
  alpaca_account_id text,
  account_number text,
  status text not null default 'active',
  last_sync_at timestamptz,
  last_sync_error text,
  disclosure_accepted_at timestamptz,
  created_at timestamptz not null default now()
);
create unique index if not exists alpaca_paper_accounts_user_idx on alpaca_paper_accounts(user_id);
alter table alpaca_paper_accounts enable row level security;
drop policy if exists alpaca_paper_accounts_isolation on alpaca_paper_accounts;
create policy alpaca_paper_accounts_isolation on alpaca_paper_accounts
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

-- STB-4: one row per strategy backtest run. Counts how many rule variants a user has
-- tried, so the overfitting warning can say so. Insert-only for the app role.
create table if not exists strategy_backtest_runs (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  definition_hash text not null,
  created_at timestamptz not null default now()
);
create index if not exists strategy_backtest_runs_user_idx on strategy_backtest_runs(user_id, created_at);
alter table strategy_backtest_runs enable row level security;
drop policy if exists strategy_backtest_runs_isolation on strategy_backtest_runs;
create policy strategy_backtest_runs_isolation on strategy_backtest_runs
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

-- One row per order ticket, inserted at SUBMITTING status BEFORE the
-- Alpaca call so an ambiguous network failure can be resolved by
-- re-querying Alpaca for client_order_id rather than blind-retried.
-- Alpaca dedupes on client_order_id natively -- no separate app-level
-- idempotency table needed.
create table if not exists paper_orders (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  alpaca_paper_account_id bigint not null references alpaca_paper_accounts(id) on delete cascade,
  client_order_id uuid not null unique,
  alpaca_order_id text,
  ticker text not null,
  side text not null,
  order_type text not null,
  time_in_force text not null,
  qty real,
  limit_price real,
  status text not null default 'DRAFT',
  filled_qty real not null default 0,
  filled_avg_price real,
  reject_reason text,
  submitted_at timestamptz,
  last_polled_at timestamptz,
  created_at timestamptz not null default now()
);
create index if not exists paper_orders_user_idx on paper_orders(user_id, created_at desc);
create index if not exists paper_orders_open_idx on paper_orders(status) where status in ('OPEN','SUBMITTING','PARTIALLY_FILLED');
alter table paper_orders enable row level security;
drop policy if exists paper_orders_isolation on paper_orders;
create policy paper_orders_isolation on paper_orders
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

-- Append-only audit trail: every ticket snapshot, check result, submit
-- attempt, broker response and user action. No app role is ever granted
-- update/delete on this table -- insert-only by grant, not just convention.
create table if not exists paper_order_audit_log (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  paper_order_id bigint references paper_orders(id) on delete cascade,
  event_type text not null,
  detail jsonb not null,
  created_at timestamptz not null default now()
);
create index if not exists paper_order_audit_log_order_idx on paper_order_audit_log(paper_order_id, created_at);
create index if not exists paper_order_audit_log_user_idx on paper_order_audit_log(user_id, created_at desc);
alter table paper_order_audit_log enable row level security;
drop policy if exists paper_order_audit_log_isolation on paper_order_audit_log;
create policy paper_order_audit_log_isolation on paper_order_audit_log
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

-- Reuses portfolio_positions AND portfolio_strategies -- mirrors
-- plaid_item_id's pattern on BOTH tables, since the Holdings/Strategies
-- UI reads from portfolio_strategies, not just portfolio_positions.
alter table portfolio_positions add column if not exists alpaca_paper_account_id bigint references alpaca_paper_accounts(id) on delete cascade;
alter table portfolio_strategies add column if not exists alpaca_paper_account_id bigint references alpaca_paper_accounts(id) on delete cascade;

-- PPR-2: monthly paper-trading challenges with a friend leaderboard.
-- No RLS on challenges/challenge_members -- these are only ever touched
-- via service_conn() from web/backend/routers/challenges.py, which does
-- its own explicit membership/ownership checks (same "shared data, no
-- policy" shape as ticker_sentiment_snapshots/earnings_release_summaries
-- above). app_user gets no grant on either table at all.
create table if not exists challenges (
  id bigint generated always as identity primary key,
  name text not null,
  created_by uuid not null references users(id) on delete cascade,
  join_code text not null unique,
  start_date date not null,
  end_date date not null,
  created_at timestamptz not null default now(),
  check (end_date > start_date)
);
-- Ranking method the creator picks (see services/challenge_service.py SCORING_METHODS).
alter table challenges add column if not exists scoring text not null default 'return';
alter table challenges add column if not exists include_quant_model boolean not null default false;
alter table challenges drop constraint if exists challenges_scoring_check;
alter table challenges add constraint challenges_scoring_check check (scoring in ('return','sharpe','sortino','calmar','excess_spy'));

create table if not exists challenge_members (
  challenge_id bigint not null references challenges(id) on delete cascade,
  user_id uuid not null references users(id) on delete cascade,
  joined_at timestamptz not null default now(),
  primary key (challenge_id, user_id)
);
create index if not exists challenge_members_user_idx on challenge_members(user_id);

-- Daily equity snapshot per linked paper-trading account -- the return
-- series a challenge leaderboard needs (a single live balance isn't
-- enough to show risk, only a point-in-time number). Keeps standard
-- self-scoped RLS like any other private financial-data table; the
-- leaderboard endpoint reads other members' rows via service_conn()
-- (bypasses RLS) but only ever returns computed percentages derived
-- from this table, never these raw equity values, to other members.
create table if not exists paper_account_equity_snapshots (
  id bigint generated always as identity primary key,
  alpaca_paper_account_id bigint not null references alpaca_paper_accounts(id) on delete cascade,
  user_id uuid not null references users(id) on delete cascade,
  as_of_date date not null,
  equity real not null,
  created_at timestamptz not null default now(),
  unique (alpaca_paper_account_id, as_of_date)
);
create index if not exists paper_account_equity_snapshots_user_idx on paper_account_equity_snapshots(user_id, as_of_date);
alter table paper_account_equity_snapshots enable row level security;
drop policy if exists paper_account_equity_snapshots_isolation on paper_account_equity_snapshots;
create policy paper_account_equity_snapshots_isolation on paper_account_equity_snapshots
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

-- Defaults true: launched with every existing account opted in so the
-- "connect with the community" directory isn't empty on day one. New
-- signups get an explicit, visible checkbox (defaulting checked, same
-- public-by-default posture) instead of silently inheriting this
-- column default -- see auth.py's /signup, which passes the checkbox
-- value explicitly rather than relying on this default applying.
alter table users add column if not exists discoverable_for_challenges boolean not null default true;

-- A request, not an auto-join: invited_user_id must explicitly accept
-- before challenge_members gets a row, same consent principle as the
-- join-code flow (nobody is added to a group without their own action).
-- No RLS, same "only ever touched via service_conn() with its own
-- explicit checks" shape as challenges/challenge_members above.
create table if not exists challenge_invites (
  id bigint generated always as identity primary key,
  challenge_id bigint not null references challenges(id) on delete cascade,
  invited_user_id uuid not null references users(id) on delete cascade,
  invited_by uuid not null references users(id) on delete cascade,
  status text not null default 'pending' check (status in ('pending', 'accepted', 'declined')),
  created_at timestamptz not null default now(),
  responded_at timestamptz,
  unique (challenge_id, invited_user_id)
);
create index if not exists challenge_invites_invited_user_idx on challenge_invites(invited_user_id, status);

-- Trading agent (Phase 4). Per-user gate and breaker state: mutable, so
-- app_service gets update. Agent journal tables below are insert-only by
-- grant (no update/delete for any app role), same convention as
-- paper_order_audit_log: status changes are new event rows, never edits.
create table if not exists agent_user_settings (
  user_id uuid primary key references users(id) on delete cascade,
  enabled boolean not null default false,
  mode text not null default 'plan' check (mode in ('plan', 'paper', 'live')),
  compliance_reference text,
  enabled_by uuid references users(id),
  enabled_at timestamptz,
  peak_equity real,
  breaker_latched boolean not null default false,
  breaker_latched_at timestamptz,
  kill_engaged boolean not null default false,
  updated_at timestamptz not null default now()
);

create table if not exists agent_runs (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  mode text not null check (mode in ('plan', 'paper', 'live')),
  status text not null check (status in ('started', 'skipped')),
  regime text,
  exposure_cap_pct real,
  equity real,
  last_equity real,
  risk_state text,
  config_version text not null,
  reason text,
  created_at timestamptz not null default now()
);
create index if not exists agent_runs_user_idx on agent_runs(user_id, created_at desc);
alter table agent_runs enable row level security;
drop policy if exists agent_runs_isolation on agent_runs;
create policy agent_runs_isolation on agent_runs
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

create table if not exists agent_order_events (
  id bigint generated always as identity primary key,
  run_id bigint not null references agent_runs(id),
  user_id uuid not null references users(id) on delete cascade,
  event_type text not null,
  ticker text,
  side text,
  qty real,
  est_price real,
  est_value real,
  trigger text,
  reason text not null,
  alpaca_order_id text,
  client_order_id text,
  detail jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);
create index if not exists agent_order_events_run_idx on agent_order_events(run_id, created_at);
alter table agent_order_events enable row level security;
drop policy if exists agent_order_events_isolation on agent_order_events;
create policy agent_order_events_isolation on agent_order_events
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

create table if not exists challenge_rank_history (
  challenge_id bigint not null references challenges(id) on delete cascade,
  user_id uuid not null references users(id) on delete cascade,
  as_of_date date not null,
  rank integer not null,
  score real,
  primary key (challenge_id, user_id, as_of_date)
);

-- One row per (challenge, member, kind, day): the notification job claims it
-- before sending, so a rerun the same day cannot send twice.
create table if not exists challenge_notification_log (
  id bigint generated always as identity primary key,
  challenge_id bigint not null references challenges(id) on delete cascade,
  user_id uuid not null references users(id) on delete cascade,
  kind text not null,
  as_of_date date not null,
  created_at timestamptz not null default now(),
  unique (challenge_id, user_id, kind, as_of_date)
);

do $$
begin
  if not exists (select from pg_roles where rolname = 'app_user') then
    create role app_user login password '{app_user_pw}';
  else
    alter role app_user password '{app_user_pw}';
  end if;
  if not exists (select from pg_roles where rolname = 'app_service') then
    create role app_service login password '{app_service_pw}' bypassrls;
  else
    alter role app_service password '{app_service_pw}';
  end if;
end
$$;

grant connect on database stanalysisengine to app_user, app_service;
grant usage on schema public to app_user, app_service;
grant select, insert, update, delete on users, trades, portfolio_positions, portfolio_strategies, saved_predictions, watchlist_alerts, strategy_plans, portfolios, saved_narratives, saved_baseline_snapshots, saved_screens, saved_portfolio_goals, portfolio_insights_snapshots, saved_monthly_plans, saved_stress_scenarios to app_user;
grant select, update on portfolio_drop_alerts to app_user;
grant select, update on basket_rebalance_alerts to app_user;
grant usage, select on all sequences in schema public to app_user;
grant select, insert on request_log to app_service;
grant select, insert, update, delete on users to app_service;
grant select, insert, update, delete on password_reset_tokens to app_service;
-- Read-only, cross-user: the portfolio drop-alert scan needs to see every
-- user's holdings, not just one RLS-scoped user's own (service_conn
-- bypasses RLS but still needs an explicit grant per table). Also used by
-- the admin Users page to show each user's portfolio_count/position_count.
grant select on portfolio_positions to app_service;
-- Widens app_service beyond the read-only grant above: the Plaid sync
-- job (web/backend/plaid_sync.py) reconciles a linked item's holdings
-- via service_conn, following the exact precedent scan_portfolios_for_drops
-- already set for portfolio_drop_alerts -- RLS is bypassed, so every
-- write in that job MUST scope by user_id AND portfolio_id AND
-- plaid_item_id together, never any one alone.
grant insert, update, delete on portfolio_positions to app_service;
grant select, insert, update, delete on portfolio_strategies to app_service;
-- sync_item_holdings (plaid_sync.py) calls the same
-- _invalidate_insights_snapshot() portfolio.py's own save path uses, so
-- it needs the same access there that app_user already has. select is
-- required too, not just delete -- Postgres requires SELECT on any
-- column referenced in a DELETE's WHERE clause, not only DELETE on the
-- table itself.
grant select, delete on portfolio_insights_snapshots to app_service;
grant select, insert, update, delete on plaid_items to app_user;
grant select, update on plaid_items to app_service;
grant select on plaid_sync_log to app_user;
grant select, insert on plaid_sync_log to app_service;
grant select, insert, update, delete on alpaca_paper_accounts to app_user;
grant select, insert on strategy_backtest_runs to app_user;
grant select, update on alpaca_paper_accounts to app_service;
grant select, insert on paper_orders to app_user;
grant select, update on paper_orders to app_service;
grant select, insert on paper_order_audit_log to app_user;
grant select, insert on paper_order_audit_log to app_service;
-- update/delete: the admin per-user portfolio panel deactivates/reactivates
-- (update) or permanently removes (delete) a specific portfolio, cross-user
-- like the rest of admin_users.py.
grant select, update, delete on portfolios to app_service;
grant select, update on saved_predictions to app_service;
grant select, update on watchlist_alerts to app_service;
grant select, insert, update on app_settings to app_service;
grant select on published_signals to app_user;
grant select, insert on published_signals to app_service;
grant select on signal_outcomes to app_user;
grant select, insert on signal_outcomes to app_service;
grant select on backtest_runs to app_user;
grant select, insert on backtest_runs to app_service;
grant select on backup_runs to app_user;
grant select, insert, update on backup_runs to app_service;
grant select on pit_prices to app_user;
grant select, insert on pit_prices to app_service;
grant select on pit_universe_membership to app_user;
grant select, insert on pit_universe_membership to app_service;
grant select on pit_fundamentals to app_user;
grant select, insert on pit_fundamentals to app_service;
grant select on pit_quant_signal to app_user;
grant select, insert on pit_quant_signal to app_service;
grant select on quant_signal_outcomes to app_user;
grant select, insert on quant_signal_outcomes to app_service;
grant select on pit_analyst_rating to app_user;
grant select, insert on pit_analyst_rating to app_service;
grant select on stock_scores to app_user;
grant select, insert on stock_scores to app_service;
grant select on ticker_sentiment_snapshots to app_user;
grant select, insert on ticker_sentiment_snapshots to app_service;
grant select on filing_summaries to app_user;
grant select, insert on filing_summaries to app_service;
grant select on earnings_release_summaries to app_user;
grant select, insert on earnings_release_summaries to app_service;
grant select, insert, update on agent_user_settings to app_service;
grant select, insert on agent_runs to app_service;
grant select on agent_runs to app_user;
grant select, insert on agent_order_events to app_service;
grant select on agent_order_events to app_user;
grant select, insert on challenge_rank_history to app_service;
grant select, insert on challenge_notification_log to app_service;
grant select, insert on challenges to app_service;
grant select, insert, delete on challenge_members to app_service;
grant select, insert on paper_account_equity_snapshots to app_service;
grant select on paper_account_equity_snapshots to app_user;
grant select, insert, update on challenge_invites to app_service;
-- update needed: scan_portfolios_for_drops refreshes an already-alerted
-- row in place (see web/backend/portfolio_alerts.py) rather than only
-- ever inserting new ones.
grant select, insert, update on portfolio_drop_alerts to app_service;
-- Same refresh-in-place rationale as portfolio_drop_alerts above (see
-- services/basket_rebalance_service.py's scan_baskets_for_rebalance).
grant select, insert, update on basket_rebalance_alerts to app_service;
-- SCN-3: scan_saved_screens_for_membership_changes (services/
-- saved_screen_alert_service.py) reads every user's saved screens
-- cross-user via service_conn, same read-only-cross-user need as
-- portfolio_positions above.
grant select on saved_screens to app_service;
-- Same refresh-in-place rationale as portfolio_drop_alerts/
-- basket_rebalance_alerts above -- a same-day re-run updates that day's
-- row rather than only ever inserting.
grant select, insert, update on saved_screen_alerts to app_service;

-- Horizon 1 (docs/signal-licensing-whitelabel-requirements.md.pdf, RS-*):
-- built and migrated so the code is ready, but gated off by
-- horizon1_subscriptions_enabled (see app_settings.py) until the real
-- business/legal gate (Gate 0->1) is actually met.
create table if not exists subscriptions (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  tier text not null default 'free' check (tier in ('free','paid')),
  status text not null default 'active' check (status in ('active','canceled','past_due','incomplete')),
  stripe_customer_id text,
  stripe_subscription_id text unique,
  current_period_end timestamptz,
  created_at timestamptz not null default now(),
  canceled_at timestamptz
);
create index if not exists subscriptions_user_idx on subscriptions(user_id);
alter table subscriptions enable row level security;
drop policy if exists subscriptions_isolation on subscriptions;
create policy subscriptions_isolation on subscriptions
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

-- RS-6 audit log + RS-5 demand instrumentation share one append-only
-- table: both need (actor, event, resource, timestamp), and splitting
-- them would just duplicate the same insert-only plumbing
-- published_signals already establishes as this app's pattern for an
-- immutable record. No RLS: written only via service_conn from backend
-- code, read only by admin.
create table if not exists subscriber_events (
  id bigint generated always as identity primary key,
  actor_user_id uuid references users(id) on delete set null,
  event_type text not null,
  resource text,
  metadata jsonb,
  created_at timestamptz not null default now()
);
create index if not exists subscriber_events_type_idx on subscriber_events(event_type, created_at);
create index if not exists subscriber_events_actor_idx on subscriber_events(actor_user_id, created_at);

create table if not exists demand_enquiries (
  id bigint generated always as identity primary key,
  user_id uuid references users(id) on delete set null,
  enquiry_type text not null check (enquiry_type in ('licensing','api','institutional','other')),
  message text,
  contact_email text not null,
  created_at timestamptz not null default now()
);

grant select, insert, update on subscriptions to app_user;
grant select, insert, update on subscriptions to app_service;
grant select, insert on subscriber_events to app_service;
grant select, insert on demand_enquiries to app_service;

insert into app_settings (key, value) values ('horizon1_subscriptions_enabled', 'false') on conflict (key) do nothing;
insert into app_settings (key, value) values ('free_tier_lag_days', '7') on conflict (key) do nothing;

-- REG-1/2/3: one row per trading day from the Market Direction Phase 1
-- Internals engine (services/market_internals_service.py) -- see that
-- module's docstring for the failed-release-gate history this ships
-- under explicit, informed override. regime_raw is the unsmoothed daily
-- label; regime_confirmed is regime_raw after SR-5 hysteresis (a change
-- only sticks after 2 consecutive sessions), which can retroactively
-- reconsider a recent day as more days land -- unlike the append-only
-- PIT-family tables, this one is refreshed in place, hence the update
-- grant below. breadth_50dma/vix/vix3m/xly_xlp/hyg_ief/rsp_spy are the
-- raw same-day inputs, stored for the banner's "trend/volatility/
-- breadth/rates" display (services/market_internals_service.py::
-- compute_internals_components) without needing to refetch history for
-- a historical date.
create table if not exists market_regime_daily (
  id bigint generated always as identity primary key,
  as_of_date date not null unique,
  internals_score real,
  mds real,
  regime_raw text,
  regime_confirmed text,
  data_completeness real not null,
  conflict_flag boolean not null default false,
  breadth_50dma real,
  vix real,
  vix3m real,
  xly_xlp real,
  hyg_ief real,
  rsp_spy real,
  computed_at_utc timestamptz not null default now()
);
create index if not exists market_regime_daily_date_idx on market_regime_daily(as_of_date desc);

grant select on market_regime_daily to app_user;
grant select, insert, update on market_regime_daily to app_service;

-- Defaults OFF, doubly deliberate here vs. every other *_ENABLED_KEY:
-- beyond the usual "deploying code must not itself start a live job"
-- rationale, this one wires up a scoring engine that failed its own
-- release-gate backtest three times (see market_internals_service.py) --
-- an admin must opt in with that history in view via /admin/settings,
-- not have it start scoring/banner-ing the moment this deploys.
insert into app_settings (key, value) values ('market_regime_enabled', 'false') on conflict (key) do nothing;

-- ALR-1: day-over-day short_signal/long_signal change, per owned or
-- watchlisted ticker. Mirrors watchlist_alerts' user isolation; a row
-- per (user, ticker, day, horizon) since short and long can each change
-- independently on the same day.
create table if not exists signal_change_alerts (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  ticker text not null,
  alert_date date not null,
  horizon text not null check (horizon in ('short', 'long')),
  old_signal text,
  new_signal text,
  created_at timestamptz not null default now(),
  seen_at timestamptz,
  unique (user_id, ticker, alert_date, horizon)
);
create index if not exists signal_change_alerts_user_idx on signal_change_alerts(user_id, created_at desc);
alter table signal_change_alerts enable row level security;
drop policy if exists signal_change_alerts_isolation on signal_change_alerts;
create policy signal_change_alerts_isolation on signal_change_alerts for all
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

-- ALR-1: earnings-in-2-days, one row per (user, ticker, day) the window
-- was first entered -- reuses stock_detail_service.upcoming_earnings_in_window
-- (called with window_days=2) for the actual date math, no new logic there.
create table if not exists earnings_alert_log (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  ticker text not null,
  alert_date date not null,
  earnings_date date not null,
  created_at timestamptz not null default now(),
  seen_at timestamptz,
  unique (user_id, ticker, alert_date)
);
create index if not exists earnings_alert_log_user_idx on earnings_alert_log(user_id, created_at desc);
alter table earnings_alert_log enable row level security;
drop policy if exists earnings_alert_log_isolation on earnings_alert_log;
create policy earnings_alert_log_isolation on earnings_alert_log for all
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

-- ALR-1: "a holding falling a set % from cost" -- distinct from the
-- existing portfolio_drop_alerts above, which compares to yesterday's
-- close, not cost basis. Same per-day-cap idiom, deliberately simpler
-- (no LLM sentiment synthesis, matching saved_screen_alerts' lighter
-- shape instead of portfolio_drop_alerts' richer one).
create table if not exists cost_drop_alerts (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  ticker text not null,
  alert_date date not null,
  avg_cost real not null,
  current_price real not null,
  pct_change real not null,
  created_at timestamptz not null default now(),
  seen_at timestamptz,
  unique (user_id, ticker, alert_date)
);
create index if not exists cost_drop_alerts_user_idx on cost_drop_alerts(user_id, created_at desc);
alter table cost_drop_alerts enable row level security;
drop policy if exists cost_drop_alerts_isolation on cost_drop_alerts;
create policy cost_drop_alerts_isolation on cost_drop_alerts for all
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

grant select, insert on signal_change_alerts to app_service;
grant select, insert on earnings_alert_log to app_service;
grant select, insert on cost_drop_alerts to app_service;
grant select, update, delete on signal_change_alerts, earnings_alert_log, cost_drop_alerts to app_user;

-- Each gets its own deliberate-opt-in flag, same rationale as
-- portfolio_drop_alerts_enabled -- these write user-visible content and
-- send email, so a deploy must not itself start alerting anyone.
insert into app_settings (key, value) values ('signal_change_alerts_enabled', 'false') on conflict (key) do nothing;
insert into app_settings (key, value) values ('earnings_alerts_enabled', 'false') on conflict (key) do nothing;
insert into app_settings (key, value) values ('cost_drop_alerts_enabled', 'false') on conflict (key) do nothing;
-- Separate from portfolio_drop_threshold_pct -- that one means "vs.
-- yesterday's close", this means "vs. cost basis", different concepts.
insert into app_settings (key, value) values ('cost_drop_threshold_pct', '10.0') on conflict (key) do nothing;

-- ALR-1/2: per-alert-type channel preference, global (ticker is null) or
-- per-ticker (overrides the global row for that alert_type). A plain
-- unique(user_id, ticker, alert_type) constraint would NOT work here --
-- Postgres treats every NULL as distinct, so it would silently allow
-- duplicate "global" rows -- hence two partial unique indexes instead.
-- No row for a given (user, alert_type) means the default in
-- services/notification_dispatcher.py applies (enabled, email+in-app on).
create table if not exists user_alert_preferences (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  ticker text,
  alert_type text not null,
  enabled boolean not null default true,
  channel_email boolean not null default true,
  channel_inapp boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create unique index if not exists user_alert_preferences_global_idx
  on user_alert_preferences(user_id, alert_type) where ticker is null;
create unique index if not exists user_alert_preferences_ticker_idx
  on user_alert_preferences(user_id, ticker, alert_type) where ticker is not null;
alter table user_alert_preferences enable row level security;
drop policy if exists user_alert_preferences_isolation on user_alert_preferences;
create policy user_alert_preferences_isolation on user_alert_preferences for all
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

-- ALR-2: quiet hours + daily digest + (ALR-3, columns added now to avoid
-- a second migration on this table later) webhook delivery settings.
-- One row per user, created on first write (no default row seeded).
-- push_enabled is deliberately NOT a column here yet -- no push
-- infrastructure exists (see the Smart Alerts plan's Stage 10), and this
-- table shouldn't carry a column for a channel that doesn't exist.
create table if not exists user_notification_settings (
  user_id uuid primary key references users(id) on delete cascade,
  quiet_hours_start time,
  quiet_hours_end time,
  digest_enabled boolean not null default false,
  digest_time time not null default '08:00',
  webhook_enabled boolean not null default false,
  webhook_url text,
  webhook_secret text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
alter table user_notification_settings enable row level security;
drop policy if exists user_notification_settings_isolation on user_notification_settings;
create policy user_notification_settings_isolation on user_notification_settings for all
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

-- ALR-2: the "quiet hours" / "digest mode" delivery queue --
-- notification_dispatcher.py writes here instead of sending immediately
-- when either applies; the hourly flush job (scheduler.py) sends one
-- consolidated email per user and marks flushed_at. Service-only table
-- (written and read exclusively via service_conn), no RLS needed.
create table if not exists pending_digest_items (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  ticker text,
  alert_type text not null,
  subject text not null,
  text_body text not null,
  created_at timestamptz not null default now(),
  flushed_at timestamptz
);
create index if not exists pending_digest_items_unflushed_idx
  on pending_digest_items(user_id) where flushed_at is null;

grant select, insert, update, delete on user_alert_preferences to app_user;
grant select, insert, update on user_alert_preferences to app_service;
grant select, insert, update on user_notification_settings to app_user;
grant select, insert, update on user_notification_settings to app_service;
grant select, insert, update on pending_digest_items to app_service;

grant usage, select on all sequences in schema public to app_service;
"""

_BACKEND_SERVICE = """[Unit]
Description=StAnalysisEngine API
After=network.target

[Service]
WorkingDirectory=/opt/stanalysisengine
ExecStart=/opt/stanalysisengine/venv/bin/uvicorn web.backend.main:app --host 127.0.0.1 --port 8000
Restart=always
User=ubuntu

[Install]
WantedBy=multi-user.target
"""

_FRONTEND_SERVICE = """[Unit]
Description=StAnalysisEngine Web
After=network.target

[Service]
WorkingDirectory=/opt/stanalysisengine/web/frontend
ExecStart=/usr/bin/npm run start
Restart=always
User=ubuntu
Environment=PORT=3000

[Install]
WantedBy=multi-user.target
"""

_NGINX_CONF = """server {
    listen 80;
    server_name _;

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
"""

_LLM_KEY_ENV_NAMES = ["OPENAI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY"]


class DeployRequest(BaseModel):
    public_ip: str
    username: str = "ubuntu"


def _worker_deploy(job_id: str, req: DeployRequest) -> None:
    job = get_job(job_id)
    try:
        client = _connect_ssh(req.public_ip, req.username, job)
        sftp = client.open_sftp()

        _, _, rc = _ssh_exec(client, f"test -d {REMOTE_DIR}/.git")
        repo_exists = rc == 0

        # Setup completion is tracked independently of the clone: a partial
        # first attempt (e.g. clone succeeds, schema/env/systemd step fails)
        # must not permanently strand the box in the "existing install, just
        # git pull" path on every later retry — check the last artifact this
        # block writes, not just whether the repo directory exists.
        _, _, rc = _ssh_exec(client, "test -f /etc/systemd/system/stanalysisengine-api.service")
        setup_done = rc == 0

        if not repo_exists:
            log(job, "Waiting for server setup (nginx/postgres/node) to finish...")
            setup_ready = False
            for _ in range(36):  # up to ~3 min at 5s intervals
                _, _, rc = _ssh_exec(client, "test -f /tmp/stanalysisengine_setup_done")
                if rc == 0:
                    setup_ready = True
                    break
                time.sleep(5)
            if not setup_ready:
                raise RuntimeError(
                    "Server setup (user-data) did not finish within 3 minutes — "
                    "check /var/log/cloud-init-output.log on the instance, then retry deploy."
                )
            log(job, "✓ Server setup complete")

            log(job, "First-time setup — cloning repo")
            _ssh_exec(client, f"sudo mkdir -p {REMOTE_DIR} && sudo chown ubuntu:ubuntu {REMOTE_DIR}")
            out, err, rc = _ssh_exec(client, f"git clone {REPO_URL} {REMOTE_DIR}", timeout=180)
            if rc != 0:
                raise RuntimeError(f"git clone failed: {err[-400:]}")
            log(job, "✓ Repo cloned")
        else:
            log(job, "Repo already present — pulling latest")
            out, err, rc = _ssh_exec(client, f"cd {REMOTE_DIR} && git pull", timeout=120)
            if rc != 0:
                raise RuntimeError(f"git pull failed: {err[-400:]}")
            log(job, f"✓ {out.strip().splitlines()[-1] if out.strip() else 'up to date'}")

        if setup_done:
            log(job, "Server already configured (schema/env/services/nginx) — skipping")
        else:
            log(job, "Completing first-time setup: Postgres schema + roles")
            app_user_pw = secrets.token_hex(16)
            app_service_pw = secrets.token_hex(16)
            session_secret = secrets.token_hex(32)

            # Plain substitution, not .format() -- _SCHEMA_SQL contains
            # literal unescaped braces (e.g. default '{}'::jsonb) that
            # .format() misreads as positional placeholders and raises
            # IndexError on, confirmed live against a local sync of this
            # exact script.
            schema_sql = _SCHEMA_SQL.replace("{app_user_pw}", app_user_pw).replace("{app_service_pw}", app_service_pw)
            sftp.putfo(io.BytesIO(schema_sql.encode()), "/tmp/schema.sql")
            _ssh_exec(client, "sudo -u postgres createdb stanalysisengine 2>/dev/null; true")
            out, err, rc = _ssh_exec(client, "sudo -u postgres psql -d stanalysisengine -f /tmp/schema.sql", timeout=60)
            if rc != 0:
                raise RuntimeError(f"schema setup failed: {err[-400:]}")
            log(job, "✓ Schema applied")

            llm_lines = "\n".join(
                f"{name}={os.environ[name]}" for name in _LLM_KEY_ENV_NAMES if os.environ.get(name)
            )
            backend_env = (
                f"DATABASE_URL=postgresql://app_user:{app_user_pw}@127.0.0.1:5432/stanalysisengine\n"
                f"DATABASE_URL_SERVICE=postgresql://app_service:{app_service_pw}@127.0.0.1:5432/stanalysisengine\n"
                f"SESSION_SECRET={session_secret}\n"
                f"COOKIE_SECURE=false\n"
                f"CORS_ALLOWED_ORIGINS=http://{req.public_ip}\n"
                f"{llm_lines}\n"
            )
            sftp.putfo(io.BytesIO(backend_env.encode()), f"{REMOTE_DIR}/web/backend/.env")

            # lib/api.ts's call sites already pass full "/api/v1/..." paths,
            # so NEXT_PUBLIC_API_BASE_URL must be empty here (same-origin —
            # nginx's /api/ location proxies that literal path straight to
            # the backend). It is NOT "/api": that would double the prefix
            # to "/api/api/v1/...", a 404. Server Actions run in Node with no
            # page origin at all, so they need a real absolute URL instead —
            # hit the backend directly via BACKEND_INTERNAL_URL, bypassing
            # nginx entirely for those.
            frontend_env = (
                f"NEXT_PUBLIC_API_BASE_URL=\n"
                f"BACKEND_INTERNAL_URL=http://127.0.0.1:8000\n"
                f"SESSION_SECRET={session_secret}\n"
                f"COOKIE_SECURE=false\n"
            )
            sftp.putfo(io.BytesIO(frontend_env.encode()), f"{REMOTE_DIR}/web/frontend/.env.local")
            log(job, "✓ .env files written (fresh secrets generated, never logged)")

            sftp.putfo(io.BytesIO(_BACKEND_SERVICE.encode()), "/tmp/stanalysisengine-api.service")
            _ssh_exec(client, "sudo mv /tmp/stanalysisengine-api.service /etc/systemd/system/")
            sftp.putfo(io.BytesIO(_FRONTEND_SERVICE.encode()), "/tmp/stanalysisengine-web.service")
            _ssh_exec(client, "sudo mv /tmp/stanalysisengine-web.service /etc/systemd/system/")

            sftp.putfo(io.BytesIO(_NGINX_CONF.encode()), "/tmp/stanalysisengine_nginx.conf")
            _ssh_exec(client, "sudo mv /tmp/stanalysisengine_nginx.conf /etc/nginx/sites-available/stanalysisengine")
            _ssh_exec(client, "sudo ln -sf /etc/nginx/sites-available/stanalysisengine /etc/nginx/sites-enabled/stanalysisengine")
            _ssh_exec(client, "sudo rm -f /etc/nginx/sites-enabled/default")
            out, err, rc = _ssh_exec(client, "sudo nginx -t && sudo systemctl reload nginx")
            if rc != 0:
                log(job, f"⚠ nginx config test warning: {err[-300:]}")
            else:
                log(job, "✓ nginx configured")
            _ssh_exec(client, "sudo systemctl daemon-reload")

        _, _, rc = _ssh_exec(client, "swapon --show=NAME --noheadings | grep -q /swapfile")
        if rc != 0:
            log(job, "No swap found — small instances OOM-kill during npm build without it. Adding 2GB swap...")
            out, err, rc = _ssh_exec(
                client,
                "sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && "
                "sudo swapon /swapfile && "
                "grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab",
                timeout=60,
            )
            if rc != 0:
                log(job, f"⚠ swap setup failed (continuing anyway): {err[-300:]}")
            else:
                log(job, "✓ 2GB swap enabled")

        log(job, "Installing backend deps (this can take a few minutes)...")
        out, err, rc = _ssh_exec(
            client,
            f"cd {REMOTE_DIR} && (test -d venv || python3 -m venv venv) && "
            f"venv/bin/pip install -q -r web/backend/requirements.txt",
            timeout=400,
        )
        if rc != 0:
            raise RuntimeError(f"pip install failed: {err[-400:]}")
        log(job, "✓ Backend deps installed")

        log(job, "Installing + building frontend (this can take a few minutes)...")
        out, err, rc = _ssh_exec(
            client, f"cd {REMOTE_DIR}/web/frontend && npm ci --silent && npm run build", timeout=600
        )
        if rc != 0:
            raise RuntimeError(f"frontend build failed: {err[-600:]}")
        log(job, "✓ Frontend built")

        _ssh_exec(client, "sudo systemctl enable stanalysisengine-api stanalysisengine-web")
        _ssh_exec(client, "sudo systemctl restart stanalysisengine-api stanalysisengine-web")
        log(job, "✓ Services restarted")

        # The app imports pandas/numpy/scikit-learn/streamlit/langchain at
        # startup, which routinely takes ~10s — a fixed short sleep here
        # produces false-negative "HTTP 000" warnings on a service that is
        # actually fine a few seconds later, so poll instead of a single shot.
        healthy = False
        for _ in range(10):  # up to ~20s
            time.sleep(2)
            out, _, _ = _ssh_exec(client, "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health")
            if out.strip() == "200":
                healthy = True
                break
        if healthy:
            log(job, "✓ Backend health check passed")
        else:
            log(job, f"⚠ Backend health check returned HTTP {out.strip() or '(no response)'} after 20s")

        client.close()
        finish(job, True)
    except Exception as e:
        log(job, f"✗ {e}")
        finish(job, False)


@router.post("/deploy")
def deploy(req: DeployRequest):
    if not os.path.isfile(_pem_path()):
        raise HTTPException(400, "No PEM on this machine — create the key pair first")
    job_id, _ = new_job(f"Deploy → {req.public_ip}")
    threading.Thread(target=_worker_deploy, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id}


# ── Job polling ──────────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}")
def poll_job(job_id: str, cursor: int = 0):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    lines = job["logs"][cursor:]
    return {
        "status": job["status"],
        "lines": lines,
        "cursor": len(job["logs"]),
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
    }


@router.post("/jobs/{job_id}/cancel")
def cancel(job_id: str):
    ok = cancel_job(job_id)
    if not ok:
        raise HTTPException(404, "Job not found or already finished")
    return {"ok": True}
