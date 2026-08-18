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
-- delete: it blocks future logins (checked in /login) but keeps the
-- account and all its data/relationships intact. Existing sessions are
-- unaffected until they next log in — see verify_bearer_token, which
-- (like `approved`) never re-checks the DB per-request.
alter table users add column if not exists is_active boolean not null default true;

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

create table if not exists portfolio_positions (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  ticker text, name text, shares real, avg_cost real, current_price real,
  unrealized_pnl_pct real, source text, created_at timestamptz not null default now()
);
create index if not exists portfolio_positions_user_idx on portfolio_positions(user_id);
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

-- Saved goals from the Strategies calculator. monthly_contribution is the
-- server-computed required-monthly at save time (locked in, not
-- recomputed later) — progress tracking compounds this same fixed
-- contribution forward from created_at and compares it against the
-- user's live portfolio value, so "ahead/behind pace" means "vs. what
-- you'd have if you'd contributed this amount every month since saving."
create table if not exists strategy_plans (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  name text,
  target_amount real not null,
  years integer not null,
  starting_capital real not null,
  annual_return_pct real not null,
  monthly_contribution real not null,
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
grant select, insert, update, delete on users, trades, portfolio_positions, portfolio_strategies, saved_predictions, watchlist_alerts, strategy_plans, portfolios, saved_narratives, saved_baseline_snapshots, saved_screens to app_user;
grant select, update on portfolio_drop_alerts to app_user;
grant usage, select on all sequences in schema public to app_user;
grant select, insert on request_log to app_service;
grant select, insert, update, delete on users to app_service;
grant select, insert, update, delete on password_reset_tokens to app_service;
-- Read-only, cross-user: the portfolio drop-alert scan needs to see every
-- user's holdings, not just one RLS-scoped user's own (service_conn
-- bypasses RLS but still needs an explicit grant per table).
grant select on portfolio_positions to app_service;
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
grant select on pit_analyst_rating to app_user;
grant select, insert on pit_analyst_rating to app_service;
-- update needed: scan_portfolios_for_drops refreshes an already-alerted
-- row in place (see web/backend/portfolio_alerts.py) rather than only
-- ever inserting new ones.
grant select, insert, update on portfolio_drop_alerts to app_service;

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

            schema_sql = _SCHEMA_SQL.format(app_user_pw=app_user_pw, app_service_pw=app_service_pw)
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
