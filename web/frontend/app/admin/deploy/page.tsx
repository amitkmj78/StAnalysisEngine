import { redirect } from "next/navigation";

import DeployControlPanel from "@/components/deploy/DeployControlPanel";
import { isAdmin } from "@/lib/admin";
import { getSession } from "@/lib/session";

export default async function AdminDeployPage() {
  const user = await getSession();

  // Defense in depth — proxy.ts already gates /admin/*, this is the second check.
  if (!isAdmin(user?.email)) {
    redirect("/predict");
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">AWS Deploy</h1>
      <p className="mt-1 text-sm text-slate-500">
        Live control panel above, manual runbook below. The control panel runs against your AWS account
        from this machine — it doesn&apos;t store credentials in the deployed app.
      </p>

      <DeployControlPanel />

      <h2 className="mt-2 text-lg font-semibold text-slate-900">Manual Runbook</h2>
      <p className="mt-1 text-sm text-slate-500">
        Reference for anything the automation above doesn&apos;t cover (e.g. HTTPS setup once a domain
        exists), or to do it by hand.
      </p>

      <Section n={1} title="Launch the instance">
        <p>
          AWS Console → EC2 → Launch Instance: Ubuntu 24.04 LTS, <Code inline>t3.small</Code> (2GB RAM —
          comfortable for model training + <Code inline>npm run build</Code>; <Code inline>t3.micro</Code> works
          on the free tier but may be tight during frontend builds). Create/select a key pair for SSH.
        </p>
        <p className="mt-2">
          Security group: inbound <Code inline>22</Code> (SSH, restrict to your IP), <Code inline>80</Code> (HTTP,
          0.0.0.0/0). Allocate and associate an <strong>Elastic IP</strong> so the address is stable across
          reboots.
        </p>
      </Section>

      <Section n={2} title="Server setup">
        <p>SSH in, then:</p>
        <Code>{`sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-venv python3-pip nginx git curl postgresql
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
git clone <this-repo-url> app && cd app`}</Code>
        <p className="mt-2 text-sm text-slate-600">
          Postgres runs directly on this instance (no RDS cost) — same choice made for local dev.
        </p>
      </Section>

      <Section n={3} title="Postgres schema & roles">
        <p>
          Create the database, then run the same schema used locally (<Code inline>users</Code>,{" "}
          <Code inline>trades</Code>, <Code inline>portfolio_positions</Code>,{" "}
          <Code inline>portfolio_strategies</Code>, <Code inline>request_log</Code> — RLS policies keyed on{" "}
          <Code inline>current_setting(&apos;app.user_id&apos;)</Code>, plus the <Code inline>app_user</Code> /{" "}
          <Code inline>app_service</Code> roles) as a fresh <Code inline>psql</Code> superuser session:
        </p>
        <Code>{`sudo -u postgres createdb stanalysisengine
sudo -u postgres psql -d stanalysisengine -f schema.sql   # the schema from services/ or this project's setup notes`}</Code>
        <p className="mt-2 rounded-md bg-amber-50 px-3 py-2 text-sm text-amber-800">
          Generate fresh passwords for <Code inline>app_user</Code>/<Code inline>app_service</Code> here — don&apos;t
          reuse the local dev ones.
        </p>
      </Section>

      <Section n={4} title="Backend service">
        <p>
          Create a venv, install deps, and write <Code inline>web/backend/.env</Code> with{" "}
          <Code inline>DATABASE_URL</Code> (the <Code inline>app_user</Code> connection string),{" "}
          <Code inline>DATABASE_URL_SERVICE</Code> (the <Code inline>app_service</Code> one),{" "}
          <Code inline>SESSION_SECRET</Code> (a fresh random value — <Code inline>python3 -c &quot;import secrets;
          print(secrets.token_hex(32))&quot;</Code>, not the local dev one), <Code inline>COOKIE_SECURE</Code>, and
          the LLM provider keys.
        </p>
        <Code>{`python3 -m venv venv
source venv/bin/activate
pip install -r web/backend/requirements.txt`}</Code>
        <p className="mt-2">
          systemd unit — <Code inline>/etc/systemd/system/stanalysisengine-api.service</Code>:
        </p>
        <Code>{`[Unit]
Description=StAnalysisEngine API
After=network.target

[Service]
WorkingDirectory=/home/ubuntu/app
ExecStart=/home/ubuntu/app/venv/bin/uvicorn web.backend.main:app --host 127.0.0.1 --port 8000
Restart=always
User=ubuntu

[Install]
WantedBy=multi-user.target`}</Code>
        <Code>{`sudo systemctl enable --now stanalysisengine-api
curl localhost:8000/health   # sanity check, run on the instance`}</Code>
      </Section>

      <Section n={5} title="Frontend service">
        <p>
          Same-origin behind nginx means <Code inline>NEXT_PUBLIC_API_BASE_URL</Code> becomes a{" "}
          <strong>relative path</strong> (<Code inline>/api</Code>), not an external URL.{" "}
          <Code inline>web/frontend/.env.local</Code> needs <Code inline>SESSION_SECRET</Code> set to the{" "}
          <strong>same value</strong> as the backend&apos;s — <Code inline>proxy.ts</Code> verifies the session
          cookie independently of the API, so the two must agree on the signing secret.
        </p>
        <Code>{`cd web/frontend && npm ci && npm run build`}</Code>
        <p className="mt-2">
          systemd unit — <Code inline>/etc/systemd/system/stanalysisengine-web.service</Code>, same pattern as
          the API but running <Code inline>npm run start</Code> on port 3000 with <Code inline>Restart=always</Code>.
        </p>
      </Section>

      <Section n={6} title="nginx reverse proxy">
        <p>
          <Code inline>/etc/nginx/sites-available/stanalysisengine</Code>:
        </p>
        <Code>{`server {
    listen 80;
    server_name <ELASTIC_IP>;

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
}`}</Code>
        <Code>{`sudo ln -s /etc/nginx/sites-available/stanalysisengine /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx`}</Code>
      </Section>

      <Section n={7} title="Redeploying on future pushes">
        <p>
          No auto-deploy-on-push here — a small script on the instance handles it:
        </p>
        <Code>{`#!/bin/bash
set -e
git pull
source venv/bin/activate && pip install -r web/backend/requirements.txt
sudo systemctl restart stanalysisengine-api
(cd web/frontend && npm ci && npm run build)
sudo systemctl restart stanalysisengine-web`}</Code>
      </Section>

      <Section n={8} title="Smoke test">
        <Code>{`curl http://<elastic-ip>/health
curl -X POST http://<elastic-ip>/api/v1/auth/signup -H "Content-Type: application/json" \\
  -d '{"email":"you@example.com","password":"a-real-password"}' -c cookies.txt
curl -b cookies.txt http://<elastic-ip>/api/v1/auth/me
curl -b cookies.txt "http://<elastic-ip>/api/v1/predict/summary?ticker=AAPL&period=1y"`}</Code>
        <p className="mt-2">
          Then browse to <Code inline>http://&lt;elastic-ip&gt;</Code>, sign up for real, drive a couple of
          pages, and confirm (via the network panel) requests go to same-origin <Code inline>/api/...</Code>{" "}
          with the <Code inline>session</Code> cookie attached and no CORS errors.
        </p>
      </Section>

      <div className="mt-8 rounded-lg border border-slate-200 bg-white p-5">
        <h3 className="font-semibold text-slate-900">No domain yet</h3>
        <p className="mt-2 text-sm text-slate-600">
          This serves plain HTTP on the instance&apos;s IP, and the session cookie is <Code inline>httpOnly</Code>
          {" "}+ <Code inline>SameSite=Lax</Code>. Keep <Code inline>COOKIE_SECURE=false</Code> until a domain +
          TLS cert exist — a <Code inline>Secure</Code> cookie is silently dropped by the browser over plain
          HTTP, which would make login look broken with no obvious error. Flip it to <Code inline>true</Code>{" "}
          once Let&apos;s Encrypt is set up, or sessions stop being confidential in transit.
        </p>
      </div>
    </div>
  );
}

function Section({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <div className="mt-6">
      <h2 className="text-lg font-semibold text-slate-900">
        {n}. {title}
      </h2>
      <div className="mt-2 text-sm text-slate-700">{children}</div>
    </div>
  );
}

function Code({ children, inline }: { children: string; inline?: boolean }) {
  if (inline) {
    return <code className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-xs text-slate-800">{children}</code>;
  }
  return (
    <pre className="mt-2 overflow-x-auto rounded-md bg-slate-900 p-4 text-xs leading-relaxed text-slate-100">
      <code className="font-mono">{children}</code>
    </pre>
  );
}
