"use client";

import { ApiError } from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "";

async function req<T>(path: string, method: "GET" | "PUT" | "POST" = "GET", body?: unknown): Promise<T> {
  const res = await fetch(new URL(`${API_BASE}${path}`, window.location.origin).toString(), {
    method,
    credentials: "include",
    headers: body !== undefined ? { "Content-Type": "application/json" } : {},
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });

  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const errBody = await res.json();
      detail = errBody.detail || detail;
    } catch {
      // response wasn't JSON — keep the generic message
    }
    throw new ApiError(detail, res.status);
  }

  return res.json();
}

export interface AwsConfig {
  access_key_id: string;
  secret_access_key: string;
  region: string;
  key_name: string;
  sg_id: string;
}

export interface AwsConfigInput {
  access_key_id: string;
  secret_access_key: string;
  region: string;
  key_name: string;
}

export interface AwsVerifyResult {
  ok: boolean;
  account?: string;
  arn?: string;
  sg_id?: string;
  sg_error?: string;
  error?: string;
}

export interface Ec2Instance {
  id: string;
  type: string;
  state: string;
  public_ip: string;
  launched: string;
}

export interface AwsStatus {
  key_pair: { exists: boolean; name?: string; pem_on_server: boolean };
  instances: Ec2Instance[];
  region?: string;
  ec2_error?: string;
}

export interface JobHandle {
  job_id: string;
}

export interface JobPollResult {
  status: "running" | "success" | "error";
  lines: string[];
  cursor: number;
  started_at: string;
  finished_at: string | null;
}

export const awsDeployApi = {
  getConfig: () => req<AwsConfig>("/api/v1/aws-deploy/config"),
  saveConfig: (cfg: AwsConfigInput) => req<{ ok: boolean }>("/api/v1/aws-deploy/config", "PUT", cfg),
  verifyConfig: () => req<AwsVerifyResult>("/api/v1/aws-deploy/config/verify"),
  getStatus: () => req<AwsStatus>("/api/v1/aws-deploy/status"),
  createKeyPair: () => req<{ key_name: string; saved_to: string }>("/api/v1/aws-deploy/key-pair", "POST"),
  launchInstance: (instanceType: string, volumeSizeGb: number) =>
    req<JobHandle>("/api/v1/aws-deploy/ec2/launch", "POST", { instance_type: instanceType, volume_size_gb: volumeSizeGb }),
  deploy: (publicIp: string, username = "ubuntu") =>
    req<JobHandle>("/api/v1/aws-deploy/deploy", "POST", { public_ip: publicIp, username }),
  pollJob: (jobId: string, cursor: number) =>
    req<JobPollResult>(`/api/v1/aws-deploy/jobs/${jobId}?cursor=${cursor}`),
  cancelJob: (jobId: string) => req<{ ok: boolean }>(`/api/v1/aws-deploy/jobs/${jobId}/cancel`, "POST"),
};
