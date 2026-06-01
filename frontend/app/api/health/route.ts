/**
 * Server-side health proxy.
 *
 * Reports the frontend status and, when reachable, the backend `/v1/health`
 * status. Never exposes the API secret (the health endpoint is public, but we
 * keep the call server-side for a consistent origin).
 */

import { NextResponse } from "next/server";

const API_BASE_URL = process.env.API_BASE_URL ?? "http://localhost:8080/v1";
const TIMEOUT_MS = 5_000;

export async function GET(): Promise<NextResponse> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const upstream = await fetch(`${API_BASE_URL}/health`, {
      signal: controller.signal,
      cache: "no-store",
    });
    clearTimeout(timeout);
    const backend = await upstream.json().catch(() => null);
    return NextResponse.json({
      frontend: "ok",
      backend_available: upstream.ok,
      backend,
    });
  } catch {
    clearTimeout(timeout);
    return NextResponse.json({
      frontend: "ok",
      backend_available: false,
      backend: null,
    });
  }
}
