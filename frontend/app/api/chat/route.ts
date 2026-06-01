/**
 * Server-side proxy for the FinSage-7B `/v1/chat` endpoint.
 *
 * The browser calls this route (never the FastAPI backend directly). The route
 * injects the `X-API-Key` header from a server-only environment variable, so
 * the API secret is never shipped to the client.
 *
 * If the backend is unreachable and demo mode is enabled, a deterministic mock
 * response is returned so the demo works without any running services.
 */

import { NextResponse } from "next/server";
import type { ChatRequest } from "@/lib/api";
import { getMockAnalysisResponse } from "@/lib/mockResponses";

// Server-only env (NOT prefixed with NEXT_PUBLIC_ → never exposed to the browser).
const API_BASE_URL = process.env.API_BASE_URL ?? "http://localhost:8080/v1";
const API_SECRET_KEY = process.env.API_SECRET_KEY ?? "change-me";
const DEMO_MODE = (process.env.NEXT_PUBLIC_DEMO_MODE ?? "true").toLowerCase() === "true";
const TIMEOUT_MS = 60_000;

function isValidRequest(body: unknown): body is ChatRequest {
  if (typeof body !== "object" || body === null) return false;
  const b = body as Record<string, unknown>;
  return (
    typeof b.question === "string" &&
    b.question.trim().length > 0 &&
    typeof b.filing_excerpt === "string" &&
    b.filing_excerpt.trim().length > 0
  );
}

export async function POST(request: Request): Promise<NextResponse> {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { error: "invalid_request", detail: "Request body must be valid JSON." },
      { status: 400 },
    );
  }

  if (!isValidRequest(body)) {
    return NextResponse.json(
      {
        error: "invalid_request",
        detail: "question and filing_excerpt are required and must be non-empty.",
      },
      { status: 422 },
    );
  }

  const payload: ChatRequest = {
    question: body.question,
    filing_excerpt: body.filing_excerpt,
    task_type: body.task_type ?? null,
    max_tokens: body.max_tokens ?? 256,
    temperature: body.temperature ?? 0.0,
    include_disclaimer: body.include_disclaimer ?? true,
  };

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const upstream = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": API_SECRET_KEY,
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
      cache: "no-store",
    });
    clearTimeout(timeout);

    const data = await upstream.json().catch(() => null);
    if (!upstream.ok) {
      // Surface the backend's normalised error shape to the client.
      return NextResponse.json(
        data ?? { error: `Backend error (HTTP ${upstream.status}).` },
        { status: upstream.status },
      );
    }
    return NextResponse.json(data, { status: 200 });
  } catch (err) {
    clearTimeout(timeout);
    // Backend unreachable / timed out. Fall back to demo mode if enabled.
    if (DEMO_MODE) {
      return NextResponse.json(getMockAnalysisResponse(payload), { status: 200 });
    }
    const aborted = err instanceof DOMException && err.name === "AbortError";
    return NextResponse.json(
      {
        error: "backend_unavailable",
        detail: aborted
          ? "The backend request timed out."
          : "Could not reach the FastAPI backend. Start it or enable demo mode.",
      },
      { status: 503 },
    );
  }
}
