/**
 * Browser-side API client.
 *
 * Calls the local Next.js proxy route (`/api/chat`) — never the FastAPI backend
 * directly — so the API secret stays server-side. The proxy injects the
 * `X-API-Key` header from a server-only environment variable.
 */

import { requestTimeoutMs } from "@/lib/config";

/** Request payload sent to the proxy (mirrors the backend ChatRequest). */
export interface ChatRequest {
  question: string;
  filing_excerpt: string;
  task_type?: string | null;
  max_tokens?: number;
  temperature?: number;
  include_disclaimer?: boolean;
}

/** Successful response (mirrors the backend ChatResponse, plus a demo flag). */
export interface ChatResponse {
  answer: string;
  model: string;
  task_type: string | null;
  disclaimer: string | null;
  request_id: string;
  latency_ms: number;
  /** True when the answer was produced by demo mode (no real backend). */
  demo_mode?: boolean;
}

/** Normalised API error surfaced to the UI. */
export class ApiError extends Error {
  readonly status: number;
  readonly requestId: string | null;
  readonly detail: string | null;

  constructor(
    message: string,
    status: number,
    detail: string | null = null,
    requestId: string | null = null,
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.requestId = requestId;
  }
}

/**
 * Analyze a filing excerpt by calling the local proxy route.
 *
 * @param request - The chat request payload.
 * @param signal - Optional abort signal; a default timeout is applied otherwise.
 * @returns The typed chat response.
 * @throws {ApiError} On timeout, network failure, or a non-2xx proxy response.
 */
export async function analyzeFiling(
  request: ChatRequest,
  signal?: AbortSignal,
): Promise<ChatResponse> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), requestTimeoutMs);
  // Forward an externally-provided signal into our controller.
  if (signal) {
    if (signal.aborted) controller.abort();
    else signal.addEventListener("abort", () => controller.abort(), { once: true });
  }

  let response: Response;
  try {
    response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timeout);
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiError("The request timed out. Is the backend running?", 408);
    }
    throw new ApiError("Network error contacting the demo proxy.", 0);
  }
  clearTimeout(timeout);

  let data: unknown = null;
  try {
    data = await response.json();
  } catch {
    // Body was not JSON; fall through to error handling below.
  }

  if (!response.ok) {
    const body = (data ?? {}) as {
      error?: string;
      detail?: string;
      request_id?: string;
    };
    throw new ApiError(
      body.error ?? `Request failed (HTTP ${response.status}).`,
      response.status,
      body.detail ?? null,
      body.request_id ?? null,
    );
  }

  return data as ChatResponse;
}
