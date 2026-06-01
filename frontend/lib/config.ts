/**
 * Public frontend configuration.
 *
 * Only non-secret, browser-safe values live here. The backend API secret is
 * never read in this module — it is used exclusively server-side inside the
 * Next.js `app/api/*` proxy routes.
 */

/** Public API base URL (used for display / health checks). */
export const apiBaseUrl: string =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8080/v1";

/** Application display name. */
export const appName: string = process.env.NEXT_PUBLIC_APP_NAME ?? "FinSage-7B";

/**
 * Whether demo mode is enabled. In demo mode the proxy route returns a
 * deterministic mock when the backend is unreachable (or always, if forced).
 */
export const demoMode: boolean =
  (process.env.NEXT_PUBLIC_DEMO_MODE ?? "true").toLowerCase() === "true";

/** Client-side request timeout (ms) for calls to the local proxy route. */
export const requestTimeoutMs = 60_000;

export const config = {
  apiBaseUrl,
  appName,
  demoMode,
  requestTimeoutMs,
} as const;
