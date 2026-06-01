/**
 * Server route serving benchmark figures with explicit provenance.
 *
 * Returns operator-supplied evaluation metrics when `BENCHMARK_DATA` (a JSON
 * string) is set — `source: "report"` — otherwise the built-in illustrative
 * sample — `source: "sample"`. Keeping this server-side means the figures are
 * data-driven rather than hardcoded in the client bundle.
 */

import { NextResponse } from "next/server";
import { parseBenchmark, SAMPLE_BENCHMARK, type BenchmarkPayload } from "@/lib/benchmark";

// Read BENCHMARK_DATA at request time (not build time) so runtime env is honoured.
export const dynamic = "force-dynamic";

export async function GET(): Promise<NextResponse<BenchmarkPayload>> {
  const raw = process.env.BENCHMARK_DATA;
  if (raw) {
    try {
      const metrics = parseBenchmark(JSON.parse(raw));
      if (metrics.length > 0) {
        return NextResponse.json({ source: "report", metrics });
      }
    } catch {
      // Malformed BENCHMARK_DATA — fall through to the sample.
    }
  }
  return NextResponse.json({ source: "sample", metrics: SAMPLE_BENCHMARK });
}
