/**
 * Benchmark data shared by the API route and the summary component.
 *
 * The figures are served by `/api/benchmark`, which reports a `source`:
 * `"report"` when a real evaluation report is supplied (via the
 * `BENCHMARK_DATA` env var), or `"sample"` for the built-in illustrative
 * values. This keeps provenance explicit instead of hardcoding numbers.
 */

export interface BenchmarkMetric {
  label: string;
  value: string;
  delta?: string;
}

export interface BenchmarkPayload {
  source: "report" | "sample";
  metrics: BenchmarkMetric[];
}

/** Built-in illustrative metrics (clearly labelled as a sample in the UI). */
export const SAMPLE_BENCHMARK: BenchmarkMetric[] = [
  { label: "Filing Q&A F1", value: "0.71", delta: "+0.14" },
  { label: "Risk Summary ROUGE-L", value: "0.38", delta: "+0.09" },
  { label: "Metric Extraction Accuracy", value: "0.86", delta: "+0.21" },
  { label: "Faithfulness Score", value: "0.92", delta: "+0.17" },
];

/**
 * Coerce arbitrary parsed JSON into a list of benchmark metrics.
 *
 * Accepts either an array of `{label, value, delta}` or an object mapping a
 * label to a numeric/string value. Returns an empty array when nothing usable
 * is found.
 *
 * @param data - Parsed JSON from the `BENCHMARK_DATA` env var.
 */
export function parseBenchmark(data: unknown): BenchmarkMetric[] {
  if (Array.isArray(data)) {
    return data
      .filter((m): m is Record<string, unknown> => typeof m === "object" && m !== null)
      .map((m) => ({
        label: String(m.label ?? ""),
        value: String(m.value ?? ""),
        delta: m.delta != null ? String(m.delta) : undefined,
      }))
      .filter((m) => m.label && m.value);
  }
  if (data && typeof data === "object") {
    return Object.entries(data as Record<string, unknown>)
      .filter(([, v]) => typeof v === "number" || typeof v === "string")
      .map(([label, v]) => ({ label, value: String(v) }));
  }
  return [];
}
