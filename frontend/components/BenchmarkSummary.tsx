"use client";

import { BarChart3, TrendingUp } from "lucide-react";
import { useEffect, useState } from "react";
import { type BenchmarkMetric, SAMPLE_BENCHMARK } from "@/lib/benchmark";
import { cn } from "@/lib/utils";

export interface BenchmarkSummaryProps {
  className?: string;
}

/**
 * Benchmark cards comparing FinSage-7B vs the base model.
 *
 * Figures come from `/api/benchmark`, which reports whether they are a real
 * evaluation report or the built-in sample; the label reflects that source.
 * Falls back to the sample while loading or if the route is unavailable.
 */
export function BenchmarkSummary({ className }: BenchmarkSummaryProps) {
  const [metrics, setMetrics] = useState<BenchmarkMetric[]>(SAMPLE_BENCHMARK);
  const [source, setSource] = useState<"report" | "sample">("sample");

  useEffect(() => {
    let active = true;
    fetch("/api/benchmark")
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (active && data?.metrics?.length) {
          setMetrics(data.metrics);
          setSource(data.source === "report" ? "report" : "sample");
        }
      })
      .catch(() => {
        /* keep the sample fallback */
      });
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className={cn("space-y-3", className)}>
      <div className="flex items-center gap-2">
        <BarChart3 className="h-4 w-4 text-brand-400" aria-hidden />
        <h3 className="text-sm font-semibold text-slate-200">Benchmark summary</h3>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {metrics.map((m) => (
          <div key={m.label} className="rounded-xl border border-white/10 bg-ink-900/40 p-3">
            <p className="text-xs text-slate-400">{m.label}</p>
            <div className="mt-1 flex items-baseline gap-2">
              <span className="text-xl font-semibold text-slate-100">{m.value}</span>
              {m.delta ? (
                <span className="flex items-center gap-0.5 text-xs font-medium text-emerald-400">
                  <TrendingUp className="h-3 w-3" aria-hidden /> {m.delta}
                </span>
              ) : null}
            </div>
          </div>
        ))}
      </div>

      <p className="text-xs italic text-slate-500">
        {source === "report"
          ? "Live figures from the evaluation report (BENCHMARK_DATA)."
          : "Sample values. Provide real Phase 6 results via BENCHMARK_DATA to show live figures."}
      </p>
    </div>
  );
}

export default BenchmarkSummary;
