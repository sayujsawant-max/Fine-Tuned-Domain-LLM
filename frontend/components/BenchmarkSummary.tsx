import { BarChart3, TrendingUp } from "lucide-react";
import { cn } from "@/lib/utils";

export interface BenchmarkMetric {
  label: string;
  value: string;
  delta?: string;
}

/**
 * Placeholder benchmark figures. Replace with actual Phase 6 benchmark results
 * (reports/figures/comparison_summary.json) after a real evaluation run.
 */
const METRICS: readonly BenchmarkMetric[] = [
  { label: "Filing Q&A F1", value: "0.71", delta: "+0.14" },
  { label: "Risk Summary ROUGE-L", value: "0.38", delta: "+0.09" },
  { label: "Metric Extraction Accuracy", value: "0.86", delta: "+0.21" },
  { label: "Faithfulness Score", value: "0.92", delta: "+0.17" },
];

export interface BenchmarkSummaryProps {
  className?: string;
}

/** Benchmark cards (sample values) comparing FinSage-7B vs the base model. */
export function BenchmarkSummary({ className }: BenchmarkSummaryProps) {
  return (
    <div className={cn("space-y-3", className)}>
      <div className="flex items-center gap-2">
        <BarChart3 className="h-4 w-4 text-brand-400" aria-hidden />
        <h3 className="text-sm font-semibold text-slate-200">Benchmark summary</h3>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {METRICS.map((m) => (
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
        Sample values. Replace with actual Phase 6 benchmark results after real evaluation.
      </p>
    </div>
  );
}

export default BenchmarkSummary;
