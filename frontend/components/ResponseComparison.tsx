"use client";

import { ArrowRight, Bot, Info } from "lucide-react";
import type { ChatResponse } from "@/lib/api";
import { cn } from "@/lib/utils";
import { StatusBadge } from "@/components/StatusBadge";

export interface ResponseComparisonProps {
  /** The FinSage-7B response to compare against the base model. */
  finsage: ChatResponse | null;
  className?: string;
}

/**
 * Representative (synthetic) base-Mistral answer used for the side-by-side
 * comparison. The Phase 8 backend does not expose a comparison endpoint, so the
 * base column is always demo content and is clearly labelled as such.
 */
const BASE_MISTRAL_SAMPLE =
  "The company seems to face various risks such as supply issues and competition, and overall performance looks reasonable. Investors may want to consider the stock given the growth, though there are some uncertainties to keep in mind.";

const IMPROVEMENT_NOTES = [
  "Grounded strictly in the provided excerpt — no outside claims.",
  "Avoids investment advice and speculative recommendations.",
  "More specific: names concrete risks/metrics instead of vague generalities.",
];

/** Side-by-side base-Mistral vs FinSage-7B comparison (demo). */
export function ResponseComparison({ finsage, className }: ResponseComparisonProps) {
  return (
    <div className={cn("space-y-4", className)}>
      <div className="flex items-start gap-2 rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs text-slate-400">
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-slate-500" aria-hidden />
        <span>
          Comparison sample shown for demo purposes unless backend comparison mode is
          configured. The base-Mistral column is illustrative.
        </span>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-xl border border-white/10 bg-ink-900/40 p-4">
          <div className="mb-2 flex items-center justify-between">
            <span className="flex items-center gap-1.5 text-sm font-medium text-slate-300">
              <Bot className="h-4 w-4 text-slate-500" aria-hidden /> Base Mistral-7B
            </span>
            <StatusBadge tone="neutral">Sample</StatusBadge>
          </div>
          <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-400">
            {BASE_MISTRAL_SAMPLE}
          </p>
        </div>

        <div className="rounded-xl border border-brand-500/30 bg-brand-500/5 p-4 shadow-glow">
          <div className="mb-2 flex items-center justify-between">
            <span className="flex items-center gap-1.5 text-sm font-medium text-brand-300">
              <Bot className="h-4 w-4 text-brand-400" aria-hidden /> FinSage-7B
            </span>
            <StatusBadge tone={finsage?.demo_mode ? "warning" : "success"}>
              {finsage?.demo_mode ? "Demo" : finsage ? "Live" : "Awaiting"}
            </StatusBadge>
          </div>
          <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-100">
            {finsage
              ? finsage.answer
              : "Run an analysis above to populate the FinSage-7B column."}
          </p>
        </div>
      </div>

      <div className="rounded-xl border border-white/10 bg-ink-900/40 p-4">
        <p className="mb-2 text-sm font-medium text-slate-200">Why fine-tuning helps</p>
        <ul className="space-y-1.5">
          {IMPROVEMENT_NOTES.map((note) => (
            <li key={note} className="flex items-start gap-2 text-sm text-slate-400">
              <ArrowRight className="mt-0.5 h-3.5 w-3.5 shrink-0 text-brand-400" aria-hidden />
              {note}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default ResponseComparison;
