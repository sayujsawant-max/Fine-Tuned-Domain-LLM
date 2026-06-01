import { ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

const STEPS = [
  "SEC Filing Excerpt",
  "FastAPI Wrapper",
  "vLLM Server",
  "FinSage-7B",
  "Grounded Answer",
] as const;

export interface ArchitectureFlowProps {
  className?: string;
}

/** Compact left-to-right request-flow diagram. */
export function ArchitectureFlow({ className }: ArchitectureFlowProps) {
  return (
    <div className={cn("space-y-3", className)}>
      <h3 className="text-sm font-semibold text-slate-200">How it works</h3>
      <ol className="flex flex-wrap items-center gap-2">
        {STEPS.map((step, i) => (
          <li key={step} className="flex items-center gap-2">
            <span
              className={cn(
                "rounded-lg border px-3 py-1.5 text-xs font-medium",
                i === STEPS.length - 1
                  ? "border-brand-500/40 bg-brand-500/10 text-brand-300"
                  : "border-white/10 bg-white/5 text-slate-300",
              )}
            >
              {step}
            </span>
            {i < STEPS.length - 1 ? (
              <ChevronRight className="h-4 w-4 text-slate-600" aria-hidden />
            ) : null}
          </li>
        ))}
      </ol>
    </div>
  );
}

export default ArchitectureFlow;
