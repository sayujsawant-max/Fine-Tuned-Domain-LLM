"use client";

import { Loader2, Search } from "lucide-react";
import { cn } from "@/lib/utils";

export interface AnalyzeButtonProps {
  onClick: () => void;
  loading?: boolean;
  disabled?: boolean;
  className?: string;
}

/** Primary call-to-action button with loading and disabled states. */
export function AnalyzeButton({ onClick, loading, disabled, className }: AnalyzeButtonProps) {
  const isDisabled = disabled || loading;
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={isDisabled}
      aria-busy={loading}
      className={cn(
        "inline-flex w-full items-center justify-center gap-2 rounded-xl bg-brand-500 px-5 py-3 text-sm font-semibold text-ink-950 transition hover:bg-brand-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-400 focus-visible:ring-offset-2 focus-visible:ring-offset-ink-950 disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
    >
      {loading ? (
        <>
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> Analyzing…
        </>
      ) : (
        <>
          <Search className="h-4 w-4" aria-hidden /> Analyze Filing
        </>
      )}
    </button>
  );
}

export default AnalyzeButton;
