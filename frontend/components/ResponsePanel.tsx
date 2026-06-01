"use client";

import { Bot, Clock, Hash, Sparkles } from "lucide-react";
import type { ApiError, ChatResponse } from "@/lib/api";
import { getTaskType } from "@/lib/taskTypes";
import { cn } from "@/lib/utils";
import { CopyButton } from "@/components/CopyButton";
import { ErrorAlert } from "@/components/ErrorAlert";
import { StatusBadge } from "@/components/StatusBadge";

export interface ResponsePanelProps {
  response: ChatResponse | null;
  loading: boolean;
  error: ApiError | null;
  className?: string;
}

function Meta({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center gap-1.5 text-xs text-slate-400">
      <span className="text-slate-500">{icon}</span>
      <span className="text-slate-500">{label}:</span>
      <span className="font-mono text-slate-300">{value}</span>
    </div>
  );
}

/** Renders the analysis result, with empty / loading / error states. */
export function ResponsePanel({ response, loading, error, className }: ResponsePanelProps) {
  if (loading) {
    return (
      <div className={cn("space-y-3", className)} aria-busy="true" aria-live="polite">
        <div className="h-4 w-1/3 animate-pulse rounded bg-white/10" />
        <div className="h-3 w-full animate-pulse rounded bg-white/10" />
        <div className="h-3 w-11/12 animate-pulse rounded bg-white/10" />
        <div className="h-3 w-4/5 animate-pulse rounded bg-white/10" />
        <div className="h-3 w-2/3 animate-pulse rounded bg-white/10" />
      </div>
    );
  }

  if (error) {
    return (
      <ErrorAlert
        className={className}
        message={error.message}
        detail={error.detail}
        requestId={error.requestId}
      />
    );
  }

  if (!response) {
    return (
      <div
        className={cn(
          "flex flex-col items-center justify-center rounded-xl border border-dashed border-white/10 px-6 py-12 text-center",
          className,
        )}
      >
        <Sparkles className="mb-3 h-6 w-6 text-slate-600" aria-hidden />
        <p className="text-sm text-slate-400">
          Enter a filing excerpt and a question, then click{" "}
          <span className="font-medium text-slate-200">Analyze Filing</span>.
        </p>
      </div>
    );
  }

  const taskLabel = response.task_type
    ? getTaskType(response.task_type)?.label ?? response.task_type
    : null;

  return (
    <div className={cn("space-y-4", className)} aria-live="polite">
      <div className="flex flex-wrap items-center gap-2">
        {response.demo_mode ? (
          <StatusBadge tone="warning">Demo mode</StatusBadge>
        ) : (
          <StatusBadge tone="success">Live backend</StatusBadge>
        )}
        {taskLabel ? <StatusBadge tone="accent">{taskLabel}</StatusBadge> : null}
      </div>

      <div className="rounded-xl border border-white/10 bg-ink-900/60 p-4">
        <div className="mb-3 flex items-center justify-between">
          <span className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-slate-400">
            <Bot className="h-4 w-4 text-brand-400" aria-hidden /> Answer
          </span>
          <CopyButton value={response.answer} label="Copy answer" />
        </div>
        <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-100">
          {response.answer}
        </p>
      </div>

      <div className="flex flex-wrap gap-x-5 gap-y-2">
        <Meta icon={<Bot className="h-3.5 w-3.5" />} label="model" value={response.model} />
        <Meta
          icon={<Clock className="h-3.5 w-3.5" />}
          label="latency"
          value={`${Math.round(response.latency_ms)} ms`}
        />
        <Meta
          icon={<Hash className="h-3.5 w-3.5" />}
          label="request_id"
          value={response.request_id}
        />
      </div>

      {response.disclaimer ? (
        <p className="rounded-lg border border-amber-500/20 bg-amber-500/5 px-3 py-2 text-xs leading-relaxed text-amber-200/80">
          {response.disclaimer}
        </p>
      ) : null}
    </div>
  );
}

export default ResponsePanel;
