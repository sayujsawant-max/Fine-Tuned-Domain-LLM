"use client";

import { Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

export interface QuestionInputProps {
  value: string;
  onChange: (value: string) => void;
  /** Suggested question for the active task type. */
  suggestedQuestion: string;
  className?: string;
}

const MAX_CHARS = 500;

/** Question input with a "use suggested" button and a max-length display. */
export function QuestionInput({
  value,
  onChange,
  suggestedQuestion,
  className,
}: QuestionInputProps) {
  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-center justify-between gap-2">
        <label htmlFor="question" className="text-sm font-medium text-slate-200">
          Question
        </label>
        {suggestedQuestion && suggestedQuestion !== value ? (
          <button
            type="button"
            onClick={() => onChange(suggestedQuestion)}
            className="inline-flex items-center gap-1.5 rounded-md border border-accent-500/30 bg-accent-500/10 px-2.5 py-1 text-xs font-medium text-accent-400 hover:bg-accent-500/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500"
          >
            <Sparkles className="h-3.5 w-3.5" /> Use suggested
          </button>
        ) : null}
      </div>
      <textarea
        id="question"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        maxLength={MAX_CHARS}
        rows={2}
        placeholder="Ask a question grounded in the excerpt above…"
        aria-describedby="question-charcount"
        className="w-full resize-y rounded-xl border border-white/10 bg-ink-900/80 p-3 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500"
      />
      <p id="question-charcount" className="text-right text-xs text-slate-500">
        {value.length} / {MAX_CHARS}
      </p>
    </div>
  );
}

export default QuestionInput;
