"use client";

import { FileText, X } from "lucide-react";
import { FILING_SAMPLES } from "@/lib/samples";
import { cn } from "@/lib/utils";

export interface FilingInputProps {
  value: string;
  onChange: (value: string) => void;
  selectedSample: string;
  onSampleChange: (sampleId: string) => void;
  className?: string;
}

const MAX_CHARS = 8000;

/** Filing-excerpt textarea with sample picker, char count, and clear button. */
export function FilingInput({
  value,
  onChange,
  selectedSample,
  onSampleChange,
  className,
}: FilingInputProps) {
  return (
    <div className={cn("space-y-3", className)}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <label
          htmlFor="filing-excerpt"
          className="flex items-center gap-2 text-sm font-medium text-slate-200"
        >
          <FileText className="h-4 w-4 text-brand-400" aria-hidden />
          Filing excerpt
        </label>
        <div className="flex items-center gap-2">
          <label htmlFor="filing-sample" className="sr-only">
            Load a sample excerpt
          </label>
          <select
            id="filing-sample"
            value={selectedSample}
            onChange={(e) => onSampleChange(e.target.value)}
            className="rounded-md border border-white/10 bg-ink-800 px-2 py-1.5 text-xs text-slate-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500"
          >
            <option value="">Load sample…</option>
            {FILING_SAMPLES.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="relative">
        <textarea
          id="filing-excerpt"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          maxLength={MAX_CHARS}
          rows={10}
          placeholder="Paste a 10-K / 10-Q / 8-K excerpt here, or load a sample above…"
          aria-describedby="filing-charcount"
          className="w-full resize-y rounded-xl border border-white/10 bg-ink-900/80 p-4 text-sm leading-relaxed text-slate-100 placeholder:text-slate-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500"
        />
        {value ? (
          <button
            type="button"
            onClick={() => {
              onChange("");
              onSampleChange("");
            }}
            aria-label="Clear filing excerpt"
            className="absolute right-3 top-3 inline-flex items-center gap-1 rounded-md border border-white/10 bg-ink-800/80 px-2 py-1 text-xs text-slate-300 hover:bg-white/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500"
          >
            <X className="h-3.5 w-3.5" /> Clear
          </button>
        ) : null}
      </div>

      <p id="filing-charcount" className="text-right text-xs text-slate-500">
        {value.length.toLocaleString()} / {MAX_CHARS.toLocaleString()} characters
      </p>
    </div>
  );
}

export default FilingInput;
