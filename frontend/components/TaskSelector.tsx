"use client";

import { TASK_TYPES, getTaskType } from "@/lib/taskTypes";
import { cn } from "@/lib/utils";

export interface TaskSelectorProps {
  value: string;
  onChange: (taskType: string) => void;
  className?: string;
}

/** Task-type picker (dropdown) with a short description of the active task. */
export function TaskSelector({ value, onChange, className }: TaskSelectorProps) {
  const active = getTaskType(value);

  return (
    <div className={cn("space-y-2", className)}>
      <label htmlFor="task-type" className="text-sm font-medium text-slate-200">
        Task type
      </label>
      <select
        id="task-type"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-xl border border-white/10 bg-ink-900/80 px-3 py-2.5 text-sm text-slate-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500"
      >
        {TASK_TYPES.map((t) => (
          <option key={t.value} value={t.value}>
            {t.label}
          </option>
        ))}
      </select>
      {active ? (
        <p className="text-xs text-slate-400">{active.description}</p>
      ) : null}
    </div>
  );
}

export default TaskSelector;
