import { cn } from "@/lib/utils";

type Tone = "brand" | "accent" | "neutral" | "warning" | "success";

const TONES: Record<Tone, string> = {
  brand: "border-brand-500/30 bg-brand-500/10 text-brand-400",
  accent: "border-accent-500/30 bg-accent-500/10 text-accent-400",
  neutral: "border-white/10 bg-white/5 text-slate-300",
  warning: "border-amber-500/30 bg-amber-500/10 text-amber-300",
  success: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
};

export interface StatusBadgeProps {
  children: React.ReactNode;
  tone?: Tone;
  className?: string;
}

/** Small pill badge used for status / labels. */
export function StatusBadge({ children, tone = "neutral", className }: StatusBadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium",
        TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export default StatusBadge;
