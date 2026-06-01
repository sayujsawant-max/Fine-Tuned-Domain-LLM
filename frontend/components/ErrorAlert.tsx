import { AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";

export interface ErrorAlertProps {
  /** Primary error message. */
  message: string;
  /** Optional secondary detail. */
  detail?: string | null;
  /** Optional request id for correlation. */
  requestId?: string | null;
  className?: string;
}

/** Accessible error alert banner. */
export function ErrorAlert({ message, detail, requestId, className }: ErrorAlertProps) {
  return (
    <div
      role="alert"
      className={cn(
        "flex gap-3 rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-200",
        className,
      )}
    >
      <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-red-400" aria-hidden />
      <div className="space-y-1">
        <p className="font-medium">{message}</p>
        {detail ? <p className="text-red-200/80">{detail}</p> : null}
        {requestId ? (
          <p className="font-mono text-xs text-red-200/60">request_id: {requestId}</p>
        ) : null}
      </div>
    </div>
  );
}

export default ErrorAlert;
