interface ValidationBadgeProps {
  validationPassed: boolean | null;
  refused: boolean;
  pending?: boolean;
}

const STYLES = {
  pending: "bg-zinc-100 text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400",
  refused: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300",
  passed: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300",
  failed: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
  unknown: "bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300",
};

export function ValidationBadge({ validationPassed, refused, pending }: ValidationBadgeProps) {
  let label = "Unknown";
  let style = STYLES.unknown;

  if (pending) {
    label = "Running";
    style = STYLES.pending;
  } else if (refused) {
    label = "Refused";
    style = STYLES.refused;
  } else if (validationPassed === true) {
    label = "Validated";
    style = STYLES.passed;
  } else if (validationPassed === false) {
    label = "Failed validation";
    style = STYLES.failed;
  }

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${style}`}
    >
      {pending && (
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" aria-hidden />
      )}
      {label}
    </span>
  );
}
