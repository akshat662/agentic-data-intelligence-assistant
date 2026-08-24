import type { StreamPhaseEvent } from "@/lib/types";

interface Step {
  node: string;
  label: string;
}

const PLANNER_PATH: Step[] = [
  { node: "feasibility", label: "Checking feasibility" },
  { node: "planner", label: "Creating investigation plan" },
  { node: "execute_tools", label: "Running analysis tools" },
  { node: "synthesizer", label: "Generating evidence" },
  { node: "validation", label: "Validating answer" },
];

const REFUSAL_PATH: Step[] = [
  { node: "feasibility", label: "Checking feasibility" },
  { node: "refusal", label: "Question could not be answered" },
  { node: "validation", label: "Validating answer" },
];

interface ProgressTimelineProps {
  phases: StreamPhaseEvent[];
  /** Whether this turn is the one currently streaming -- only it gets a spinner on its next
   * pending step; a finished or errored turn just shows its completed history. */
  isActive: boolean;
}

export function ProgressTimeline({ phases, isActive }: ProgressTimelineProps) {
  const completed = new Set(phases.map((p) => p.node));
  // The graph has exactly two mutually-exclusive paths after `feasibility`; once a `refusal`
  // event has actually arrived we know which one this run took, otherwise assume the common
  // (planner) path -- a brief, harmless guess for the one render frame before we know for sure.
  const steps = completed.has("refusal") ? REFUSAL_PATH : PLANNER_PATH;
  const firstPendingIndex = steps.findIndex((step) => !completed.has(step.node));

  return (
    <ol className="flex flex-col gap-1.5">
      {steps.map((step, index) => {
        const done = completed.has(step.node);
        const isCurrent = isActive && !done && index === firstPendingIndex;
        return (
          <li key={step.node} className="flex items-center gap-2 text-sm">
            <StepIcon done={done} current={isCurrent} />
            <span
              className={
                done
                  ? "text-zinc-800 dark:text-zinc-200"
                  : isCurrent
                    ? "text-zinc-600 dark:text-zinc-400"
                    : "text-zinc-400 dark:text-zinc-600"
              }
            >
              {step.label}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

function StepIcon({ done, current }: { done: boolean; current: boolean }) {
  if (done) {
    return (
      <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-emerald-500 text-[10px] font-bold text-white">
        ✓
      </span>
    );
  }
  if (current) {
    return (
      <span
        className="h-4 w-4 shrink-0 animate-spin rounded-full border-2 border-zinc-300 border-t-zinc-600 dark:border-zinc-700 dark:border-t-zinc-300"
        aria-hidden
      />
    );
  }
  return (
    <span className="h-4 w-4 shrink-0 rounded-full border-2 border-zinc-200 dark:border-zinc-700" />
  );
}
