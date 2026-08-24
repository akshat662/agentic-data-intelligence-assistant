import type { Turn } from "@/hooks/useChat";

import { EvidencePanel } from "./EvidencePanel";
import { ProgressTimeline } from "./ProgressTimeline";
import { ValidationBadge } from "./ValidationBadge";

interface MessageBubbleProps {
  turn: Turn;
  isActive: boolean;
}

export function MessageBubble({ turn, isActive }: MessageBubbleProps) {
  const pending = !turn.final && !turn.error;

  return (
    <div className="flex flex-col gap-3">
      <div className="ml-auto max-w-xl rounded-2xl rounded-br-sm bg-zinc-900 px-4 py-2.5 text-sm text-white dark:bg-zinc-100 dark:text-zinc-900">
        {turn.question}
      </div>

      <div className="mr-auto flex w-full max-w-2xl flex-col gap-3 rounded-2xl rounded-bl-sm border border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-950">
        <ProgressTimeline phases={turn.phases} isActive={isActive} />

        {turn.error && (
          <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700 dark:bg-red-900/20 dark:text-red-300">
            {turn.error}
          </p>
        )}

        {turn.final?.answer && (
          <p className="text-sm leading-relaxed whitespace-pre-wrap text-zinc-800 dark:text-zinc-200">
            {turn.final.answer}
          </p>
        )}

        {!pending && !turn.error && (
          <div className="flex flex-wrap items-center gap-2">
            <ValidationBadge
              validationPassed={turn.final?.validation_passed ?? null}
              refused={turn.final?.refused ?? false}
            />
            {turn.final?.tools_used.map((tool) => (
              <span
                key={tool}
                className="rounded-full bg-zinc-100 px-2.5 py-1 text-xs text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300"
              >
                {tool}
              </span>
            ))}
            {turn.final && (
              <span className="text-xs text-zinc-400 dark:text-zinc-500">
                {Math.round(turn.final.duration_ms)} ms
              </span>
            )}
          </div>
        )}

        <EvidencePanel evidence={turn.evidence} />
      </div>
    </div>
  );
}
