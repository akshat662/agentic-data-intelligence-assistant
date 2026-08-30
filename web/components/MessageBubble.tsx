import type { Turn } from "@/hooks/useChat";

import { EvidencePanel, evidenceElementId } from "./EvidencePanel";
import { ProgressTimeline } from "./ProgressTimeline";
import { ValidationBadge } from "./ValidationBadge";

interface MessageBubbleProps {
  turn: Turn;
  isActive: boolean;
}

// Matches the `[[ev_<tool>_<hash>]]` citation markers `adia/validate/static.py` itself
// requires every grounded answer to use -- kept in the split group so `.split()` retains
// the markers instead of discarding them.
const CITATION_PATTERN = /(\[\[ev_[a-zA-Z0-9_]+\]\])/g;
const CITATION_ID_PATTERN = /^\[\[(ev_[a-zA-Z0-9_]+)\]\]$/;

/** Renders answer text with every `[[ev_...]]` citation as a clickable chip that jumps to
 * (and expands) the matching card in the evidence panel below, instead of showing the raw
 * bracket syntax verbatim.
 */
function AnswerText({ text }: { text: string }) {
  return (
    <p className="text-sm leading-relaxed whitespace-pre-wrap text-zinc-800 dark:text-zinc-200">
      {text.split(CITATION_PATTERN).map((part, index) => {
        const match = part.match(CITATION_ID_PATTERN);
        if (!match) return <span key={index}>{part}</span>;
        const evidenceId = match[1];
        return (
          <a
            key={index}
            href={`#${evidenceElementId(evidenceId)}`}
            onClick={() => {
              const el = document.getElementById(evidenceElementId(evidenceId));
              if (el instanceof HTMLDetailsElement) el.open = true;
            }}
            className="ml-0.5 inline-flex items-center rounded-full bg-blue-50 px-1.5 py-0.5 align-middle font-mono text-[11px] whitespace-nowrap text-blue-700 no-underline hover:bg-blue-100 dark:bg-blue-900/30 dark:text-blue-300 dark:hover:bg-blue-900/50"
            title="Jump to this evidence record"
          >
            {evidenceId}
          </a>
        );
      })}
    </p>
  );
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

        {turn.final?.answer && <AnswerText text={turn.final.answer} />}

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
