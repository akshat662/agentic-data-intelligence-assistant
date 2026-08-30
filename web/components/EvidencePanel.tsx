import { formatEvidenceValue, metricHintFromArguments } from "@/lib/format";
import type { RenderedEvidence } from "@/lib/types";

interface EvidencePanelProps {
  evidence: RenderedEvidence[];
}

/** DOM id an evidence card is addressable by -- citation chips in `MessageBubble` link here. */
export function evidenceElementId(evidenceId: string): string {
  return `evidence-${evidenceId}`;
}

/**
 * Renders each `RenderedEvidence` record produced during a run. This is already the bounded,
 * tool-agnostic summary the Synthesizer itself is shown (`adia/evidence/renderer.py`) -- never
 * the raw, unbounded tool output -- so nothing here needs to further truncate anything.
 * Numbers are formatted for readability only (`lib/format.ts`); the underlying value used for
 * grounding is unaffected.
 */
export function EvidencePanel({ evidence }: EvidencePanelProps) {
  if (evidence.length === 0) return null;

  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-xs font-medium tracking-wide text-zinc-500 uppercase dark:text-zinc-400">
        Evidence ({evidence.length})
      </span>
      <div className="flex flex-col gap-1.5">
        {evidence.map((item) => {
          const metricHint = metricHintFromArguments(item.arguments);
          return (
            <details
              key={item.evidence_id}
              id={evidenceElementId(item.evidence_id)}
              className="rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm open:pb-3 target:ring-2 target:ring-blue-400 dark:border-zinc-800 dark:bg-zinc-900"
            >
              <summary className="flex cursor-pointer items-center gap-2 font-mono text-xs text-zinc-600 select-none dark:text-zinc-400">
                <span className="rounded bg-zinc-200 px-1.5 py-0.5 font-sans font-medium text-zinc-700 dark:bg-zinc-700 dark:text-zinc-200">
                  {item.tool}
                </span>
                <span className="truncate">{item.evidence_id}</span>
              </summary>
              <dl className="mt-2 grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1 text-xs">
                {Object.entries(item.key_values).map(([key, value]) => (
                  <div key={key} className="contents">
                    <dt className="text-zinc-500 dark:text-zinc-400">{key}</dt>
                    <dd className="font-mono text-zinc-800 dark:text-zinc-200">
                      {formatEvidenceValue(key, value, metricHint)}
                    </dd>
                  </div>
                ))}
              </dl>
            </details>
          );
        })}
      </div>
    </div>
  );
}
