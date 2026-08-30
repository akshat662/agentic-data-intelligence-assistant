/**
 * Presentation-only number formatting for evidence values. Purely cosmetic: the underlying
 * value used for grounding/validation is always the raw number from `adia/evidence/renderer.py`
 * -- nothing here changes what's sent to or received from the backend.
 *
 * This is a heuristic, not a schema: `RenderedEvidence.key_values` is a bag of dotted-path keys
 * from whichever tool produced it (`adia/evidence/renderer.py::_walk_summary`), with no type
 * tag telling a client "this one is currency." The heuristics below key off naming conventions
 * already used consistently by every tool in `adia/tools/` (`share_of_total`, `*_count`, a
 * `metric_column`/`target_column` argument naming the field being measured) rather than
 * guessing from the value alone.
 */

const PERCENTAGE_KEY_PATTERN = /(?:^|_)(?:share|share_of_total|rate|pct|percentage)$/i;
const COUNT_KEY_PATTERN = /(?:^|_)(?:count|rank)$/i;
const CURRENCY_NAME_PATTERN = /sales|profit|price|revenue/i;

/**
 * Format one evidence value for display.
 *
 * @param key - The dotted-path key this value was found under (e.g. `"share_of_total"`,
 *   `"entities[0].total"`) -- only its final segment is used for pattern matching.
 * @param value - The raw value, exactly as `RenderedEvidence.key_values` holds it.
 * @param metricHint - The tool call's own `metric_column`/`target_column` argument, if any
 *   (e.g. `"Sales"`), used to decide whether a plain float should get a currency prefix.
 */
export function formatEvidenceValue(key: string, value: unknown, metricHint?: string): string {
  if (value === null || value === undefined) return "—";
  if (typeof value !== "number" || Number.isNaN(value)) {
    return typeof value === "object" ? JSON.stringify(value) : String(value);
  }

  const lastSegment = key.split(/[.[\]]/).filter(Boolean).pop() ?? key;

  if (PERCENTAGE_KEY_PATTERN.test(lastSegment)) {
    return `${(value * 100).toFixed(1)}%`;
  }
  if (COUNT_KEY_PATTERN.test(lastSegment) && Number.isInteger(value)) {
    return value.toLocaleString("en-US");
  }
  if (Number.isInteger(value)) {
    return value.toLocaleString("en-US");
  }

  const rounded = value.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
  const looksLikeCurrency = CURRENCY_NAME_PATTERN.test(metricHint ?? lastSegment);
  return looksLikeCurrency ? `$${rounded}` : rounded;
}

/** Pull a metric-name hint (e.g. `"Sales"`) out of one evidence record's tool arguments, if any. */
export function metricHintFromArguments(args: Record<string, unknown>): string | undefined {
  const metric = args["metric_column"] ?? args["target_column"];
  return typeof metric === "string" ? metric : undefined;
}
