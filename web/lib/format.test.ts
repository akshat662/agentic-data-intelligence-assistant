/**
 * Tests for lib/format.ts, using Node's built-in test runner and native TypeScript support --
 * no new devDependency (jest/vitest) added just to test two pure functions. Run with:
 *   node --test lib/format.test.ts
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { formatEvidenceValue, metricHintFromArguments } from "./format.ts";

test("formats a currency-hinted float with two decimals and a dollar sign", () => {
  assert.equal(formatEvidenceValue("overall_mean", 452.70927612344343, "Sales"), "$452.71");
});

test("formats a non-currency float with two decimals, no dollar sign", () => {
  assert.equal(formatEvidenceValue("mean", 0.1739226779820839, "Discount"), "0.17");
});

test("formats a share_of_total as a percentage", () => {
  assert.equal(formatEvidenceValue("share_of_total", 0.39467256148485264), "39.5%");
});

test("formats a nested rows[0].share_of_total key the same way", () => {
  assert.equal(formatEvidenceValue("entities[0].share_of_total", 0.226), "22.6%");
});

test("formats a count as a plain grouped integer, not a decimal", () => {
  assert.equal(formatEvidenceValue("row_count", 9994), "9,994");
});

test("formats a rank as a plain integer", () => {
  assert.equal(formatEvidenceValue("rank", 1), "1");
});

test("infers currency from metricHint even when the key itself doesn't say so", () => {
  assert.equal(formatEvidenceValue("total", 836154.033, "Sales"), "$836,154.03");
});

test("does not add a dollar sign without a currency-sounding key or hint", () => {
  assert.equal(formatEvidenceValue("total", 3.756903086085544, "Quantity"), "3.76");
});

test("null and undefined render as an em dash", () => {
  assert.equal(formatEvidenceValue("mean", null), "—");
  assert.equal(formatEvidenceValue("std", undefined), "—");
});

test("non-numeric values pass through as strings", () => {
  assert.equal(formatEvidenceValue("group", "Technology"), "Technology");
  assert.equal(formatEvidenceValue("group", false), "false");
});

test("objects/arrays fall back to JSON", () => {
  assert.equal(formatEvidenceValue("nested", { a: 1 }), '{"a":1}');
});

test("metricHintFromArguments reads metric_column", () => {
  assert.equal(metricHintFromArguments({ metric_column: "Sales" }), "Sales");
});

test("metricHintFromArguments falls back to target_column", () => {
  assert.equal(metricHintFromArguments({ target_column: "Profit" }), "Profit");
});

test("metricHintFromArguments returns undefined when neither is present", () => {
  assert.equal(metricHintFromArguments({ query: "SELECT 1" }), undefined);
});
