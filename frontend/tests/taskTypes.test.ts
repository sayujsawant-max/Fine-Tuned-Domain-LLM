import { describe, expect, it } from "vitest";
import {
  getDefaultQuestion,
  isValidTaskType,
  TASK_TYPES,
} from "@/lib/taskTypes";

const EXPECTED_VALUES = [
  "risk_summary",
  "mda_explanation",
  "metric_extraction",
  "yoy_comparison",
  "business_risk_identification",
  "revenue_driver_explanation",
  "filing_qa",
  "analyst_summary",
  "outlook_classification",
  "hallucination_detection",
];

describe("taskTypes", () => {
  it("defines exactly the 10 supported task types", () => {
    expect(TASK_TYPES).toHaveLength(10);
    expect(TASK_TYPES.map((t) => t.value).sort()).toEqual([...EXPECTED_VALUES].sort());
  });

  it("provides a non-empty label, description, and default question for each", () => {
    for (const t of TASK_TYPES) {
      expect(t.label.length).toBeGreaterThan(0);
      expect(t.description.length).toBeGreaterThan(0);
      expect(t.defaultQuestion.length).toBeGreaterThan(0);
    }
  });

  it("returns a default question for a known task type", () => {
    expect(getDefaultQuestion("risk_summary").length).toBeGreaterThan(0);
  });

  it("returns an empty string for an unknown task type", () => {
    expect(getDefaultQuestion("nope")).toBe("");
  });

  it("validates task types", () => {
    expect(isValidTaskType("filing_qa")).toBe(true);
    expect(isValidTaskType("not_a_task")).toBe(false);
    expect(isValidTaskType(null)).toBe(false);
    expect(isValidTaskType(undefined)).toBe(false);
  });
});
