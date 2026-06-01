import { describe, expect, it } from "vitest";
import { getMockAnalysisResponse } from "@/lib/mockResponses";

const baseRequest = {
  question: "Summarize the risks.",
  filing_excerpt: "The company faces supply chain disruption and competition.",
};

describe("getMockAnalysisResponse", () => {
  it("returns a non-empty answer", () => {
    const res = getMockAnalysisResponse({ ...baseRequest, task_type: "risk_summary" });
    expect(res.answer.length).toBeGreaterThan(0);
  });

  it("includes the demo marker and a disclaimer by default", () => {
    const res = getMockAnalysisResponse({ ...baseRequest, task_type: "risk_summary" });
    expect(res.answer).toContain("Demo mode response");
    expect(res.disclaimer).not.toBeNull();
    expect(res.demo_mode).toBe(true);
  });

  it("preserves the task type", () => {
    const res = getMockAnalysisResponse({ ...baseRequest, task_type: "metric_extraction" });
    expect(res.task_type).toBe("metric_extraction");
  });

  it("omits the disclaimer when include_disclaimer is false", () => {
    const res = getMockAnalysisResponse({
      ...baseRequest,
      task_type: "filing_qa",
      include_disclaimer: false,
    });
    expect(res.disclaimer).toBeNull();
  });
});
