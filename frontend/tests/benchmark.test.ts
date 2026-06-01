import { describe, expect, it } from "vitest";
import { parseBenchmark, SAMPLE_BENCHMARK } from "@/lib/benchmark";

describe("benchmark", () => {
  it("provides four sample metrics with deltas", () => {
    expect(SAMPLE_BENCHMARK).toHaveLength(4);
    for (const m of SAMPLE_BENCHMARK) {
      expect(m.label.length).toBeGreaterThan(0);
      expect(m.value.length).toBeGreaterThan(0);
    }
  });

  it("parses an array of metric objects", () => {
    const parsed = parseBenchmark([
      { label: "Filing Q&A F1", value: "0.71", delta: "+0.14" },
      { label: "Faithfulness", value: 0.92 },
    ]);
    expect(parsed).toHaveLength(2);
    expect(parsed[0].label).toBe("Filing Q&A F1");
    expect(parsed[1].value).toBe("0.92");
  });

  it("parses a flat label→value object", () => {
    const parsed = parseBenchmark({ mean_absolute_delta: 0.1764, metrics_improved: 5 });
    expect(parsed).toHaveLength(2);
    expect(parsed.map((m) => m.label)).toContain("mean_absolute_delta");
  });

  it("returns an empty list for unusable input", () => {
    expect(parseBenchmark(null)).toEqual([]);
    expect(parseBenchmark("nope")).toEqual([]);
  });
});
