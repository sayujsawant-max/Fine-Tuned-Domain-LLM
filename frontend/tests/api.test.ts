import { afterEach, describe, expect, it, vi } from "vitest";
import { analyzeFiling, ApiError } from "@/lib/api";

const request = {
  question: "Summarize the risks.",
  filing_excerpt: "Supply chain disruption and competition.",
  task_type: "risk_summary",
};

const successBody = {
  answer: "Key risks: supply chain and competition.",
  model: "finsage-7b",
  task_type: "risk_summary",
  disclaimer: "Not financial advice.",
  request_id: "req-123",
  latency_ms: 321,
};

afterEach(() => {
  vi.restoreAllMocks();
});

describe("analyzeFiling", () => {
  it("calls the local /api/chat proxy with a JSON POST", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(
        new Response(JSON.stringify(successBody), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      );

    const res = await analyzeFiling(request);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/chat");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body))).toMatchObject({ task_type: "risk_summary" });
    expect(res.answer).toBe(successBody.answer);
    expect(res.request_id).toBe("req-123");
  });

  it("throws a typed ApiError on a non-200 response", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({ error: "vllm_unavailable", detail: "down", request_id: "r1" }),
        { status: 503, headers: { "Content-Type": "application/json" } },
      ),
    );

    await expect(analyzeFiling(request)).rejects.toMatchObject({
      name: "ApiError",
      status: 503,
      detail: "down",
      requestId: "r1",
    });
  });

  it("wraps a network failure as an ApiError", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("network down"));
    await expect(analyzeFiling(request)).rejects.toBeInstanceOf(ApiError);
  });
});
