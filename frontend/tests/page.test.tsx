import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import HomePage from "@/app/page";

const CHAT_RESPONSE = {
  answer: "Mocked grounded answer about supply chain risk.",
  model: "finsage-7b",
  task_type: "risk_summary",
  disclaimer: "Not financial advice.",
  request_id: "req-int-1",
  latency_ms: 210,
};

function mockFetch() {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.includes("/api/benchmark")) {
      return new Response(JSON.stringify({ source: "sample", metrics: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    if (url.includes("/api/chat")) {
      return new Response(JSON.stringify(CHAT_RESPONSE), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    return new Response("{}", { status: 200 });
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("HomePage analyze flow", () => {
  it("submits a filing + question and renders the answer", async () => {
    const fetchSpy = mockFetch();
    render(<HomePage />);

    // The question is pre-filled from the default task; add a filing excerpt.
    fireEvent.change(screen.getByLabelText("Filing excerpt"), {
      target: { value: "The company faces supply chain disruption and competition." },
    });

    fireEvent.click(screen.getByRole("button", { name: /Analyze Filing/i }));

    // The mocked answer appears in the response panel.
    expect(await screen.findByText(/Mocked grounded answer/)).toBeInTheDocument();
    expect(screen.getByText("req-int-1")).toBeInTheDocument();

    // The proxy route — not the backend — was called.
    const calledChat = fetchSpy.mock.calls.some(([u]) => String(u).includes("/api/chat"));
    expect(calledChat).toBe(true);
  });

  it("disables analysis until a filing excerpt is provided", () => {
    mockFetch();
    render(<HomePage />);
    // No filing text yet → the button is disabled.
    expect(screen.getByRole("button", { name: /Analyze Filing/i })).toBeDisabled();
  });
});
