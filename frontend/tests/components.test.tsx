import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { ChatResponse } from "@/lib/api";
import { FILING_SAMPLES } from "@/lib/samples";
import { TASK_TYPES } from "@/lib/taskTypes";
import { CopyButton } from "@/components/CopyButton";
import { ErrorAlert } from "@/components/ErrorAlert";
import { FilingInput } from "@/components/FilingInput";
import { ResponsePanel } from "@/components/ResponsePanel";
import { TaskSelector } from "@/components/TaskSelector";

describe("FilingInput", () => {
  it("renders all sample options", () => {
    render(
      <FilingInput value="" onChange={vi.fn()} selectedSample="" onSampleChange={vi.fn()} />,
    );
    for (const sample of FILING_SAMPLES) {
      expect(screen.getByRole("option", { name: sample.label })).toBeInTheDocument();
    }
  });
});

describe("TaskSelector", () => {
  it("renders all 10 task types as options", () => {
    render(<TaskSelector value={TASK_TYPES[0].value} onChange={vi.fn()} />);
    for (const task of TASK_TYPES) {
      expect(screen.getByRole("option", { name: task.label })).toBeInTheDocument();
    }
  });
});

describe("ResponsePanel", () => {
  const response: ChatResponse = {
    answer: "These are the key risks.",
    model: "finsage-7b",
    task_type: "risk_summary",
    disclaimer: "Not financial advice.",
    request_id: "req-xyz",
    latency_ms: 250,
  };

  it("renders the answer, model, and request id", () => {
    render(<ResponsePanel response={response} loading={false} error={null} />);
    expect(screen.getByText("These are the key risks.")).toBeInTheDocument();
    expect(screen.getByText("finsage-7b")).toBeInTheDocument();
    expect(screen.getByText("req-xyz")).toBeInTheDocument();
  });

  it("shows an empty state when there is no response", () => {
    render(<ResponsePanel response={null} loading={false} error={null} />);
    expect(screen.getByText(/Analyze Filing/i)).toBeInTheDocument();
  });
});

describe("ErrorAlert", () => {
  it("renders the error message with role=alert", () => {
    render(<ErrorAlert message="Something went wrong" />);
    expect(screen.getByRole("alert")).toHaveTextContent("Something went wrong");
  });
});

describe("CopyButton", () => {
  it("renders with an accessible label", () => {
    render(<CopyButton value="hello" />);
    expect(screen.getByRole("button", { name: /copy/i })).toBeInTheDocument();
  });
});
