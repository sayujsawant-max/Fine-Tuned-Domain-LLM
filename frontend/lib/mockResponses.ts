/**
 * Deterministic demo-mode responses.
 *
 * Used by the proxy route when demo mode is on and the backend is unreachable,
 * so the UI is fully explorable without any running services. The output is
 * obviously synthetic and clearly labelled.
 */

import type { ChatRequest, ChatResponse } from "@/lib/api";
import { getTaskType } from "@/lib/taskTypes";

const DEMO_DISCLAIMER =
  "FinSage-7B is not a licensed financial advisor. Outputs are not investment " +
  "recommendations. Always verify responses against the original filing.";

const DEMO_MARKER =
  "Demo mode response. Connect the FastAPI backend for real FinSage-7B output.";

/** Task-specific synthetic answer bodies. */
const TEMPLATES: Record<string, string> = {
  risk_summary:
    "Key risks in this excerpt center on supplier concentration, intense competition from larger firms, exposure to trade-policy and export-control changes, evolving data-privacy regulation, and foreign-currency volatility.",
  mda_explanation:
    "Management attributes revenue growth to higher volumes and improved pricing, with margin expansion from manufacturing efficiencies. Spending discipline and continued R&D investment are emphasized, alongside a statement that liquidity is sufficient for the next twelve months.",
  metric_extraction:
    "Net revenue: $4,820M (vs $4,230M prior year); gross profit: $2,180M (45.2% margin); operating income: $760M; net income: $590M ($2.41 diluted EPS); operating cash flow: $910M.",
  yoy_comparison:
    "Year over year, revenue rose ~14% and gross margin expanded ~180 bps. Operating expenses grew more slowly than revenue, and R&D increased ~9%, indicating operating leverage with continued investment.",
  business_risk_identification:
    "Identified business risks: (1) concentrated supplier base, (2) competitive pricing pressure, (3) trade-policy/export-control restrictions, (4) data-privacy compliance exposure, and (5) currency risk on foreign-denominated revenue.",
  revenue_driver_explanation:
    "The primary revenue drivers are higher unit volumes in the cloud-infrastructure segment and improved average selling prices, supported by manufacturing efficiencies that aided gross margin.",
  filing_qa:
    "Based on the excerpt, the company reported broad-based growth with improving profitability and stated that its liquidity position is sufficient to fund operations and planned capital expenditures.",
  analyst_summary:
    "Solid double-digit revenue growth with margin expansion and disciplined cost control. Continued R&D investment supports the roadmap; liquidity is described as strong. Watch supplier concentration and FX exposure.",
  outlook_classification:
    "Outlook: Positive. The language emphasizes growth, margin expansion, and adequate liquidity, with no indication of material near-term deterioration in the provided text.",
  hallucination_detection:
    "No unsupported claims detected relative to the excerpt. Any figure or assertion not present in the provided text (e.g., specific guidance ranges) should be treated as unsupported.",
};

/**
 * Build a deterministic mock analysis response for demo mode.
 *
 * @param request - The chat request.
 * @returns A synthetic, clearly-labelled chat response.
 */
export function getMockAnalysisResponse(request: ChatRequest): ChatResponse {
  const taskType = request.task_type ?? null;
  const meta = taskType ? getTaskType(taskType) : undefined;
  const body =
    (taskType && TEMPLATES[taskType]) ??
    "This is a representative grounded answer derived only from the provided filing excerpt.";

  const label = meta ? `[${meta.label}] ` : "";
  const includeDisclaimer = request.include_disclaimer !== false;

  const answer = `${label}${body}\n\n${DEMO_MARKER}${
    includeDisclaimer ? `\n\n${DEMO_DISCLAIMER}` : ""
  }`;

  return {
    answer,
    model: "finsage-7b (demo)",
    task_type: taskType,
    disclaimer: includeDisclaimer ? DEMO_DISCLAIMER : null,
    request_id: "demo-" + Math.random().toString(36).slice(2, 10),
    latency_ms: 420,
    demo_mode: true,
  };
}
