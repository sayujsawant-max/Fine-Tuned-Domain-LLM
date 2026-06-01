/**
 * Synthetic sample filing excerpts for the demo.
 *
 * IMPORTANT: All text below is fabricated for demonstration purposes. It is NOT
 * copied from any real SEC filing and refers to a fictional company.
 */

export interface FilingSample {
  id: string;
  label: string;
  /** Suggested task type to pair with this sample. */
  suggestedTaskType: string;
  text: string;
}

export const FILING_SAMPLES: readonly FilingSample[] = [
  {
    id: "risk_factors",
    label: "Risk Factors (sample)",
    suggestedTaskType: "risk_summary",
    text: `Item 1A. Risk Factors (illustrative, fictional company "Northwind Components, Inc.").

Our business depends on a concentrated set of suppliers for specialized semiconductor components. A disruption at any of these suppliers — whether from natural disaster, geopolitical tension, or capacity constraints — could materially delay our production and harm operating results. We face intense competition from larger, better-capitalized firms that may price aggressively or bundle products. Changes in trade policy and export controls could restrict our ability to sell into key markets. We are also subject to evolving data-privacy regulation across multiple jurisdictions; non-compliance could result in fines and reputational damage. Finally, a significant portion of our revenue is denominated in foreign currencies, exposing us to exchange-rate volatility.`,
  },
  {
    id: "mda",
    label: "MD&A (sample)",
    suggestedTaskType: "mda_explanation",
    text: `Item 7. Management's Discussion and Analysis (illustrative, fictional company).

Total revenue increased 14% year over year, driven primarily by higher unit volumes in our cloud-infrastructure segment and improved average selling prices. Gross margin expanded approximately 180 basis points as manufacturing efficiencies offset elevated logistics costs. Operating expenses grew more slowly than revenue as we maintained disciplined hiring. We continue to invest in research and development, which rose 9% as we expand our next-generation product roadmap. Management believes liquidity remains strong, with cash and equivalents sufficient to fund operations and planned capital expenditures for at least the next twelve months.`,
  },
  {
    id: "metrics",
    label: "Financial Metrics (sample)",
    suggestedTaskType: "metric_extraction",
    text: `Selected Financial Data (illustrative, fictional company, in millions except per-share data).

Net revenue was $4,820 in fiscal 2024 compared to $4,230 in fiscal 2023. Gross profit was $2,180, representing a gross margin of 45.2%. Operating income was $760, and net income was $590, or $2.41 per diluted share. Cash provided by operating activities was $910. Total assets were $7,450 and total stockholders' equity was $3,980 at year end. The company repurchased $300 of common stock and paid $120 in dividends during the year.`,
  },
] as const;

/** Return a sample by id, if it exists. */
export function getSample(id: string): FilingSample | undefined {
  return FILING_SAMPLES.find((s) => s.id === id);
}
