/**
 * Centralised metadata for the ten FinSage-7B task types.
 *
 * The `value` strings must match the backend's accepted `task_type` values
 * (see `finsage.config.SUPPORTED_TASK_TYPES`).
 */

export interface TaskType {
  /** Backend task_type value. */
  value: string;
  /** Human-friendly label. */
  label: string;
  /** Short description shown under the selector. */
  description: string;
  /** Suggested default question for this task. */
  defaultQuestion: string;
}

export const TASK_TYPES: readonly TaskType[] = [
  {
    value: "risk_summary",
    label: "Risk Summary",
    description: "Summarise the key risk factors disclosed in the excerpt.",
    defaultQuestion: "Summarize the main risk factors described in this excerpt.",
  },
  {
    value: "mda_explanation",
    label: "MD&A Explanation",
    description: "Explain management's discussion and analysis in plain terms.",
    defaultQuestion: "Explain what management is highlighting in this MD&A section.",
  },
  {
    value: "metric_extraction",
    label: "Metric Extraction",
    description: "Extract reported financial metrics and figures.",
    defaultQuestion: "Extract the key financial metrics mentioned in this excerpt.",
  },
  {
    value: "yoy_comparison",
    label: "YoY Comparison",
    description: "Compare year-over-year changes in the reported figures.",
    defaultQuestion: "Compare the year-over-year changes described in this excerpt.",
  },
  {
    value: "business_risk_identification",
    label: "Business Risk Identification",
    description: "Identify specific business risks the company faces.",
    defaultQuestion: "Identify the specific business risks mentioned in this excerpt.",
  },
  {
    value: "revenue_driver_explanation",
    label: "Revenue Driver Explanation",
    description: "Explain the primary drivers of revenue.",
    defaultQuestion: "What are the main revenue drivers described in this excerpt?",
  },
  {
    value: "filing_qa",
    label: "Filing Q&A",
    description: "Answer a direct question grounded in the filing text.",
    defaultQuestion: "What does this excerpt say about the company's performance?",
  },
  {
    value: "analyst_summary",
    label: "Analyst Summary",
    description: "Produce a concise analyst-style summary.",
    defaultQuestion: "Provide a concise analyst summary of this excerpt.",
  },
  {
    value: "outlook_classification",
    label: "Outlook Classification",
    description: "Classify the forward-looking outlook (positive/neutral/negative).",
    defaultQuestion: "Classify the overall outlook conveyed in this excerpt.",
  },
  {
    value: "hallucination_detection",
    label: "Hallucination Detection",
    description: "Flag claims not supported by the provided excerpt.",
    defaultQuestion:
      "Are there any claims here that would not be supported by the filing text?",
  },
] as const;

const TASK_TYPE_VALUES = new Set(TASK_TYPES.map((t) => t.value));

/**
 * Return whether a string is a valid/supported task type value.
 *
 * @param taskType - Candidate task type value.
 */
export function isValidTaskType(taskType: string | null | undefined): boolean {
  return typeof taskType === "string" && TASK_TYPE_VALUES.has(taskType);
}

/**
 * Return the suggested default question for a task type.
 *
 * @param taskType - Task type value.
 * @returns The default question, or an empty string for an unknown task type.
 */
export function getDefaultQuestion(taskType: string): string {
  return TASK_TYPES.find((t) => t.value === taskType)?.defaultQuestion ?? "";
}

/**
 * Return the full metadata for a task type, if known.
 *
 * @param taskType - Task type value.
 */
export function getTaskType(taskType: string): TaskType | undefined {
  return TASK_TYPES.find((t) => t.value === taskType);
}
