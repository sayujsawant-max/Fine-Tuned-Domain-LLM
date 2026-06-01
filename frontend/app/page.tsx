"use client";

import { Github, GitCompare, Layers, ShieldCheck } from "lucide-react";
import { useMemo, useState } from "react";
import { analyzeFiling, ApiError, type ChatResponse } from "@/lib/api";
import { appName, demoMode } from "@/lib/config";
import { getSample } from "@/lib/samples";
import { getDefaultQuestion, getTaskType, TASK_TYPES } from "@/lib/taskTypes";
import { AnalyzeButton } from "@/components/AnalyzeButton";
import { ArchitectureFlow } from "@/components/ArchitectureFlow";
import { BenchmarkSummary } from "@/components/BenchmarkSummary";
import { FilingInput } from "@/components/FilingInput";
import { QuestionInput } from "@/components/QuestionInput";
import { ResponseComparison } from "@/components/ResponseComparison";
import { ResponsePanel } from "@/components/ResponsePanel";
import { StatusBadge } from "@/components/StatusBadge";
import { TaskSelector } from "@/components/TaskSelector";

const DEFAULT_TASK = TASK_TYPES[0].value;

export default function HomePage() {
  const [filing, setFiling] = useState("");
  const [selectedSample, setSelectedSample] = useState("");
  const [taskType, setTaskType] = useState(DEFAULT_TASK);
  const [question, setQuestion] = useState(getDefaultQuestion(DEFAULT_TASK));
  const [compareMode, setCompareMode] = useState(false);

  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const suggestedQuestion = useMemo(() => getDefaultQuestion(taskType), [taskType]);

  function handleSampleChange(sampleId: string) {
    setSelectedSample(sampleId);
    if (!sampleId) return;
    const sample = getSample(sampleId);
    if (!sample) return;
    setFiling(sample.text);
    setTaskType(sample.suggestedTaskType);
    setQuestion(getDefaultQuestion(sample.suggestedTaskType));
  }

  function handleTaskChange(next: string) {
    setTaskType(next);
    // Update the question only if the user hasn't customised it.
    if (!question.trim() || question === getDefaultQuestion(taskType)) {
      setQuestion(getDefaultQuestion(next));
    }
  }

  async function handleAnalyze() {
    setLoading(true);
    setError(null);
    setResponse(null);
    try {
      const result = await analyzeFiling({
        question: question.trim(),
        filing_excerpt: filing.trim(),
        task_type: taskType,
        max_tokens: 256,
        temperature: 0.0,
        include_disclaimer: true,
      });
      setResponse(result);
    } catch (err) {
      setError(
        err instanceof ApiError ? err : new ApiError("Unexpected error.", 0),
      );
    } finally {
      setLoading(false);
    }
  }

  const canAnalyze = filing.trim().length > 0 && question.trim().length > 0;
  const activeTaskLabel = getTaskType(taskType)?.label ?? taskType;

  return (
    <main className="mx-auto max-w-6xl px-4 py-10 sm:px-6 lg:py-14">
      {/* Hero */}
      <header className="mb-10 text-center">
        <div className="mb-4 flex justify-center gap-2">
          <StatusBadge tone="brand">QLoRA + SEC Filings + vLLM</StatusBadge>
          {demoMode ? <StatusBadge tone="warning">Demo mode</StatusBadge> : null}
        </div>
        <h1 className="bg-gradient-to-b from-white to-slate-400 bg-clip-text text-4xl font-bold tracking-tight text-transparent sm:text-5xl">
          {appName}
        </h1>
        <p className="mx-auto mt-4 max-w-2xl text-balance text-base text-slate-400 sm:text-lg">
          Fine-tuned Mistral-7B for financial filing analysis. Paste a 10-K / 10-Q
          excerpt, pick a task, and get a <span className="text-brand-300">grounded</span>{" "}
          answer — not generic boilerplate, and never investment advice.
        </p>
        <div className="mt-5 flex flex-wrap justify-center gap-4 text-xs text-slate-500">
          <span className="flex items-center gap-1.5">
            <Layers className="h-3.5 w-3.5" /> 10 filing-analysis tasks
          </span>
          <span className="flex items-center gap-1.5">
            <ShieldCheck className="h-3.5 w-3.5" /> Grounded &amp; disclaimer-bound
          </span>
          <a
            href="https://github.com/sayujsawant-max/Fine-Tuned-Domain-LLM"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1.5 hover:text-slate-300"
          >
            <Github className="h-3.5 w-3.5" /> Source
          </a>
        </div>
      </header>

      <div className="grid gap-6 lg:grid-cols-5">
        {/* Input column */}
        <section className="card space-y-5 lg:col-span-3" aria-label="Filing analysis input">
          <FilingInput
            value={filing}
            onChange={setFiling}
            selectedSample={selectedSample}
            onSampleChange={handleSampleChange}
          />
          <div className="grid gap-4 sm:grid-cols-2">
            <TaskSelector value={taskType} onChange={handleTaskChange} />
            <div className="flex items-end">
              <label className="flex cursor-pointer items-center gap-2 text-sm text-slate-300">
                <input
                  type="checkbox"
                  checked={compareMode}
                  onChange={(e) => setCompareMode(e.target.checked)}
                  className="h-4 w-4 rounded border-white/20 bg-ink-800 text-brand-500 focus:ring-brand-500"
                />
                <GitCompare className="h-4 w-4 text-slate-400" />
                Compare vs base Mistral
              </label>
            </div>
          </div>
          <QuestionInput
            value={question}
            onChange={setQuestion}
            suggestedQuestion={suggestedQuestion}
          />
          <AnalyzeButton onClick={handleAnalyze} loading={loading} disabled={!canAnalyze} />
          {!canAnalyze ? (
            <p className="text-center text-xs text-slate-500">
              Add a filing excerpt and a question to enable analysis.
            </p>
          ) : null}
        </section>

        {/* Output column */}
        <section className="card lg:col-span-2" aria-label="FinSage-7B response">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-400">
            {activeTaskLabel} result
          </h2>
          <ResponsePanel response={response} loading={loading} error={error} />
        </section>
      </div>

      {/* Comparison */}
      {compareMode ? (
        <section className="card mt-6" aria-label="Model comparison">
          <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-200">
            <GitCompare className="h-4 w-4 text-brand-400" /> Base Mistral vs FinSage-7B
          </h2>
          <ResponseComparison finsage={response} />
        </section>
      ) : null}

      {/* Benchmark + architecture */}
      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <section className="card" aria-label="Benchmark summary">
          <BenchmarkSummary />
        </section>
        <section className="card" aria-label="Architecture flow">
          <ArchitectureFlow />
          <p className="mt-4 text-xs leading-relaxed text-slate-500">
            The browser calls a server-side Next.js proxy (<code>/api/chat</code>), which
            injects the API key and forwards to the FastAPI wrapper. The API secret is
            never exposed to the browser.
          </p>
        </section>
      </div>

      {/* Disclaimer footer */}
      <footer className="mt-10 border-t border-white/10 pt-6 text-center text-xs leading-relaxed text-slate-500">
        <p className="mx-auto max-w-3xl">
          <strong className="text-slate-400">Not financial advice.</strong> FinSage-7B is
          not a licensed financial advisor; outputs are not investment recommendations and
          may contain errors. Always verify against the original filing. Sample filing text
          shown in this demo is fabricated and not copied from any real SEC filing.
        </p>
      </footer>
    </main>
  );
}
