<!--
Reusable Markdown template for the FinSage-7B benchmark report.

The report is normally generated programmatically by
`finsage.reporting.report_builder.BenchmarkReportBuilder`, which renders the
same sections directly. This template documents the canonical structure and the
placeholder names a template-based renderer would substitute.
-->

# {{ project_name }} — Benchmark Report

_Generated {{ generated_at }} · report v{{ report_version }}_

{{ mock_banner }}

---

## Executive Summary

{{ executive_summary }}

## Project Overview

{{ project_overview }}

## Why Financial Filings

{{ why_financial_filings }}

## Dataset Summary

{{ dataset_table }}

## Training Setup

{{ training_table }}

## Evaluation Methodology

{{ evaluation_methodology }}

## Overall Results

{{ overall_metrics_table }}

## Task-wise Results

{{ task_metrics_table }}

## Hallucination and Faithfulness Analysis

{{ hallucination_analysis }}

## Latency and Deployment Summary

{{ latency_table }}

## Qualitative Examples

{{ qualitative_examples }}

## Error Analysis and Regressions

{{ error_analysis }}

## Limitations

{{ limitations }}

## Financial Safety Disclaimer

{{ disclaimer }}

## Reproducibility Guide

{{ reproducibility }}

## Appendix

{{ appendix }}
