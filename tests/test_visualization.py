"""Tests for chart generation (degrades gracefully without matplotlib)."""

from __future__ import annotations

import importlib.util

from finsage.evaluation import visualization

_HAS_MPL = importlib.util.find_spec("matplotlib") is not None

COMPARISON = {
    "overall_comparison": {
        "token_f1": {"baseline": 0.4, "finetuned": 0.6, "absolute_delta": 0.2},
        "rouge_l": {"baseline": 0.3, "finetuned": 0.45, "absolute_delta": 0.15},
    },
    "by_task_comparison": {
        "filing_qa": {"token_f1": {"baseline": 0.4, "finetuned": 0.6, "absolute_delta": 0.2}},
    },
}
RESULTS = {"count_by_task": {"filing_qa": 3, "risk_summary": 2}}


def test_chart_functions_do_not_crash(tmp_path):
    """Chart functions return a Path (mpl present) or None (absent), never raise."""
    out1 = visualization.plot_overall_metric_comparison(COMPARISON, tmp_path / "a.png")
    out2 = visualization.plot_task_metric_deltas(COMPARISON, tmp_path / "b.png")
    out3 = visualization.plot_task_counts(RESULTS, tmp_path / "c.png")

    if _HAS_MPL:
        assert out1.exists() and out2.exists() and out3.exists()
    else:
        assert out1 is None and out2 is None and out3 is None


def test_empty_comparison_returns_none(tmp_path):
    """Empty inputs yield None without raising."""
    assert visualization.plot_overall_metric_comparison({}, tmp_path / "x.png") is None
    assert visualization.plot_task_counts({}, tmp_path / "y.png") is None
