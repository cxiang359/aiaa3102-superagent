"""Aggregate all run summaries under runs/ into a compact markdown report.

Usage: python -m superagent.report [run_root]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .config import PROJECT_ROOT


def collect_results(run_root: str | Path) -> list[dict]:
    run_root = Path(run_root)
    records = []
    for summary_path in sorted(run_root.glob("*/run_*/logs/run_summary.json")):
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        cost = data.get("cost", {})
        records.append(
            {
                "task": data.get("task_name"),
                "run": summary_path.parent.parent.name,
                "status": data.get("status"),
                "metric": data.get("metric"),
                "best_val": data.get("best_val_metric"),
                "test": data.get("test_metric"),
                "iters": len(data.get("history", [])),
                "n_calls": cost.get("n_calls"),
                "in_tokens": cost.get("input_tokens"),
                "out_tokens": cost.get("output_tokens"),
                "cost_rmb": cost.get("spent_rmb"),
                "run_dir": str(summary_path.parent.parent),
            }
        )
    return records


def render_markdown(records: list[dict]) -> str:
    lines = [
        "| task | run | status | val | test | iters | LLM calls | in tok | out tok | cost ¥ |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    total_cost = 0.0
    for r in records:
        total_cost += r["cost_rmb"] or 0.0
        lines.append(
            f"| {r['task']} | {r['run']} | {r['status']} | "
            f"{r['best_val'] if r['best_val'] is not None else '—'} | "
            f"{r['test'] if r['test'] is not None else '—'} | {r['iters']} | "
            f"{r['n_calls']} | {r['in_tokens']} | {r['out_tokens']} | {r['cost_rmb']} |"
        )
    lines.append("")
    lines.append(f"**Total recorded cost: ¥{total_cost:.4f}**")
    return "\n".join(lines)


if __name__ == "__main__":
    run_root = sys.argv[1] if len(sys.argv) > 1 else PROJECT_ROOT / "runs"
    recs = collect_results(run_root)
    print(render_markdown(recs))
