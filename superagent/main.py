"""Unified CLI entry point for the SuperAgent.

Usage:
  python run_agent.py --task-dir <path> --task-name <name> [--metric accuracy]
                      [--model deepseek-chat] [--max-iters 5] [--budget-rmb 20]
"""

from __future__ import annotations

import argparse
import json
import sys

from .agent import SuperAgent
from .config import BUDGET_RMB_PER_TASK, DEFAULT_MAX_ITERS, DEFAULT_MODEL, PROJECT_ROOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="General Algorithm SuperAgent")
    parser.add_argument("--task-dir", help="Path to the released task folder")
    parser.add_argument("--task-name", help="Short name for logs/results")
    parser.add_argument("--metric", default="accuracy", help="Selection metric (higher better)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument("--budget-rmb", type=float, default=BUDGET_RMB_PER_TASK)
    parser.add_argument(
        "--run-root",
        default=str(PROJECT_ROOT / "runs"),
        help="Directory where run outputs are stored",
    )
    parser.add_argument("--backend", choices=("legacy", "campus", "offline"), default="legacy")
    parser.add_argument("--strategy", choices=("single", "multi"), default="single")
    parser.add_argument("--max-trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-column")
    parser.add_argument("--key-file")
    parser.add_argument("--campus-direct", action="store_true", help="Ignore system proxy for campus API on authorized direct networks.")
    parser.add_argument("--feature-search", action="store_true", help="Allow the campus model to compose validated tabular feature operations.")
    parser.add_argument("--oof-ensemble", action="store_true", help="Evaluate one additional OOF-weighted ensemble beyond --max-trials.")
    parser.add_argument("--resume-run", help="Resume an interrupted structured search without resetting its API ledger.")
    parser.add_argument("--finalize", action="store_true", help="After a structured search, evaluate even a saved fallback if the API failed.")
    parser.add_argument("--evaluate-run", help="Evaluate an already frozen structured run on test")
    parser.add_argument("--compare-and-evaluate", action="store_true", help="Compare both strategies on validation, then evaluate only the winner on test")
    args = parser.parse_args(argv)

    if args.resume_run:
        from .experiment import finalize_run, resume_search

        if args.evaluate_run or args.compare_and_evaluate or args.task_dir or args.task_name:
            parser.error("--resume-run uses the original search configuration.")
        summary = resume_search(args.resume_run, key_file=args.key_file, direct=args.campus_direct)
        if args.finalize and summary["best"] is not None:
            summary["final"] = finalize_run(summary["run_dir"])
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["status"] == "success" else 1

    if args.evaluate_run:
        from .experiment import finalize_run

        if args.task_dir is not None or args.task_name is not None:
            parser.error("--evaluate-run uses the task recorded in the frozen run.")
        summary = finalize_run(args.evaluate_run)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if not args.task_dir or not args.task_name:
        parser.error("--task-dir and --task-name are required for search.")
    if args.backend != "legacy":
        from .experiment import compare_and_evaluate, finalize_run, run_search
        from .task_data import load_development

        from .metrics import METRICS

        if args.metric not in METRICS:
            parser.error(f"Choose a structured metric from {METRICS}.")
        if args.model != DEFAULT_MODEL:
            parser.error("The structured campus backend uses a fixed model. Omit --model.")
        task = load_development(args.task_dir, args.label_column)
        settings = {
            "backend": args.backend, "max_trials": args.max_trials,
            "budget": args.budget_rmb, "seed": args.seed, "key_file": args.key_file,
            "metric": args.metric, "direct": args.campus_direct, "feature_search": args.feature_search,
            "oof_ensemble": args.oof_ensemble,
        }
        if args.compare_and_evaluate:
            if args.finalize:
                parser.error("--compare-and-evaluate already includes finalization.")
            summary = compare_and_evaluate(task, args.run_root, args.task_name, **settings)
        else:
            summary = run_search(task, args.run_root, args.task_name, strategy=args.strategy, **settings)
            if args.finalize and summary["best"] is not None:
                summary["final"] = finalize_run(summary["run_dir"])
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["status"] == "success" else 1

    if args.compare_and_evaluate or args.finalize:
        parser.error("Structured finalization requires --backend campus or offline.")

    agent = SuperAgent(
        task_dir=args.task_dir,
        task_name=args.task_name,
        metric=args.metric,
        model=args.model,
        budget_rmb=args.budget_rmb,
        run_root=args.run_root,
        max_iters=args.max_iters,
    )
    summary = agent.run()

    print("\n===== RUN SUMMARY =====")
    for key in ("status", "best_val_metric", "test_metric", "cost", "out_of_budget"):
        if key in summary:
            print(f"{key}: {summary[key]}")
    return 0 if summary.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
